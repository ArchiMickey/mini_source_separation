import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import numpy as np
import librosa
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb


from mss.models.fourier import Fourier


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


class BSRoformer(Fourier):
    def __init__(
        self,
        sr: int = 44100,
        mel_bins: int = 256,
        n_fft: int = 2048,
        hop_length: int = 441,
        input_channels: int = 2,
        patch_size: list = [4, 1],
        depths: list = [2, 2],
        f_q_strides: list = [2, 1],
        dim: int = 384,
        dim_mult: float = 2.0,
        dim_heads: int = 64,
    ):
        super().__init__(n_fft, hop_length)

        self.input_channels = input_channels
        self.f_q_strides = f_q_strides
        self.dim = dim
        self.dim_mult = dim_mult
        self.n_heads = dim // dim_heads
        self.depths = depths

        self.cmplx_num = 2

        self.head_dim = self.dim // self.n_heads

        self.patch_size = patch_size
        out_channels = mel_bins // self.patch_size[0]

        self.stft_to_image = StftToImage(
            in_channels=self.input_channels * self.cmplx_num,
            sr=sr,
            n_fft=n_fft,
            mel_bins=mel_bins,
            out_channels=out_channels,
        )

        self.fc_in = nn.Linear(
            in_features=out_channels * np.prod(self.patch_size), out_features=self.dim
        )

        rotary_emb_t = RotaryEmbedding(dim=self.head_dim)
        rotary_emb_f = RotaryEmbedding(dim=self.head_dim)

        self.stages = nn.ModuleList([])

        assert len(self.depths) == len(
            self.f_q_strides
        ), "Stages and f_q_strides must have the same length."

        assert self.f_q_strides[0] == 1, "The first f_q_stride must be 1."

        init_dim = dim
        dim_out = dim
        dims = [dim]
        for depth, f_q_stride in zip(self.depths, self.f_q_strides):
            stage = nn.ModuleList([])
            for i in range(depth):
                if i == 0:
                    dim_out = int(dim * (dim_mult if f_q_stride > 1 else 1))
                stage.append(
                    BSRoformerBlock(
                        dim=dim,
                        dim_out=dim_out,
                        dim_heads=dim_heads,
                        rotary_emb_t=rotary_emb_t,
                        rotary_emb_f=rotary_emb_f,
                        f_q_stride=f_q_stride if i == 0 else 1,
                    )
                )
                dim = dim_out
            dims.append(dim)
            self.stages.append(stage)

        self.ups = []
        pool_strides = [s for s in self.f_q_strides if s > 1]

        for s in pool_strides:
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        dim_out, dim_out // 2, kernel_size=(1, s), stride=(1, s)
                    ),
                    RMSNorm2d(dim_out // 2),
                    nn.SiLU(),
                )
            )
            dim_out = dim_out // 2
        self.ups = nn.ModuleList(self.ups)

        self.fc_out = nn.Linear(
            in_features=self.dim,
            out_features=out_channels * np.prod(self.patch_size),
        )

    def forward(self, mixture):
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)

        Constants:
            b: batch_size
            c: channels_num=2
            z: complex_num=2
        """

        # Complex spectrum.
        complex_sp = self.stft(mixture)
        # shape: (b, c, t, f)

        time_steps = complex_sp.shape[2]

        x = torch.view_as_real(complex_sp)
        # shape: (b, c, t, f, z)

        x = rearrange(x, "b c t f z -> b (c z) t f")

        x = self.stft_to_image.transform(x)
        # shape: (b, d, t, f)

        x = self.patchify(x)
        # shape: (b, d, t, f)

        hs = [x]
        for i, stage in enumerate(self.stages):
            for block in stage:
                # x = checkpoint(block, x)
                x = block(x)
            if i != len(self.stages) - 1:
                hs.append(x)
        
        for up in self.ups:
            x = up(x)
            x = x + hs.pop()

        x = self.unpatchify(x, time_steps)

        x = self.stft_to_image.inverse_transform(x)

        x = rearrange(x, "b (c z) t f -> b c t f z", c=self.input_channels)
        # shape: (b, c, t, f, z)

        mask = torch.view_as_complex(x.contiguous())
        # shape: (b, c, t, f)

        sep_stft = mask * complex_sp

        output = self.istft(sep_stft)
        # (b, c, samples_num)

        return output

    def patchify(self, x):

        B, C, T, Freq = x.shape
        patch_size_t = self.patch_size[0]
        pad_len = int(np.ceil(T / patch_size_t)) * patch_size_t - T
        x = F.pad(x, pad=(0, 0, 0, pad_len))

        t2, f2 = self.patch_size
        x = rearrange(x, "b d (t1 t2) (f1 f2) -> b t1 f1 (t2 f2 d)", t2=t2, f2=f2)
        x = self.fc_in(x)  # (b, t, f, d)
        x = rearrange(x, "b t f d -> b d t f")

        return x

    def unpatchify(self, x, time_steps):
        t2, f2 = self.patch_size
        x = rearrange(x, "b d t f -> b t f d")
        x = self.fc_out(x)  # (b, t, f, d)
        x = rearrange(x, "b t1 f1 (t2 f2 d) -> b d (t1 t2) (f1 f2)", t2=t2, f2=f2)

        x = x[:, :, 0:time_steps, :]

        return x


class BSRoformerBlock(nn.Module):
    def __init__(
        self, dim, dim_out, dim_heads, rotary_emb_t, rotary_emb_f, f_q_stride=1
    ):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim
        self.dim = dim
        self.n_heads = dim_out // dim_heads
        self.dim_out = dim_out

        self.block_t = TransformerBlock(
            dim=dim, dim_out=dim, dim_heads=dim_heads, rotary_emb=rotary_emb_t
        )
        self.block_f = TransformerBlock(
            dim=dim,
            dim_out=dim_out,
            dim_heads=dim_heads,
            rotary_emb=rotary_emb_f,
            q_stride=f_q_stride,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "b d t f -> (b f) t d")
        x = self.block_t(x)

        x = rearrange(x, "(b f) t d -> (b t) f d", b=batch_size)
        x = self.block_f(x)

        x = rearrange(x, "(b t) f d -> b d t f", b=batch_size)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        r"""https://github.com/meta-llama/llama/blob/main/llama/model.py"""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.weight
        return output


class RMSNorm2d(RMSNorm):
    def forward(self, x):
        t, f = x.shape[-2:]
        x = rearrange(x, "b c t f -> b (t f) c")
        norm_x = torch.mean(x**2, dim=[-1, -2], keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.weight
        output = rearrange(output, "b (t f) c -> b c t f", t=t, f=f)
        return output


class StftToImage(nn.Module):

    def __init__(
        self, in_channels: int, sr: float, n_fft: int, mel_bins: int, out_channels: int
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_fft = n_fft
        self.mel_bins = mel_bins

        melbanks = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=self.mel_bins - 2, norm=None
        )

        melbank_first = np.zeros(melbanks.shape[-1])
        melbank_first[0] = 1.0

        melbank_last = np.zeros(melbanks.shape[-1])
        idx = np.argmax(melbanks[-1])
        melbank_last[idx:] = 1.0 - melbanks[-1, idx:]

        melbanks = np.concatenate(
            [melbank_first[None, :], melbanks, melbank_last[None, :]], axis=0
        )

        sum_banks = np.sum(melbanks, axis=0)
        assert np.allclose(a=sum_banks, b=1.0)

        self.band_nets = nn.ModuleList([])
        self.inv_band_nets = nn.ModuleList([])
        self.indexes = []
        #
        for f in range(self.mel_bins):

            idxes = (melbanks[f] != 0).nonzero()[0]
            self.indexes.append(idxes)

            in_dim = len(idxes) * in_channels
            self.band_nets.append(nn.Linear(in_dim, out_channels))
            self.inv_band_nets.append(nn.Linear(out_channels, in_dim))

        #
        self.register_buffer(name="melbanks", tensor=torch.Tensor(melbanks))

    def transform(self, x):

        vs = []

        for f in range(self.mel_bins):

            idxes = self.indexes[f]

            bank = self.melbanks[f, idxes]  # (banks,)
            stft_bank = x[..., idxes]  # (b, c, t, banks)

            v = stft_bank * bank  # (b, c, t, banks)
            v = rearrange(v, "b c t q -> b t (c q)")

            v = self.band_nets[f](v)  # (b, t, d)
            vs.append(v)

        x = torch.stack(vs, dim=2)  # (b, t, f, d)
        x = rearrange(x, "b t f d -> b d t f")

        return x

    def inverse_transform(self, x):

        B, _, T, _ = x.shape
        y = torch.zeros(B, self.in_channels, T, self.n_fft // 2 + 1).to(x.device)

        for f in range(self.mel_bins):

            idxes = self.indexes[f]
            v = x[..., f]  # (b, d, t)
            v = rearrange(v, "b d t -> b t d")
            v = self.inv_band_nets[f](v)  # (b, t, d)
            v = rearrange(v, "b t (c q) -> b c t q", q=len(idxes))
            y[..., idxes] += v

        return y


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        dim_out: int,
        dim_heads: int,
        rotary_emb: RotaryEmbedding,
        q_stride=1,
    ):
        super().__init__()

        assert dim_out % dim_heads == 0

        self.n_heads = dim_out // dim_heads
        self.dim = dim
        self.rotary_emb = rotary_emb
        self.q_stride = q_stride

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        assert self.flash, "Must have flash attention."

        self.qkv = nn.Linear(dim, 3 * dim_out, bias=False)
        self.c_proj = nn.Linear(dim_out, dim_out, bias=False)

    def forward(self, x):
        r"""
        Args:
            x: (b, t, h*d)

        Constants:
            b: batch_size
            t: time steps
            r: 3
            h: heads_num
            d: heads_dim
        """
        B, T, C = x.size()

        q, k, v = rearrange(
            self.qkv(x), "b t (r h d) -> r b h t d", r=3, h=self.n_heads
        )
        # q, k, v: (b, h, t, d)

        if self.q_stride > 1:
            q = reduce(q, "b h (t t1) d -> b h t d", "max", t1=self.q_stride)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0, is_causal=False
            )

        y = rearrange(y, "b h t d -> b t (h d)")

        y = self.c_proj(y)
        # shape: (b, t, h*d)

        return y


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


class LinearAttention(nn.Module):
    """
    this flavor of linear attention proposed in https://arxiv.org/abs/2106.09681 by El-Nouby et al.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        dim_heads: int,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        dim_inner = dim_heads
        heads = dim_out // dim_heads
        self.norm = RMSNorm(dim)
        self.rotary_emb = rotary_emb

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h d n", qkv=3, h=heads),
        )

        self.temperature = nn.Parameter(torch.zeros(heads, 1, 1))

        self.to_out = nn.Sequential(
            Rearrange("b h d n -> b n (h d)"), nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_heads: int,
        rotary_emb: RotaryEmbedding,
        dim_out=None,
        q_stride=1,
    ):

        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.dim_heads = dim_heads

        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim_out)
        self.att = Attention(
            dim=dim,
            dim_out=dim_out,
            dim_heads=dim_heads,
            rotary_emb=rotary_emb,
            q_stride=q_stride,
        )
        self.mlp = MLP(dim=dim_out)

        if self.att.q_stride > 1:
            self.proj = nn.Linear(dim, dim_out)

    def forward(
        self,
        x: torch.Tensor,
    ):
        x_norm = self.att_norm(x)
        if self.att.q_stride > 1:
            x = self.proj(x)
            x = reduce(x, "b (t t1) d -> b t d", "max", t1=self.att.q_stride)
        x = x + self.att(x_norm)
        x = x + self.mlp(self.ffn_norm(x))
        return x


if __name__ == "__main__":
    model = BSRoformer(dim=8, dim_heads=8, depths=[2, 3, 8], f_q_strides=[1, 2, 2])
    print(model)
    mixture = torch.randn(2, 2, 44100 * 2)
    output = model(mixture)
    print(output.shape)
    print(output)
