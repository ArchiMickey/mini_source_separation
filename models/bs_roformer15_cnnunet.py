import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import numpy as np
import librosa
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from models.fourier import Fourier


class BSRoformer15a(Fourier):
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 441,
        input_channels: int = 2,
        depth: int = 12,
        dim: int = 384,
        num_downs: int = 2,
        n_heads: int = 12
    ):
        super().__init__(n_fft, hop_length)

        self.input_channels = input_channels
        self.depth = depth
        self.dim = dim
        self.n_heads = n_heads

        self.cmplx_num = 2
        
        self.head_dim = self.dim // self.n_heads
        self.downsampling_ratio = 2**num_downs

        sr = 44100
        mel_bins = 256
        out_channels = 64

        self.stft_to_image = StftToImage(
            in_channels=self.input_channels * self.cmplx_num, 
            sr=sr, 
            n_fft=n_fft, 
            mel_bins=mel_bins,
            out_channels=out_channels
        )

        init_dim = self.dim // 2**(num_downs)
        self.init_conv = nn.Conv2d(out_channels, init_dim, kernel_size=7, padding=3)
        
        dims = [init_dim * 2**i for i in range(num_downs+1)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.downs = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in),
                ResnetBlock(dim_in, dim_in),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 1)
            ]))
        self.ups = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i == len(in_out) - 1
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_in + dim_out, dim_out),
                ResnetBlock(dim_in + dim_out, dim_out),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 1)
            ]))

        rotary_emb_t = RotaryEmbedding(dim=self.head_dim)
        rotary_emb_f = RotaryEmbedding(dim=self.head_dim)
        
        self.transformers = nn.ModuleList([])

        for _ in range(self.depth):
            self.transformers.append(nn.ModuleList([
                TransformerBlock(dim=self.dim, n_heads=self.n_heads, rotary_emb=rotary_emb_t),
                TransformerBlock(dim=self.dim, n_heads=self.n_heads, rotary_emb=rotary_emb_f)
            ]))

        self.final_resblock = ResnetBlock(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, out_channels, kernel_size=1)
        
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

        batch_size = complex_sp.shape[0]
        time_steps = complex_sp.shape[2]
                    

        x = torch.view_as_real(complex_sp)
        # shape: (b, c, t, f, z)

        x = rearrange(x, 'b c t f z -> b (c z) t f')

        x = self.stft_to_image.transform(x)
        # shape: (b, d, t, f)
        
        if time_steps % self.downsampling_ratio != 0:
            x = F.pad(x, (0, 0, 0, self.downsampling_ratio - time_steps % self.downsampling_ratio))

        x = self.init_conv(x)
        
        skips = [x]
        for block1, block2, down in self.downs:
            x = block1(x)
            skips.append(x)
            x = block2(x)
            skips.append(x)
            x = down(x)
        # shape: (b, d, t, f)

        for t_transformer, f_transformer in self.transformers:

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_transformer(x)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batch_size)
            x = f_transformer(x)

            x = rearrange(x, '(b t) f d -> b d t f', b=batch_size)

        for block1, block2, up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block2(x)
            x = up(x)
        
        x = self.final_resblock(torch.cat([x, skips.pop()], dim=1))
        x = self.final_conv(x)
        
        x = x[:, :, :time_steps, :]

        x = self.stft_to_image.inverse_transform(x)

        x = rearrange(x, 'b (c z) t f -> b c t f z', c=self.input_channels)
        # shape: (b, c, t, f, z)

        mask = torch.view_as_complex(x.contiguous())
        # shape: (b, c, t, f)

        sep_stft = mask * complex_sp

        output = self.istft(sep_stft)
        # (b, c, samples_num)

        return output

class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2, padding=(0, 0))
        
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2, padding=(0, 0))
        
    def forward(self, x):
        return self.conv(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=3, stride=1, padding=1),
            RMSNorm2d(dim_out),
            nn.SiLU(),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            RMSNorm2d(dim_out),
            nn.SiLU(),
        )
        
        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1, stride=1, padding=0) if dim != dim_out else nn.Identity()
    
    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        r"""https://github.com/meta-llama/llama/blob/main/llama/model.py"""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.weight
        return output


class RMSNorm2d(RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim, eps)
        
    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = super().forward(x)
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class StftToImage(nn.Module):

    def __init__(self, in_channels: int, sr: float, n_fft: int, mel_bins: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.n_fft = n_fft
        self.mel_bins = mel_bins

        melbanks = librosa.filters.mel(
            sr=sr, 
            n_fft=n_fft, 
            n_mels=self.mel_bins - 2, 
            norm=None
        )

        melbank_first = np.zeros(melbanks.shape[-1])
        melbank_first[0] = 1.

        melbank_last = np.zeros(melbanks.shape[-1])
        idx = np.argmax(melbanks[-1])
        melbank_last[idx :] = 1. - melbanks[-1, idx :]

        melbanks = np.concatenate(
            [melbank_first[None, :], melbanks, melbank_last[None, :]], axis=0
        )

        sum_banks = np.sum(melbanks, axis=0)
        assert np.allclose(a=sum_banks, b=1.)

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
        self.register_buffer(name='melbanks', tensor=torch.Tensor(melbanks))

    def transform(self, x):

        vs = []

        for f in range(self.mel_bins):
            
            idxes = self.indexes[f]

            bank = self.melbanks[f, idxes]  # (banks,)
            stft_bank = x[..., idxes]  # (b, c, t, banks)

            v = stft_bank * bank  # (b, c, t, banks)
            v = rearrange(v, 'b c t q -> b t (c q)')

            v = self.band_nets[f](v)  # (b, t, d)
            vs.append(v)

        x = torch.stack(vs, dim=2)  # (b, t, f, d)
        x = rearrange(x, 'b t f d -> b d t f')

        return x

    def inverse_transform(self, x):

        B, _, T, _ = x.shape
        y = torch.zeros(B, self.in_channels, T, self.n_fft // 2 + 1).to(x.device)

        for f in range(self.mel_bins):

            idxes = self.indexes[f]
            v = x[..., f]  # (b, d, t)
            v = rearrange(v, 'b d t -> b t d')
            v = self.inv_band_nets[f](v)  # (b, t, d)
            v = rearrange(v, 'b t (c q) -> b c t q', q=len(idxes))
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

    def __init__(self, dim: int, n_heads: int, rotary_emb: RotaryEmbedding):
        super().__init__()
        
        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.dim = dim
        self.rotary_emb = rotary_emb

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        assert self.flash, "Must have flash attention."
        
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        
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

        q, k, v = rearrange(self.c_attn(x), 'b t (r h d) -> r b h t d', r=3, h=self.n_heads)
        # q, k, v: (b, h, t, d)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False)
        
        y = rearrange(y, 'b h t d -> b t (h d)')

        y = self.c_proj(y)
        # shape: (b, t, h*d)

        return y


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, rotary_emb: RotaryEmbedding):
        
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.att = Attention(dim=dim, n_heads=n_heads, rotary_emb=rotary_emb)
        self.mlp = MLP(dim=dim)
        

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x
