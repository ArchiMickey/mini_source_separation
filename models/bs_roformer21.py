import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import numpy as np
import librosa
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from models.fourier import Fourier


class BSRoformer21(Fourier):
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 441,
        input_channels: int = 2,
        dim: int = 96,
        dim_mults: list = [1, 2, 4],
        depths: list = [2, 2, 2],
        block_types: list = ['resnet', 'resnet', 'resnet'],
        mid_transformer_depth: int = 6,
        attn_dim_head: int = 32,
    ):
        super().__init__(n_fft, hop_length)

        assert len(block_types) == len(depths) == len(dim_mults) - 1
        
        self.input_channels = input_channels
        self.dim = dim

        self.cmplx_num = 2
        
        self.head_dim = attn_dim_head
        num_downs = len(dim_mults) - 1
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

        init_dim = dim
        self.init_conv = nn.Conv2d(out_channels, dim, kernel_size=7, padding=3)
        
        dims = [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        rotary_emb_t = RotaryEmbedding(dim=self.head_dim)
        rotary_emb_f = RotaryEmbedding(dim=self.head_dim)
        
        self.downs = nn.ModuleList([])
        for i, ((dim_in, dim_out), block_type, depth) in enumerate(zip(in_out, block_types, depths)):
            is_last = i == len(in_out) - 1
            blocks = nn.ModuleList([])
            for _ in range(depth):
                if block_type == 'resnet':
                    blocks.append(ResnetBlock(dim_in, dim_in))
                elif block_type == 'bsroformer':
                    blocks.append(BSRoformerBlock(dim_in, dim_in // attn_dim_head, rotary_emb_t, rotary_emb_f))
                elif block_type == 'resnet+bsroformer':
                    blocks.append(nn.Sequential(
                        ResnetBlock(dim_in, dim_in),
                        BSRoformerBlock(dim_in, dim_in // attn_dim_head, rotary_emb_t, rotary_emb_f)
                    ))
                else:
                    raise NotImplementedError(f"Block type {block_type} is not implemented.")
            blocks.append(Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 1))
            self.downs.append(blocks)
        
        self.transformers = nn.ModuleList([])

        for _ in range(mid_transformer_depth):
            self.transformers.append(BSRoformerBlock(dim_out, dim_out // attn_dim_head, rotary_emb_t, rotary_emb_f))
        
        self.ups = nn.ModuleList([])
        for i, ((dim_in, dim_out), block_type, depth) in enumerate(zip(reversed(in_out), reversed(block_types), reversed(depths))):
            is_last = i == len(in_out) - 1
            blocks = nn.ModuleList([])
            for _ in range(depth):
                if block_type == 'resnet':
                    blocks.append(ResnetBlock(dim_in + dim_out, dim_out))
                elif block_type == 'bsroformer':
                    blocks.append(nn.Sequential(
                        nn.Conv2d(dim_in + dim_out, dim_out, 1),
                        BSRoformerBlock(dim_out, dim_out // attn_dim_head, rotary_emb_t, rotary_emb_f)
                    ))
                elif block_type == 'resnet+bsroformer':
                    blocks.append(nn.Sequential(
                        ResnetBlock(dim_in + dim_out, dim_out),
                        BSRoformerBlock(dim_out, dim_out // attn_dim_head, rotary_emb_t, rotary_emb_f)
                    ))
                else:
                    raise NotImplementedError(f"Block type {block_type} is not implemented.")
            blocks.append(Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 1))
            self.ups.append(blocks)

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
        for stage in self.downs:
            blocks, down = stage[:-1], stage[-1]
            for block in blocks:
                x = block(x)
                skips.append(x)
            x = down(x)
        # shape: (b, d, t, f)

        for transformer in self.transformers:
            x = transformer(x)

        for stage in self.ups:
            blocks, up = stage[:-1], stage[-1]
            for block in blocks:
                x = torch.cat([x, skips.pop()], dim=1)
                x = block(x)
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

class LayerScale(nn.Module):
    def __init__(self, dim, fn, init_eps=0.1):
        super().__init__()
        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim, 8 * dim, bias=False)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = self.silu(x) * gate
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
        
        self.q_norm = RMSNorm(dim // n_heads)
        self.k_norm = RMSNorm(dim // n_heads)
        
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
        
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False)
        
        y = rearrange(y, 'b h t d -> b t (h d)')

        y = self.c_proj(y)
        # shape: (b, t, h*d)

        return y

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

class LinearAttention2d(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        assert self.flash, "Must have flash attention."
        
        self.temperature = nn.Parameter(torch.zeros(n_heads, 1, 1))
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h d n', h=self.n_heads), (q, k, v))
        
        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()
        
        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False)
        
        out = rearrange(out, 'b h d n -> b n (h d)')
        
        return self.c_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, rotary_emb: RotaryEmbedding=None, linear_attn=False):
        
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.att = LayerScale(dim, Attention(dim=dim, n_heads=n_heads, rotary_emb=rotary_emb) if not linear_attn else LinearAttention2d(dim=dim, n_heads=n_heads))
        self.mlp = LayerScale(dim, MLP(dim=dim))
        

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x

class BSRoformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, rotary_emb_t: RotaryEmbedding, rotary_emb_f: RotaryEmbedding):
        super().__init__()
        self.transformer = TransformerBlock(dim=dim, n_heads=n_heads, linear_attn=True)
        self.transformer_t = TransformerBlock(dim=dim, n_heads=n_heads, rotary_emb=rotary_emb_t)
        self.transformer_f = TransformerBlock(dim=dim, n_heads=n_heads, rotary_emb=rotary_emb_f)
    def forward(self, x):
        b, t = x.size(0), x.size(2)
        x = rearrange(x, 'b d t f -> b (t f) d')
        x = self.transformer(x)
        x = rearrange(x, 'b (t f) d -> (b f) t d', t=t)
        x = self.transformer_t(x)
        x = rearrange(x, '(b f) t d -> (b t) f d', b=b)
        x = self.transformer_f(x)
        x = rearrange(x, '(b t) f d -> b d t f', b=b)
        return x