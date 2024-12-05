import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import librosa
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from mss.models.fourier import Fourier
from torch.nn import Module, ModuleList
from functools import partial



class BSRoformerUNet(Fourier):
    def __init__(
        self,
        sr: int = 44100,
        mel_bins: int = 256,
        n_fft: int = 2048,
        hop_length: int = 441,
        input_channels: int = 2,
        dim: int = 128,
        dim_mults = [1, 2, 4, 8],
        attn_dim_head: int = 32,
        dropout=0.1
    ):
        super().__init__(n_fft, hop_length)

        self.input_channels = input_channels
        self.dim = dim

        self.cmplx_num = 2

        out_channels = mel_bins
        
        self.model = Unet(dim=dim, dim_mults=dim_mults, channels=out_channels, attn_dim_head=attn_dim_head, dropout=dropout)

        self.stft_to_image = StftToImage(
            in_channels=self.input_channels * self.cmplx_num, 
            sr=sr, 
            n_fft=n_fft, 
            mel_bins=mel_bins,
            out_channels=out_channels
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

        batch_size = complex_sp.shape[0]
        time_steps = complex_sp.shape[2]

        x = torch.view_as_real(complex_sp)
        # shape: (b, c, t, f, z)

        x = rearrange(x, 'b c t f z -> b (c z) t f')

        x = self.stft_to_image.transform(x)
        # shape: (b, d, t, f)

        x = self.model(x)
        
        x = self.stft_to_image.inverse_transform(x)

        x = rearrange(x, 'b (c z) t f -> b c t f z', c=self.input_channels)
        # shape: (b, c, t, f, z)

        mask = torch.view_as_complex(x.contiguous())
        # shape: (b, c, t, f)

        sep_stft = mask * complex_sp

        output = self.istft(sep_stft)
        # (b, c, samples_num)

        return output




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


# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm2d(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

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

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm2d(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, dropout = 0.):
        super().__init__()

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):        

        h = self.block1(x)

        h = self.block2(h)

        return h + self.res_conv(x)


class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        b, _, n, d = x.shape
        
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b z n (h d) -> b z h n d', h = self.heads), qkv)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b z h n d -> b z n (h d)")
        return self.to_out(out)

class BSRoformerBlock(Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        
        self.attn_t = Attention(dim, heads=heads, dim_head=dim_head)
        self.attn_f = Attention(dim, heads=heads, dim_head=dim_head)
    
    def forward(self, x):
        x = rearrange(x, "b d t f -> b f t d")
        x = x + self.attn_t(x)
        x = rearrange(x, "b f t d -> b t f d")
        x = x + self.attn_f(x)
        return rearrange(x, "b t f d -> b d t f")

# model

class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        dropout = 0.,
        attn_dim_head = 32,
        full_attn = None,    # defaults to full attention only for inner most layer
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        resnet_block = partial(ResnetBlock, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = BSRoformerBlock

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = dim_in // layer_attn_dim_head),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = BSRoformerBlock(mid_dim, heads = mid_dim // attn_dim_head[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = BSRoformerBlock

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = dim_out // layer_attn_dim_head),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x):
        # assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        init_shape = x.shape[-2:]
        if not all([divisible_by(d, self.downsample_factor) for d in init_shape]):
            new_shape = [int(np.ceil(d / self.downsample_factor) * self.downsample_factor) for d in init_shape]
            x = F.pad(x, (0, new_shape[-1] - init_shape[-1], 0, new_shape[-2] - init_shape[-2]))

        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)

            x = block2(x)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x)
        x = self.final_conv(x)
        x = x[..., :init_shape[0], :init_shape[1]]
        return x

if __name__ == "__main__":
    model = BSRoformerUNet(dim=32)
    print(model)
    mixture = torch.randn(2, 2, 44100*2)
    output = model(mixture)
    print(output.shape)
    print(output)