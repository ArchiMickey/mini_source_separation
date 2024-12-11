import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import math
import numpy as np
import librosa
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from mss.models.fourier import Fourier
from torch.nn import Module, ModuleList
from functools import partial
from typing import List


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

class BSRoformerUNet(Fourier):
    def __init__(
        self,
        sr: int = 44100,
        mel_bins: int = 256,
        n_fft: int = 2048,
        hop_length: int = 441,
        input_channels: int = 2,
        channels: int = 128,
        channel_multipliers: List[int] = [1, 2, 2, 4],
        block_types: List[str] = ["resblock", "resblock", "resblock+bst", "resblock+bst"],
        depths: List[int] = [1, 1, 1, 1],
        downsample_type: str = "Conv",
        upsample_type: str = "InterpolateConv",
        n_heads: int = 4,
        tf_layers: int = 1,
        checkpoint=False
    ):
        super().__init__(n_fft, hop_length)

        self.input_channels = input_channels

        self.cmplx_num = 2

        out_channels = mel_bins
        
        self.model = UNetModel(
            in_channels=out_channels,
            out_channels=out_channels,
            channels=channels,
            channel_multipliers=channel_multipliers,
            block_types=block_types,
            depths=depths,
            downsample_type=downsample_type,
            upsample_type=upsample_type,
            n_heads=n_heads,
            tf_layers=tf_layers,
            checkpoint=checkpoint
        )

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

class UNetModel(nn.Module):
    """
    ## U-Net model
    """

    def __init__(
            self, *,
            in_channels: int,
            out_channels: int,
            channels: int,
            channel_multipliers: List[int],
            block_types: List[str],
            depths: List[int],
            downsample_type: str = "Conv",
            upsample_type: str = "InterpolateConv",
            n_heads: int,
            tf_layers: int=1,
            checkpoint=False
        ):
        """
        :param in_channels: is the number of channels in the input feature map
        :param out_channels: is the number of channels in the output feature map
        :param channels: is the base channel count for the model
        :param n_res_blocks: number of residual blocks at each level
        :param attention_levels: are the levels at which attention should be performed
        :param channel_multipliers: are the multiplicative factors for number of channels for each level
        :param n_heads: is the number of attention heads in the transformers
        :param tf_layers: is the number of transformer layers in the transformers
        :param d_cond: is the size of the conditional embedding in the transformers
        """
        super().__init__()
        self.channels = channels
        
        assert len(channel_multipliers) == len(depths) == len(block_types)

        # Number of levels
        levels = len(channel_multipliers)
        self.downsample_factor = 2 ** (levels - 1)

        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()
        
        self.input_blocks.append(nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1)))
        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]
        
        # Prepare levels
        for i in range(levels):
            # Add the residual blocks and attentions
            for _ in range(depths[i]):
                # Residual block maps from previous number of channels to the number of
                # channels in the current level
                if block_types[i] == "resblock":
                    layers = [ResBlock(channels, out_channels=channels_list[i])]
                elif block_types[i] == "resblock+bst":
                    layers = [ResBlock(channels, out_channels=channels_list[i])]
                    layers.append(BSTransformer(channels_list[i], n_heads, tf_layers))
                else:
                    raise ValueError(f"Invalid block type: {block_types[i]}")
                
                channels = channels_list[i]
                
                # Add them to the input half of the U-Net and keep track of the number of channels of
                # its output
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_channels.append(channels)
            # Down sample at all levels except last
            if i != levels - 1:
                if downsample_type == "Conv":
                    self.input_blocks.append(nn.Sequential(DownSample(channels)))
                elif downsample_type == "AveragingConv":
                    self.input_blocks.append(nn.Sequential(ResidualBlock(DownSample(channels), PixelUnshuffleChannelAveragingDownSampleLayer(channels, channels, 2))))
                else:
                    raise ValueError(f"Invalid downsample type: {downsample_type}")
                input_block_channels.append(channels)

        self.middle_blocks = nn.Sequential(
            ResBlock(channels, out_channels=channels),
            BSTransformer(channels, n_heads, tf_layers),
            ResBlock(channels, out_channels=channels),
        )
        
        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])
        # Prepare levels in reverse order
        for i in reversed(range(levels)):
            # Add the residual blocks and attentions
            for j in range(depths[i] + 1):
                # Residual block maps from previous number of channels plus the
                # skip connections from the input half of U-Net to the number of
                # channels in the current level.
                
                if block_types[i] == "resblock":
                    layers = [ResBlock(channels + input_block_channels.pop(), out_channels=channels_list[i])]
                elif block_types[i] == "resblock+bst":
                    layers = [ResBlock(channels + input_block_channels.pop(), out_channels=channels_list[i])]
                    layers.append(BSTransformer(channels_list[i], n_heads, tf_layers))
                else:
                    raise ValueError(f"Invalid block type: {block_types[i]}")
                
                channels = channels_list[i]
                # Up-sample at every level after last residual block
                # except the last one.
                # Note that we are iterating in reverse; i.e. `i == 0` is the last.
                if i != 0 and j == depths[i]:
                    if upsample_type == "InterpolateConv":
                        layers.append(UpSample(channels))
                    elif upsample_type == "DuplicatingInterpolateConv":
                        layers.append(ResidualBlock(UpSample(channels), ChannelDuplicatingPixelUnshuffleUpSampleLayer(channels, channels, 2)))
                    else:
                        raise ValueError(f"Invalid upsample type: {upsample_type}")
                # Add to the output half of the U-Net
                self.output_blocks.append(nn.Sequential(*layers))

        # Final normalization and $3 \times 3$ convolution
        self.out = nn.Sequential(
            RMSNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        
        self.checkpoint = checkpoint
    
    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """
        init_shape = x.shape[-2:]
        if not all([divisible_by(d, self.downsample_factor) for d in init_shape]):
            new_shape = [int(np.ceil(d / self.downsample_factor) * self.downsample_factor) for d in init_shape]
            x = F.pad(x, (0, new_shape[-1] - init_shape[-1], 0, new_shape[-2] - init_shape[-2]))
        
        # To store the input half outputs for skip connections
        x_input_block = []
        
        # Input half of the U-Net
        for module in self.input_blocks:
            x = checkpoint(module, x) if self.checkpoint else module(x)
            x_input_block.append(x)
        
        # Middle blocks
        x = checkpoint(self.middle_blocks, x) if self.checkpoint else self.middle_blocks(x)
        
        # Output half of the U-Net
        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = checkpoint(module, x) if self.checkpoint else module(x)

        # Final normalization and $3 \times 3$ convolution
        x = checkpoint(self.out, x) if self.checkpoint else self.out(x)
        x = x[..., :init_shape[0], :init_shape[1]]
        return x

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


class Attention(Module):
    def __init__(
        self,
        dim,
        n_heads = 4,
    ):
        super().__init__()
        self.n_heads = n_heads

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, _, n, d = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b z n (h d) -> b z h n d', h = self.n_heads), qkv)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b z h n d -> b z n (h d)")
        return self.to_out(out)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        n_heads = 4,
    ):
        super().__init__()
        self.n_heads = n_heads

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim * 3, bias = False),
            Rearrange('b z n (qkv h d) -> qkv b z h d n', qkv = 3, h = n_heads)
        )

        self.temperature = nn.Parameter(torch.zeros(n_heads, 1, 1))

        self.to_out = nn.Sequential(
            Rearrange('b z h d n -> b z n (h d)'),
            nn.Linear(dim, dim, bias = False)
        )
    
    def forward(self, x):
        q, k, v = self.to_qkv(x)

        q, k = map(lambda x: F.normalize(x, dim = -1, p = 2), (q, k))
        q = q * self.temperature.exp()
        
        out = F.scaled_dot_product_attention(q, k, v)
        
        return self.to_out(out)

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
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, linear_attn=False):
        
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        
        attn_klass = Attention if not linear_attn else LinearAttention
        
        self.att = attn_klass(dim=dim, n_heads=n_heads)
        self.mlp = MLP(dim=dim)
        

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x

class BasicBSTransformerBlock(Module):
    def __init__(self, dim, n_heads, linear_attn=False):
        super().__init__()
        self.transformer = TransformerBlock(dim, n_heads=n_heads, linear_attn=True)
        self.transformer_t = TransformerBlock(dim, n_heads=n_heads, linear_attn=linear_attn)
        self.transformer_f = TransformerBlock(dim, n_heads=n_heads, linear_attn=linear_attn)
    
    def forward(self, x):
        x = rearrange(x, "b d t f -> b f t d")
        b, f, t, d = x.shape
        x = rearrange(x, "b f t d -> b (f t) d").unsqueeze(1)
        x = self.transformer(x)
        x = rearrange(x.squeeze(1), "b (f t) d -> b f t d", f=f, t=t)
        x = self.transformer_t(x)
        x = rearrange(x, "b f t d -> b t f d")
        x = self.transformer_f(x)
        return rearrange(x, "b t f d -> b d t f")

class BSTransformer(Module):
    def __init__(self, dim, n_heads, n_layers):
        super().__init__()
        self.norm = RMSNorm2d(dim)
        self.proj_in = nn.Conv2d(dim, dim, 1)
        
        self.transformer_blocks = nn.ModuleList([BasicBSTransformerBlock(dim, n_heads) for _ in range(n_layers)])
        
        self.proj_out = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x):
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.proj_out(x)
        return x + x_in
        

def divisible_by(numer, denom):
    return (numer % denom) == 0

class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.op(x)

class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module,
        shortcut: nn.Module,
    ):
        super(ResidualBlock, self).__init__()

        self.main = main
        self.shortcut = shortcut

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.forward_main(x) + self.shortcut(x)
        return res

class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, H, W)
        x = x.mean(dim=2)
        return x

class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x

class ResBlock(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(self, channels: int, *, out_channels=None):
        """
        :param channels: the number of input channels
        :param d_t_emb: the size of timestep embeddings
        :param out_channels: is the number of out channels. defaults to `channels.
        """
        super().__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            RMSNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        # Final convolution layer
        self.out_layers = nn.Sequential(
            RMSNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        # Initial convolution
        h = self.in_layers(x)
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        return self.skip_connection(x) + h

if __name__ == "__main__":
    model = BSRoformerUNet(channels=32)
    print(model)
    mixture = torch.randn(2, 2, 44100*2)
    output = model(mixture)
    print(output.shape)
    print(output)