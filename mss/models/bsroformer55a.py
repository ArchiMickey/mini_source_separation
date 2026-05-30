from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models.attention import Block
from mss.models.bandsplit42a import BandSplit
from mss.models.fourier import Fourier
from mss.models.rope import RoPE
import time



class SiGLUFusion(nn.Module):
    def __init__(self, dim, dim_mult=2):
        super().__init__()
        self.norm_enc = nn.GroupNorm(8, dim)
        self.norm_dec = nn.GroupNorm(8, dim)
        
        # We project to 2 * dim so we can split the tensor
        self.proj_enc = nn.Conv2d(dim, dim * dim_mult, kernel_size=1)
        self.proj_dec = nn.Conv2d(dim, dim * dim_mult, kernel_size=1)
        self.proj_out = nn.Conv2d(dim * dim_mult, dim, kernel_size=1)

    def forward(self, x_dec, x_enc):
        # Contextual gate from Decoder, Content from Encoder
        gate = self.proj_dec(self.norm_dec(x_dec))
        up = self.proj_enc(self.norm_enc(x_enc))
        
        # This is the "SiGLU" fusion logic:
        # It's essentially SiLU(gate) * value
        # Or more accurately: (gate * sigmoid(gate)) * value
        fused = F.silu(gate) * up
        
        return x_dec + self.proj_out(fused)

class BSRoformerBlock(nn.Module):
    def __init__(self, dim, n_heads, axis='tf'):
        super().__init__()
        assert axis in ['t', 'f', 'tf', 'ft'], "Axis must be one of 't', 'f', 'tf', or 'ft'."
        self.axis = axis
        self.time_block = Block(dim, n_heads) if 't' in axis else None
        self.freq_block = Block(dim, n_heads) if 'f' in axis else None

    def forward(self, x: Tensor, rope: RoPE) -> Tensor:
        r"""BSRoformer block.

        b: batch_size
        d: feature_dim
        t: frames_num
        f: freq_bins

        Args:
            x: (b, d, t, f)

        Outputs:
            output: (b, d, t, f)
        """

        B = x.shape[0]

        if self.time_block is not None:
            # Time attention
            x = rearrange(x, 'b d t f -> (b f) t d')
            x = self.time_block(x, rope=rope, pos=None)  # shape: (b*f, t, d)
            x = rearrange(x, '(b f) t d -> b d t f', b=B)

        if self.freq_block is not None:
            # Frequency attention
            x = rearrange(x, 'b d t f -> (b t) f d')
            x = self.freq_block(x, rope=rope, pos=None)  # shape: (b*t, f, d)
            x = rearrange(x, '(b t) f d -> b d t f', b=B)

        return x

class BSRoformer53a(Fourier):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_fft=2048,
        hop_length=480,
        n_bands=256,
        band_dim=64,
        dim=96,
        dim_head=32,
        dim_mults=[1, 2, 4],
        strides=[(2, 2), (4, 4)], # len(strides) = number of stages - 1
        depths=[1, 2, 7], # 24 transformer blocks
        mode=['f', 'tf', 'tf'],
        rope_len=8192,
        **kwargs
    ) -> None:

        super().__init__(
            n_fft=n_fft, 
            hop_length=hop_length, 
            return_complex=True, 
            normalized=True
        )

        self.ac = audio_channels
        self.patch_size = torch.prod(torch.tensor(strides), dim=0).tolist()

        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate, 
            n_fft=n_fft, 
            n_bands=n_bands,
            in_channels=2,  # real + imag
            out_channels=band_dim
        )

        # RoPE
        self.rope = RoPE(head_dim=dim_head, max_len=rope_len)
        
        # Blocks
        bottleneck_depth = depths[-1]
        self.patch = nn.Conv2d(band_dim * audio_channels, dim, kernel_size=1)
        self.unpatch = nn.Conv2d(dim, band_dim * audio_channels, kernel_size=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.fusions = nn.ModuleList()
        
        dims = [dim * m for m in dim_mults]
        for i in range(len(depths) - 1):
            depth = depths[i]
            stride = strides[i]
            mode_i = mode[i]
            stage = nn.ModuleList([
                BSRoformerBlock(dim=dims[i], n_heads=dims[i] // dim_head, axis=mode_i) for _ in range(depth)
            ])
            down = nn.Conv2d(dims[i], dims[i+1], kernel_size=stride, stride=stride)
            self.downs.append(nn.ModuleList([stage, down]))
            
            self.fusions.insert(0, SiGLUFusion(dims[i]))
            
            stage = nn.ModuleList([
                BSRoformerBlock(dim=dims[i], n_heads=dims[i] // dim_head, axis=mode_i) for _ in range(depth)
            ])
            up = nn.ConvTranspose2d(dims[i+1], dims[i], kernel_size=stride, stride=stride)
            self.ups.insert(0, nn.ModuleList([up, stage]))
        
        # Ensure dims[-1] is actually 384 and that's what the bottleneck gets
        self.bottleneck = nn.ModuleList([
            BSRoformerBlock(dim=dims[-1], n_heads=dims[-1] // dim_head, axis=mode[-1]) 
            for _ in range(bottleneck_depth)
        ])

    def forward(self, audio: Tensor) -> Tensor:
        r"""Separation model.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: freq_bins

        Args:
            audio: (b, c, t)

        Outputs:
            output: (b, c, t)
        """


        # --- 1. Encode ---
        # 1.1 Complex spectrum
        complex_sp = self.stft(audio)  # shape: (b, c, t, f)
        T0 = complex_sp.shape[2]

        x = torch.view_as_real(complex_sp)  # shape: (b, c, t, f, 2)

        # 1.2 Pad stft
        x = self.pad_tensor(x)  # x: (b, d, t, f)

        # 1.3 Convert STFT to mel scale
        x = self.bandsplit.transform(x)  # shape: (b, c, t, f, o)

        x = self.patch(rearrange(x, 'b c t f i -> b (c i) t f'))  # shape: (b, d, t, f)

        #Downsample
        hs = []
        for stage, down in self.downs:
            for block in stage:
                x = block(x, rope=self.rope)  # shape: (b, d, t, f)
            hs.append(x)
            x = down(x)  # shape: (b, d*, t//s_t, f//s_f)

        # Middle
        for block in self.bottleneck:
            x = block(x, rope=self.rope)  # shape: (b, d, t, f)
        
        # Upsample
        for (up, stage), fusion in zip(self.ups, self.fusions):
            x = up(x)
            
            x = fusion(x, hs.pop())
            
            for block in stage:
                x = block(x, rope=self.rope)  # shape: (b, d, t, f)
        

        # --- 3. Decode ---
        # 3.1 Unpatchify
        x = self.unpatch(x)  # shape: (b, c*o, t, f)
        x = rearrange(x, 'b (c i) t f -> b c t f i', c=self.ac)

        # 3.2 Convert mel scale STFT to original STFT
        x = self.bandsplit.inverse_transform(x)  # shape: (b, c, t, f, k)

        # Unpad
        x = x[:, :, 0 : T0, :, :]
        
        # 3.3 Get complex mask
        mask = torch.view_as_complex(x)  # shape: (b, c, t, f)

        # 3.5 Calculate stft of separated audio
        sep_stft = mask * complex_sp  # shape: (b, c, t, f)

        # 3.6 ISTFT
        output = self.istft(sep_stft)  # shape: (b, c, l)

        return output

    def pad_tensor(self, x: Tensor) -> tuple[Tensor, int]:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f)
        
        Outpus:
            output: E.g., (b, c, t=204, f)
        """

        # Pad last frames, e.g., 201 -> 204
        pad_t = -x.shape[2] % self.patch_size[0]  # Equals to p - (T % p)
        x = F.pad(x, pad=(0, 0, 0, 0, 0, pad_t))

        return x

if __name__ == "__main__":
    model = BSRoformer53a()
    dummy_audio = torch.randn(1, 2, 48000 * 2)
    output = model(dummy_audio)
    print(output.shape)