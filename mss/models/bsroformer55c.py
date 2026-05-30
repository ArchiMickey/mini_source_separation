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



class RMSNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        r"""RMSNorm over channel dimension for 2D feature map.

        Args:
            x: (b, d, t, f)

        Outputs:
            output: (b, d, t, f)
        """

        norm_x = x.norm(2, dim=1, keepdim=True)
        rms = norm_x * (x.shape[1] ** -0.5)
        x = x / (rms + self.eps)
        return x * self.scale


class GEGLUFusion(nn.Module):
    """
    Robust GEGLU fusion module. 
    Concatenates features, projects to gate/value, and fuses.
    """

    def __init__(self, dim, dim_mult=2 ):
        super().__init__()

        # 1. Normalization (Pre-norm) - separate norms for decoder and encoder
        self.norm_dec = RMSNorm2d(dim)
        self.norm_enc = RMSNorm2d(dim)

        # 2. Projection
        # We take 2 * dim inputs (enc + dec)
        # We project to 2 * (dim * dim_mult) because GEGLU requires splitting
        hidden_dim = int(dim * dim_mult)
        self.proj_in = nn.Conv2d(dim * 2, hidden_dim * 2, kernel_size=1)
        
        # 3. Output projection
        self.proj_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        # Optional: Zero-init output for identity-start training stability
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x_dec, x_enc):
        """
        Args:
            x_dec: Decoder features (B, C, H, W) -> Acts as the residual backbone
            x_enc: Encoder features (B, C, H', W')
        """
        # 1. Spatial Alignment (Crucial for U-Nets)
        if x_enc.shape[-2:] != x_dec.shape[-2:]:
            x_enc = F.interpolate(x_enc, size=x_dec.shape[-2:], mode='bilinear', align_corners=False)

        # 2. Prepare inputs
        # We apply the residual connection to the Decoder stream, so we save x_dec
        residual = x_dec
        
        # Norm and Concat
        # (Using separate norms for decoder and encoder features)
        x_dec = self.norm_dec(x_dec)
        x_enc = self.norm_enc(x_enc)
        
        combined = torch.cat([x_dec, x_enc], dim=1)

        # 3. Project and Split (The "GEGLU" mechanism)
        # Project to [B, 2*hidden, H, W]
        gate_value = self.proj_in(combined) 
        
        # Split into two chunks along channel dim
        gate, value = gate_value.chunk(2, dim=1)

        # 4. Gating
        hidden = F.gelu(gate) * value

        # 5. Output Project + Residual
        return residual + self.proj_out(hidden)

class BSRoformer(Fourier):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_fft=2048,
        hop_length=480,
        n_bands=256,
        band_dim=64,
        patch_size=[4, 4],
        n_layers=11,
        n_stft_layers=2,
        dim_head=32,
        dim=384,
        dim_stft=96,
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
        self.patch_size = patch_size

        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate, 
            n_fft=n_fft, 
            n_bands=n_bands,
            in_channels=2,  # real + imag
            out_channels=band_dim
        )

        self.patch_stft = nn.Conv2d(band_dim * audio_channels, dim_stft, kernel_size=(1, 1), stride=(1, 1))
        self.unpatch_stft = nn.ConvTranspose2d(dim_stft, band_dim * audio_channels, kernel_size=(1, 1), stride=(1, 1))
        # self.patch_stft = Patch(band_dim * audio_channels, dim_stft, (1, 1))
        # self.unpatch_stft = UnPatch(dim_stft, band_dim * audio_channels, (1, 1))

        kernel_size = patch_size
        self.patch = nn.Conv2d(band_dim * audio_channels, dim, kernel_size=kernel_size, stride=kernel_size)
        self.unpatch = nn.ConvTranspose2d(dim, dim_stft, kernel_size=kernel_size, stride=kernel_size)

        # RoPE
        self.rope = RoPE(head_dim=dim_head, max_len=rope_len)

        # Transformer blocks
        self.blocks_stft = nn.ModuleList(Block(dim_stft, dim_stft // dim_head) for _ in range(n_stft_layers))
        
        self.t_blocks = nn.ModuleList(Block(dim, dim // dim_head) for _ in range(n_layers))
        self.f_blocks = nn.ModuleList(Block(dim, dim // dim_head) for _ in range(n_layers))

        self.fusion = GEGLUFusion(dim_stft)
        

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
        x = rearrange(x, 'b c t f o -> b (c o) t f')  # shape: (b, d, t, f)

        x_stft = self.patch_stft(x)

        #
        B = x.shape[0]
        x = self.patch(x)

        # Middle
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        x = self.unpatch(x)
        

        #
        x = self.fusion(x, x_stft)
        
        x = rearrange(x, 'b d t f -> (b t) f d')
        for block in self.blocks_stft:
            x = block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)
        x = rearrange(x, '(b t) f d -> b d t f', b=B)

        # --- 3. Decode ---
        # 3.1 Unpatchify
        x = self.unpatch_stft(x)
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

'''
class BSRoformer42a(Fourier):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_fft=2048,
        hop_length=480,
        n_bands=256,
        band_dim=64,
        patch_size=[4, 4],
        n_layers=12,
        n_heads=12,
        dim=768,
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
        self.patch_size = patch_size

        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate, 
            n_fft=n_fft, 
            n_bands=n_bands,
            in_channels=2,  # real + imag
            out_channels=band_dim
        )

        kernel_size = (4, 4)
        self.patch = Patch(band_dim * audio_channels, dim, kernel_size)
        self.unpatch = UnPatch(dim, band_dim * audio_channels, kernel_size)

        # self.patch_f = nn.Conv1d(in_channels=band_dim, out_channels=dim, kernel_size=4, stride=4)
        # self.patch_t = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=4, stride=4)
        # # self.patch_c = nn.Conv1d(in_channels=band_dim, out_channels=dim, kernel_size=audio_channels, stride=audio_channels)
        # self.patch_c = nn.Linear(dim * audio_channels, dim)

        # self.patch = nn.Conv2d(band_dim, dim, kernel_size=patch_size, stride=patch_size)
        # self.unpatch = nn.ConvTranspose2d(dim, band_dim, kernel_size=patch_size, stride=patch_size)

        # RoPE
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)

        # Transformer blocks
        self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.f_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

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

        torch.cuda.synchronize()
        t1 = time.time()
        # 1.3 Convert STFT to mel scale
        x = self.bandsplit.transform(x)  # shape: (b, c, t, f, o)
        torch.cuda.synchronize()
        print("a1", time.time() - t1)
        torch.cuda.synchronize()

        x = self.patch(x)

        B = x.shape[0]
        # T1 = x.shape[2]

        torch.cuda.synchronize()
        t1 = time.time()
        # --- 2. Transformer along time and frequency axes ---
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        torch.cuda.synchronize()
        print("a2", time.time() - t1)
        torch.cuda.synchronize()

        # --- 3. Decode ---
        # 3.1 Unpatchify
        x = self.unpatch(x, self.ac)

        torch.cuda.synchronize()
        t1 = time.time()

        # 3.2 Convert mel scale STFT to original STFT
        x = self.bandsplit.inverse_transform(x)  # shape: (b, c, t, f, k)

        torch.cuda.synchronize()
        print("a3", time.time() - t1)
        torch.cuda.synchronize()

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
'''

class Patch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x):
        x = rearrange(x, 'b c t f i -> b (c i) t f')
        x = self.conv(x)
        return x


class UnPatch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x, audio_channels):
        x = self.conv(x)
        x = rearrange(x, 'b (c i) t f -> b c t f i', c=audio_channels)
        
        return x