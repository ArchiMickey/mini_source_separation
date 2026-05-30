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


class GEGLUFusion(nn.Module):
    """
    Robust GEGLU fusion module. 
    Concatenates features, projects to gate/value, and fuses.
    """

    def __init__(self, dim, dim_mult=2, num_groups=8):
        super().__init__()
        
        # Ensure groups is valid
        if dim % num_groups != 0:
            num_groups = 1  # Fallback to LayerNorm style if dimensions are odd

        # 1. Normalization (Pre-norm) - separate norms for decoder and encoder
        self.norm_dec = nn.GroupNorm(num_groups, dim)
        self.norm_enc = nn.GroupNorm(num_groups, dim)

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


class BSRoformerBlock(nn.Module):
    """Band-Split RoFormer block with time and/or frequency attention."""

    def __init__(self, dim, n_heads, axis='tf'):
        super().__init__()
        assert axis in ['t', 'f', 'tf', 'ft'], "Axis must be one of 't', 'f', 'tf', or 'ft'."
        self.axis = axis
        self.time_block = Block(dim, n_heads) if 't' in axis else None
        self.freq_block = Block(dim, n_heads) if 'f' in axis else None

    def forward(self, x: Tensor, rope: RoPE) -> Tensor:
        """
        Apply time and/or frequency attention.

        Args:
            x: Input tensor of shape (b, d, t, f).
            rope: Rotary position embedding.

        Returns:
            Output tensor of shape (b, d, t, f).
        """
        B = x.shape[0]

        if self.time_block is not None:
            # Time attention
            x = rearrange(x, 'b d t f -> (b f) t d')
            x = self.time_block(x, rope=rope, pos=None)
            x = rearrange(x, '(b f) t d -> b d t f', b=B)

        if self.freq_block is not None:
            # Frequency attention
            x = rearrange(x, 'b d t f -> (b t) f d')
            x = self.freq_block(x, rope=rope, pos=None)
            x = rearrange(x, '(b t) f d -> b d t f', b=B)

        return x


class BSRoformer53a(Fourier):
    """Band-Split RoFormer for music source separation."""

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
        strides=[(2, 2), (4, 4)],  # len(strides) = number of stages - 1
        depths=[1, 2, 7],  # Total transformer blocks
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

            # Encoder stage
            stage = nn.ModuleList([
                BSRoformerBlock(dim=dims[i], n_heads=dims[i] // dim_head, axis=mode_i)
                for _ in range(depth)
            ])
            down = nn.Conv2d(dims[i], dims[i + 1], kernel_size=stride, stride=stride)
            self.downs.append(nn.ModuleList([stage, down]))

            # Fusion module
            self.fusions.insert(0, GEGLUFusion(dims[i]))

            # Decoder stage
            stage = nn.ModuleList([
                BSRoformerBlock(dim=dims[i], n_heads=dims[i] // dim_head, axis=mode_i)
                for _ in range(depth)
            ])
            up = nn.ConvTranspose2d(dims[i + 1], dims[i], kernel_size=stride, stride=stride)
            self.ups.insert(0, nn.ModuleList([up, stage]))

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            BSRoformerBlock(dim=dims[-1], n_heads=dims[-1] // dim_head, axis=mode[-1])
            for _ in range(bottleneck_depth)
        ])

    def forward(self, audio: Tensor) -> Tensor:
        """
        Separate audio sources.

        Args:
            audio: Input audio of shape (b, c, l).

        Returns:
            Separated audio of shape (b, c, l).
        """
        # --- Encode ---
        # Complex spectrum
        complex_sp = self.stft(audio)
        T0 = complex_sp.shape[2]

        x = torch.view_as_real(complex_sp)

        # Pad STFT
        x = self.pad_tensor(x)

        # Convert STFT to mel scale
        x = self.bandsplit.transform(x)

        x = self.patch(rearrange(x, 'b c t f i -> b (c i) t f'))

        # Downsample
        hs = []
        for stage, down in self.downs:
            for block in stage:
                x = block(x, rope=self.rope)
            hs.append(x)
            x = down(x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x, rope=self.rope) 

        # Upsample
        for (up, stage), fusion in zip(self.ups, self.fusions):
            x = up(x)
            x = fusion(x, hs.pop())
            for block in stage:
                x = block(x, rope=self.rope)

        # --- Decode ---
        # Unpatchify
        x = self.unpatch(x)
        x = rearrange(x, 'b (c i) t f -> b c t f i', c=self.ac)

        # Convert mel scale STFT to original STFT
        x = self.bandsplit.inverse_transform(x)

        # Unpad
        x = x[:, :, 0:T0, :, :]

        # Get complex mask
        mask = torch.view_as_complex(x)

        # Calculate STFT of separated audio
        sep_stft = mask * complex_sp

        # ISTFT
        output = self.istft(sep_stft)

        return output

    def pad_tensor(self, x: Tensor) -> tuple[Tensor, int]:
        """
        Pad a spectrum so it can be evenly divided by downsample ratio.

        Args:
            x: Input tensor, e.g., (b, c, t=201, f).

        Returns:
            Padded tensor, e.g., (b, c, t=204, f).
        """
        # Pad last frames, e.g., 201 -> 204
        pad_t = -x.shape[2] % self.patch_size[0]
        x = F.pad(x, pad=(0, 0, 0, 0, 0, pad_t))

        return x


if __name__ == "__main__":
    model = BSRoformer53a()
    dummy_audio = torch.randn(1, 2, 48000 * 2)
    output = model(dummy_audio)
    print(output.shape)
