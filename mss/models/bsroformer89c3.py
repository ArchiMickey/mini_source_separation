from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models.attention import Block
from mss.models.rope import RoPE
from mss.models2.dsp3.banks import erb_linear_banks
from mss.models2.dsp3.subband_fast import SubbandFilter
from mss.utils import fast_sdr


class BSRoformer89c3(nn.Module):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_layers=12,
        n_heads=12,
        dim=768,
        rope_len=8192,
        hop_length=4,
        **kwargs,
    ) -> None:
        super().__init__()

        n_bands = 112
        self.audio_channels = audio_channels
        self.dim = dim
        self.hop_length = int(hop_length)
        self.patch_size_t = 4
        max_bandwidth = 390
        factor = sample_rate // 400
        self.sb_factor = factor
        chunk_size = 16
        self._last_cnn_input_length: int | None = None

        # Subband filter
        banks = erb_linear_banks(
            sr=sample_rate,
            n_bands=n_bands,
            max_bandwidth=max_bandwidth,
        )
        self.sb_filter = SubbandFilter(
            sample_rate,
            banks,
            factor,
            chunk_size=chunk_size,
        )

        # CNN analysis/synthesis filterbank replacing the local STFT/ISTFT pair.
        self.cnn_encoder = nn.Conv2d(
            audio_channels * 2,
            dim,
            kernel_size=(self.hop_length, 1),
            stride=(self.hop_length, 1),
        )
        self.cnn_decoder = nn.ConvTranspose2d(
            dim,
            audio_channels * 2,
            kernel_size=(self.hop_length, 1),
            stride=(self.hop_length, 1),
        )

        # Patch
        in_channels = dim
        self.patch = Patch(in_channels, dim, (self.patch_size_t, 1))
        self.unpatch = UnPatch(dim, in_channels, (self.patch_size_t, 1))

        # RoPE
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)

        # Transformer blocks
        self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.k_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

    def forward(self, audio: Tensor) -> Tensor:
        r"""Separation model.

        Args:
            audio: (b, c, l)

        Returns:
            out: (b, c, l)
        """

        # Subband Analysis
        audio_length = audio.shape[-1]
        audio = self.pad_audio(audio, self.sb_factor)
        x = self.sb_filter.analysis(audio)  # (b, c, k, l') complex analytic
        features = self.cnn_analysis(x)  # (b, d, t, k)

        if False:  # For debug. Analysis-synthesis SDR should over 30 dB.
            self.check_sdr(audio, features)
            os._exit(0)

        # Patchify
        B, _, T, K = features.shape
        x = features
        x = self.pad_tensor(x, self.patch_size_t)  # x: (b, d, t, k)
        x = self.patch(x)

        # Transformer along time and frequency axes.
        for t_block, k_block in zip(self.t_blocks, self.k_blocks):
            x = rearrange(x, "b d t f -> (b f) t d")
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, "(b f) t d -> (b t) f d", b=B)
            x = k_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, "(b t) f d -> b d t f", b=B)  # shape: (b, d, t, f)

        # Unpatchify
        x = self.unpatch(x)
        x = x[:, :, 0:T, :]
        mask = x
        x = features * mask

        # Subband synthesis
        x = self.cnn_synthesis(x)  # (b, c, k, l') complex analytic
        out = self.sb_filter.synthesis(x)

        return out[..., 0:audio_length]

    def cnn_analysis(self, x: Tensor) -> Tensor:
        r"""CNN analysis replacing the old STFT method.

        Args:
            x: complex analytic subbands, (b, c, k, l)

        Returns:
            CNN features, (b, d, t, k)
        """
        B, C, K, L = x.shape
        self._last_cnn_input_length = L
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        x = torch.view_as_real(x.contiguous())  # (b, c, k, l, 2)
        x = rearrange(x, "b c k l ri -> b (c ri) l k")
        x = self._pad_for_cnn_analysis(x)
        return self.cnn_encoder(x)  # (b, d, t, k)

    def cnn_synthesis(self, x: Tensor) -> Tensor:
        r"""CNN synthesis replacing the old ISTFT method.

        Args:
            x: CNN features, (b, d, t, k)

        Returns:
            complex analytic subbands, (b, c, k, l)
        """
        B = x.shape[0]
        C = self.audio_channels
        K = x.shape[-1]
        target_length = self._last_cnn_input_length
        if target_length is None:
            raise RuntimeError("cnn_analysis() must be called before cnn_synthesis() so synthesis length is known")

        x = self.cnn_decoder(x)  # (b, c*2, l_pad, k)
        x = x[:, :, 0:target_length, :]
        if x.shape[2] < target_length:
            x = F.pad(x, pad=(0, 0, 0, target_length - x.shape[2]))
        x = rearrange(x, "b (c ri) l k -> b c k l ri", c=C, ri=2, k=K)
        return torch.view_as_complex(x.contiguous().float())

    def _pad_for_cnn_analysis(self, x: Tensor) -> Tensor:
        length = x.shape[-2]
        kernel_size = self.cnn_encoder.kernel_size[0]
        stride = self.cnn_encoder.stride[0]
        if length <= kernel_size:
            target_length = kernel_size
        else:
            frames = (length - kernel_size + stride - 1) // stride + 1
            target_length = (frames - 1) * stride + kernel_size
        pad_right = target_length - length
        if pad_right > 0:
            x = F.pad(x, pad=(0, 0, 0, pad_right))
        return x

    def pad_tensor(self, x: Tensor, patch_size_t) -> Tensor:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio."""
        pad_t = -x.shape[2] % patch_size_t
        x = F.pad(x, pad=(0, 0, 0, pad_t))
        return x

    def pad_audio(self, audio: Tensor, multiple_l: int) -> Tensor:
        pad_l = -audio.shape[-1] % multiple_l
        return F.pad(audio, pad=(0, pad_l))

    def check_sdr(self, audio, complex_sp) -> None:
        y = self.cnn_synthesis(complex_sp)
        y = self.sb_filter.synthesis(y)
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"SDR: {sdr:.2f} dB")


class Patch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, x):
        return self.conv(x)


class UnPatch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    model = BSRoformer89c3()
    audio = torch.randn(2, 2, 48000 * 2)
    out = model(audio)
    print(out.shape)