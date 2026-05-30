from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models.attention import Block
from mss.models.bsroformer91 import FreqTransformerBlock, Patch, UnPatch
from mss.models.rope import RoPE
from mss.models2.dsp3.banks import erb_linear_banks
from mss.models2.dsp3.subband_fast import SubbandFilter
from mss.utils import fast_sdr


class BSRoformer91Cnn(nn.Module):
    def __init__(
        self,
        audio_channels: int = 2,
        sample_rate: int = 48000,
        n_layers: int = 12,
        n_heads: int = 12,
        dim: int = 768,
        rope_len: int = 8192,
        n_bands: int = 112,
        n_fft: int = 16,
        hop_length: int = 4,
        patch_size_t: int = 4,
        max_bandwidth: int = 390,
        chunk_size: int = 16,
        pre_post_depth: int = 1,
        pre_post_heads: int = 4,
        skip_connection_type: str = "add",
        **kwargs,
    ) -> None:
        super().__init__()

        if "patch_size" in kwargs:
            patch_size_t = int(kwargs.pop("patch_size")[0])
        del kwargs

        self.audio_channels = audio_channels
        self.dim = dim
        self.hop_length = int(hop_length)
        self.n_fft = n_fft
        self.patch_size_t = patch_size_t
        self.total_stride_t = self.patch_size_t
        self.skip_connection_type = skip_connection_type
        self._last_cnn_input_length: int | None = None

        factor = max(sample_rate // 400, 1)
        self.sb_factor = factor

        banks = erb_linear_banks(
            sr=sample_rate,
            n_bands=n_bands,
            max_bandwidth=max_bandwidth,
        )
        self.sb_filter = SubbandFilter(
            sample_rate, banks, factor, chunk_size=chunk_size
        )

        self.cnn_encoder = nn.Conv2d(
            audio_channels * 2,
            dim,
            kernel_size=(self.hop_length, 1),
            stride=(self.hop_length, 1),
            bias=False,
        )
        self.cnn_decoder = nn.ConvTranspose2d(
            dim,
            audio_channels * 2,
            kernel_size=(self.hop_length, 1),
            stride=(self.hop_length, 1),
            bias=False,
        )

        self.pre_block = FreqTransformerBlock(
            channels=dim,
            depth=pre_post_depth,
            n_heads=pre_post_heads,
        )
        self.patch = Patch(dim, dim, (self.patch_size_t, 1))
        self.unpatch = UnPatch(dim, dim, (self.patch_size_t, 1))
        self.post_block = FreqTransformerBlock(
            channels=dim,
            depth=pre_post_depth,
            n_heads=pre_post_heads,
            skip_connection_type=skip_connection_type,
        )

        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)
        self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.k_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

    def forward(self, audio: Tensor) -> Tensor:
        audio_length = audio.shape[-1]
        audio = self.pad_audio(audio, self.sb_factor)

        subbands = self.sb_filter.analysis(audio)
        features = self.cnn_analysis(subbands)

        if False:
            self.check_sdr(audio, features)
            os._exit(0)

        batch_size, _, frames_num, _ = features.shape
        x = self.pad_tensor(features, self.total_stride_t)
        skip = self.pre_block(x)
        x = self.patch(skip)

        for t_block, k_block in zip(self.t_blocks, self.k_blocks):
            x = rearrange(x, "b d t f -> (b f) t d")
            x = t_block(x, rope=self.rope, pos=None)

            x = rearrange(x, "(b f) t d -> (b t) f d", b=batch_size)
            x = k_block(x, rope=self.rope, pos=None)

            x = rearrange(x, "(b t) f d -> b d t f", b=batch_size)

        x = self.unpatch(x)
        x = self.post_block(x, skip)
        mask = x[:, :, 0:frames_num, :]
        x = features * mask

        x = self.cnn_synthesis(x)
        out = self.sb_filter.synthesis(x)
        return self._trim_or_pad_output(out, audio_length)

    def cnn_analysis(self, x: Tensor) -> Tensor:
        self._last_cnn_input_length = x.shape[-1]
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        x = torch.view_as_real(x.contiguous())
        x = rearrange(x, "b c k l ri -> b (c ri) l k")
        x = self._pad_for_cnn_analysis(x)
        return self.cnn_encoder(x)

    def cnn_synthesis(self, x: Tensor) -> Tensor:
        target_length = self._last_cnn_input_length
        if target_length is None:
            raise RuntimeError("cnn_analysis() must run before cnn_synthesis()")

        x = self.cnn_decoder(x)
        x = x[:, :, 0:target_length, :]
        if x.shape[2] < target_length:
            x = F.pad(x, pad=(0, 0, 0, target_length - x.shape[2]))
        x = rearrange(
            x,
            "b (c ri) l k -> b c k l ri",
            c=self.audio_channels,
            ri=2,
        )
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

    def pad_tensor(self, x: Tensor, multiple_t: int) -> Tensor:
        pad_t = -x.shape[2] % multiple_t
        return F.pad(x, pad=(0, 0, 0, pad_t))

    def pad_audio(self, audio: Tensor, multiple_l: int) -> Tensor:
        pad_l = -audio.shape[-1] % multiple_l
        return F.pad(audio, pad=(0, pad_l))

    def _trim_or_pad_output(self, out: Tensor, audio_length: int) -> Tensor:
        out = out[..., 0:audio_length]
        if out.shape[-1] < audio_length:
            out = F.pad(out, pad=(0, audio_length - out.shape[-1]))
        return out

    def check_sdr(self, audio: Tensor, features: Tensor) -> None:
        y = self.cnn_synthesis(features)
        y = self.sb_filter.synthesis(y)
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"SDR: {sdr:.2f} dB")
