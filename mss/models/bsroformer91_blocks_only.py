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


class BSRoformer91BlocksOnly(nn.Module):
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

        del patch_size_t, pre_post_depth, pre_post_heads, skip_connection_type, kwargs

        self.n_fft = n_fft
        self.hop_length = hop_length

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

        in_channels = audio_channels * self.n_fft * 2
        self.input_proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.output_proj = nn.Conv2d(dim, in_channels, kernel_size=1)

        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)
        self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.k_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

    def forward(self, audio: Tensor) -> Tensor:
        audio_length = audio.shape[-1]
        audio = self.pad_audio(audio, self.sb_factor)

        x = self.sb_filter.analysis(audio)
        complex_sp = self.stft(x)

        if False:
            self.check_sdr(audio, complex_sp)
            os._exit(0)

        batch_size, audio_channels, _, frames_num = complex_sp.shape[0:4]
        x = rearrange(torch.view_as_real(complex_sp), "b c k t f x -> b (c f x) t k")
        x = self.input_proj(x)

        for t_block, k_block in zip(self.t_blocks, self.k_blocks):
            x = rearrange(x, "b d t f -> (b f) t d")
            x = t_block(x, rope=self.rope, pos=None)

            x = rearrange(x, "(b f) t d -> (b t) f d", b=batch_size)
            x = k_block(x, rope=self.rope, pos=None)

            x = rearrange(x, "(b t) f d -> b d t f", b=batch_size)

        x = self.output_proj(x)
        x = x[:, :, 0:frames_num, :]
        x = rearrange(x, "b (c f x) t k -> b c k t f x", c=audio_channels, x=2)
        mask = torch.view_as_complex(x.contiguous().float())
        sep_stft = complex_sp * mask

        x = self.istft(sep_stft)
        out = self.sb_filter.synthesis(x)
        return self._trim_or_pad_output(out, audio_length)

    def _trim_or_pad_output(self, out: Tensor, audio_length: int) -> Tensor:
        out = out[..., 0:audio_length]
        if out.shape[-1] < audio_length:
            out = F.pad(out, pad=(0, audio_length - out.shape[-1]))
        return out

    def stft(self, x: Tensor) -> Tensor:
        batch_size, audio_channels = x.shape[0:2]
        x = rearrange(x, "b c k l -> (b c k) l")
        x = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=x.device),
            normalized=True,
            onesided=False,
            return_complex=True,
        )
        x = rearrange(x, "(b c k) f t -> b c k t f", b=batch_size, c=audio_channels)
        return x

    def istft(self, x: Tensor) -> Tensor:
        batch_size, audio_channels = x.shape[0:2]
        x = rearrange(x, "b c k t f -> (b c k) f t")
        x = torch.istft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=x.device),
            normalized=True,
            onesided=False,
            return_complex=True,
        )
        x = rearrange(x, "(b c k) l -> b c k l", b=batch_size, c=audio_channels)
        return x

    def pad_audio(self, audio: Tensor, multiple_l: int) -> Tensor:
        pad_l = -audio.shape[-1] % multiple_l
        return F.pad(audio, pad=(0, pad_l))

    def check_sdr(self, audio: Tensor, complex_sp: Tensor) -> None:
        y = self.istft(complex_sp)
        y = self.sb_filter.synthesis(y)
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"SDR: {sdr:.2f} dB")
