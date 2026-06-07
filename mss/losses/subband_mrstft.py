from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models2.dsp3.banks import mel_linear_banks_triangle
from mss.models2.dsp3.subband_fast_triangle import SubbandFilter


SUBBAND_BANKS = (
    {
        "bins": 64,
        "n_bands": 58,
        "max_half_bandwidth": 780,
        "factor": 30,
        "window_sizes": (8, 16, 32, 64, 128),
    },
    {
        "bins": 128,
        "n_bands": 118,
        "max_half_bandwidth": 390,
        "factor": 60,
        "window_sizes": (4, 8, 16, 32, 64),
    },
    {
        "bins": 256,
        "n_bands": 237,
        "max_half_bandwidth": 195,
        "factor": 120,
        "window_sizes": (2, 4, 8, 16, 32),
    },
)


class MultiSubbandL1Loss(nn.Module):
    r"""L1 loss on multi-resolution triangular subband features."""

    def __init__(
        self,
        sample_rate: int = 48000,
        chunk_size: int = 16,
        bandpass_filter_len: int = 48000,
        upsample_filter_len: int = 12000,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.filters = nn.ModuleList()

        for bank_config in SUBBAND_BANKS:
            banks = mel_linear_banks_triangle(
                sr=sample_rate,
                n_bands=bank_config["n_bands"],
                max_bandwidth=bank_config["max_half_bandwidth"],
            )
            if len(banks) != bank_config["bins"]:
                raise ValueError(
                    f"Expected {bank_config['bins']} subband bins, got {len(banks)}"
                )

            self.filters.append(
                SubbandFilter(
                    sample_rate,
                    banks,
                    bank_config["factor"],
                    chunk_size=chunk_size,
                    bandpass_filter_len=bandpass_filter_len,
                    upsample_filter_len=upsample_filter_len,
                )
            )

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        self._ensure_device(output.device)

        output = output.float()
        target = target.float()

        loss = output.new_tensor(0.0)
        for sb_filter in self.filters:
            output_pad = self.pad_audio(output, sb_filter.factor)
            target_pad = self.pad_audio(target, sb_filter.factor)
            output_sb = sb_filter.analysis(output_pad)
            target_sb = sb_filter.analysis(target_pad)
            loss = loss + (output_sb - target_sb).abs().mean()

        return loss / len(self.filters)

    def _ensure_device(self, device: torch.device) -> None:
        first_buffer = next(self.buffers(), None)
        if first_buffer is not None and first_buffer.device != device:
            self.to(device)

    def pad_audio(self, x: Tensor, factor: int) -> Tensor:
        pad = -x.shape[-1] % factor
        if pad == 0:
            return x
        return F.pad(x, (0, pad))


class MultiSubbandMRSTFTLoss(MultiSubbandL1Loss):
    r"""MR-STFT loss on multi-resolution triangular subband features."""

    def __init__(
        self,
        sample_rate: int = 48000,
        chunk_size: int = 16,
        bandpass_filter_len: int = 48000,
        upsample_filter_len: int = 12000,
        normalized: bool = True,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            bandpass_filter_len=bandpass_filter_len,
            upsample_filter_len=upsample_filter_len,
        )
        self.normalized = normalized

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        self._ensure_device(output.device)

        output = output.float()
        target = target.float()

        loss = output.new_tensor(0.0)
        for bank_config, sb_filter in zip(SUBBAND_BANKS, self.filters):
            output_pad = self.pad_audio(output, sb_filter.factor)
            target_pad = self.pad_audio(target, sb_filter.factor)
            output_sb = sb_filter.analysis(output_pad)
            target_sb = sb_filter.analysis(target_pad)

            bank_loss = output.new_tensor(0.0)
            hop_length = max(round(147 / sb_filter.factor), 1)
            for window_size in bank_config["window_sizes"]:
                output_stft = self.stft(output_sb, window_size, hop_length)
                target_stft = self.stft(target_sb, window_size, hop_length)
                bank_loss = bank_loss + (output_stft - target_stft).abs().mean()

            loss = loss + bank_loss / len(bank_config["window_sizes"])

        return loss / len(self.filters)

    def stft(self, x: Tensor, n_fft: int, hop_length: int) -> Tensor:
        B, C = x.shape[0:2]
        x = rearrange(x, "b c k l -> (b c k) l")
        x = torch.stft(
            input=x,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=x.device, dtype=x.real.dtype),
            normalized=self.normalized,
            onesided=False,
            return_complex=True,
        )
        x = rearrange(x, "(b c k) f t -> b c k t f", b=B, c=C)
        return x
