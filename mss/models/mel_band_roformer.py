from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from einops import rearrange
from torch import Tensor

from mss.models.attention import Block, RMSNorm
from mss.models.bsroformer91 import Patch, UnPatch
from mss.models.rope import RoPE


def _to_2tuple(value: int | list[int] | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, 1)
    if len(value) == 1:
        return (int(value[0]), 1)
    return (int(value[0]), int(value[1]))


class BandSplit(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...]) -> None:
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList(
            nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))
            for dim_in in dim_inputs
        )

    def forward(self, x: Tensor) -> Tensor:
        splits = x.split(self.dim_inputs, dim=-1)
        return torch.stack(
            [to_feature(split) for split, to_feature in zip(splits, self.to_features)],
            dim=-2,
        )


class MaskEstimator(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_inputs: tuple[int, ...],
        depth: int = 1,
        expansion_factor: int = 1,
    ) -> None:
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.to_freqs = nn.ModuleList()

        for dim_in in dim_inputs:
            layers: list[nn.Module] = []
            in_dim = dim
            for _ in range(max(depth, 0)):
                layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
                in_dim = hidden_dim
            layers += [nn.Linear(in_dim, dim_in * 2), nn.GLU(dim=-1)]
            self.to_freqs.append(nn.Sequential(*layers))

    def forward(self, x: Tensor) -> Tensor:
        band_features = x.unbind(dim=-2)
        return torch.cat(
            [to_freqs(band) for band, to_freqs in zip(band_features, self.to_freqs)],
            dim=-1,
        )


class MelBandRoformer(nn.Module):
    def __init__(
        self,
        audio_channels: int = 2,
        sample_rate: int = 48000,
        num_stems: int = 1,
        n_fft: int = 2048,
        hop_length: int = 480,
        win_length: int | None = None,
        n_bands: int = 64,
        dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        patch_size: int | list[int] | tuple[int, int] = (4, 4),
        mask_estimator_depth: int = 1,
        mask_mlp_expansion_factor: int = 1,
        zero_dc: bool = True,
        normalized: bool = True,
        rope_len: int = 8192,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        self.audio_channels = audio_channels
        self.sample_rate = sample_rate
        self.num_stems = num_stems
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.n_bands = n_bands
        self.dim = dim
        self.zero_dc = zero_dc
        self.normalized = normalized
        self.patch_size = _to_2tuple(patch_size)

        self.stft_kwargs = dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=self.win_length,
            normalized=normalized,
            onesided=True,
            return_complex=True,
        )

        freq_bins = n_fft // 2 + 1
        mel_filter_bank = AF.melscale_fbanks(
            n_freqs=freq_bins,
            f_min=0.0,
            f_max=float(sample_rate // 2),
            n_mels=n_bands,
            sample_rate=sample_rate,
            norm=None,
            mel_scale="htk",
        ).transpose(0, 1)
        mel_filter_bank[0, 0] = 1.0
        mel_filter_bank[-1, -1] = 1.0

        freqs_per_band = mel_filter_bank > 0
        if not freqs_per_band.any(dim=0).all():
            raise ValueError(
                "All STFT frequency bins must be covered by at least one mel band"
            )

        repeated_freq_indices = torch.arange(freq_bins).repeat(n_bands, 1)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if audio_channels > 1:
            freq_indices = (
                freq_indices[:, None] * audio_channels
                + torch.arange(audio_channels)[None, :]
            ).flatten()

        num_freqs_per_band = freqs_per_band.sum(dim=-1)
        num_bands_per_freq = freqs_per_band.sum(dim=0).float()

        self.register_buffer("freq_indices", freq_indices.long(), persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)
        self.register_buffer(
            "num_freqs_per_band", num_freqs_per_band.long(), persistent=False
        )
        self.register_buffer(
            "num_bands_per_freq", num_bands_per_freq, persistent=False
        )

        dim_inputs = tuple(
            int(2 * audio_channels * freq_count.item())
            for freq_count in num_freqs_per_band
        )

        self.band_split = BandSplit(dim=dim, dim_inputs=dim_inputs)
        self.patch = Patch(dim, dim, self.patch_size)
        self.unpatch = UnPatch(dim, dim, self.patch_size)

        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)
        self.layers = nn.ModuleList(
            nn.ModuleList([Block(dim, n_heads), Block(dim, n_heads)])
            for _ in range(n_layers)
        )
        self.mask_estimators = nn.ModuleList(
            MaskEstimator(
                dim=dim,
                dim_inputs=dim_inputs,
                depth=mask_estimator_depth,
                expansion_factor=mask_mlp_expansion_factor,
            )
            for _ in range(num_stems)
        )

    def forward(self, audio: Tensor) -> Tensor:
        if audio.ndim != 3:
            raise ValueError(
                f"Expected audio shape (batch, channels, samples), got {audio.shape}"
            )
        if audio.shape[1] != self.audio_channels:
            raise ValueError(
                f"Expected {self.audio_channels} channels, got {audio.shape[1]}"
            )

        batch_size, channels, audio_length = audio.shape
        window = torch.hann_window(self.win_length, device=audio.device)

        x = rearrange(audio, "b c l -> (b c) l")
        complex_sp = torch.stft(x, window=window, **self.stft_kwargs)
        complex_sp = rearrange(
            complex_sp, "(b c) f t -> b c f t", b=batch_size, c=channels
        )

        stft_repr = torch.view_as_real(complex_sp)
        stft_repr = rearrange(stft_repr, "b c f t ri -> b (f c) t ri")

        batch_arange = torch.arange(batch_size, device=audio.device)[:, None]
        x = stft_repr[batch_arange, self.freq_indices]
        x = rearrange(x, "b f t ri -> b t (f ri)")
        x = self.band_split(x)

        time_steps, bands = x.shape[1:3]
        x = rearrange(x, "b t f d -> b d t f")
        x = self.pad_tensor(x, self.patch_size)
        x = self.patch(x)

        for time_block, freq_block in self.layers:
            x = rearrange(x, "b d t f -> (b f) t d")
            x = time_block(x, rope=self.rope, pos=None)
            x = rearrange(x, "(b f) t d -> (b t) f d", b=batch_size)
            x = freq_block(x, rope=self.rope, pos=None)
            x = rearrange(x, "(b t) f d -> b d t f", b=batch_size)

        x = self.unpatch(x)
        x = x[:, :, :time_steps, :bands]
        x = rearrange(x, "b d t f -> b t f d")

        masks = torch.stack(
            [mask_estimator(x) for mask_estimator in self.mask_estimators],
            dim=1,
        )
        masks = rearrange(masks, "b n t (f ri) -> b n f t ri", ri=2)

        stft_repr = rearrange(stft_repr, "b f t ri -> b 1 f t ri")
        stft_complex = torch.view_as_complex(stft_repr.contiguous())
        masks_complex = torch.view_as_complex(masks.contiguous().float()).type(
            stft_complex.dtype
        )

        scatter_indices = self.freq_indices.view(1, 1, -1, 1).expand(
            batch_size, self.num_stems, -1, stft_complex.shape[-1]
        )
        stem_stft = stft_complex.expand(-1, self.num_stems, -1, -1)
        masks_summed = torch.zeros_like(stem_stft).scatter_add_(
            dim=2,
            index=scatter_indices,
            src=masks_complex,
        )
        denom = self.num_bands_per_freq.repeat_interleave(channels).view(
            1, 1, -1, 1
        )
        masks_averaged = masks_summed / denom.clamp(min=1e-8)
        separated = stem_stft * masks_averaged

        separated = rearrange(
            separated,
            "b n (f c) t -> (b n c) f t",
            c=self.audio_channels,
        )
        if self.zero_dc:
            separated = separated.index_fill(
                1,
                torch.tensor(0, device=audio.device),
                0,
            )

        audio_out = torch.istft(
            separated,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            normalized=self.normalized,
            onesided=True,
            window=window,
            length=audio_length,
        )
        audio_out = rearrange(
            audio_out,
            "(b n c) l -> b n c l",
            b=batch_size,
            n=self.num_stems,
            c=self.audio_channels,
        )
        if self.num_stems == 1:
            audio_out = rearrange(audio_out, "b 1 c l -> b c l")
        return audio_out

    def pad_tensor(self, x: Tensor, patch_size: tuple[int, int]) -> Tensor:
        pad_t = -x.shape[2] % patch_size[0]
        pad_f = -x.shape[3] % patch_size[1]
        return F.pad(x, pad=(0, pad_f, 0, pad_t))
