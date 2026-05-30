from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from torch import LongTensor, Tensor
import math

import librosa
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class MultidimRoPE(nn.Module):
    r"""Multi-dimensional Rotary Position Embedding.

    Strictly 2D RoPE with LLaMA-style precomputed cos/sin cache.
    """

    def __init__(
        self,
        head_dim: int,
        n_dims: int = 2,
        max_len: int = 8192,
        base: int = 10000,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.n_dims = 2
        self.max_len = max_len
        self.base = base

        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by 4 for vanilla 2D RoPE"
            )

        self.axis_dim = head_dim // 2
        self.rotary_pairs = self.axis_dim // 2
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self.max_seq_len_cached = 0
        self._set_cache(max_len, device=torch.device("cpu"))

    def _set_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device=device))
        self.cos_cached = freqs.cos()
        self.sin_cached = freqs.sin()
        self.max_seq_len_cached = seq_len

    def _maybe_update_cache(self, needed_len: int, device: torch.device) -> None:
        if needed_len <= self.max_seq_len_cached and self.cos_cached.device == device:
            return

        new_len = max(
            needed_len,
            self.max_seq_len_cached * 2 if self.max_seq_len_cached else needed_len,
        )
        self._set_cache(new_len, device=device)

    def build_grid_positions(
        self, time_steps: int, freq_bins: int, device: torch.device
    ) -> tuple[LongTensor, LongTensor]:
        time_idx = torch.arange(time_steps, dtype=torch.long)
        freq_idx = torch.arange(freq_bins, dtype=torch.long)

        time_pos = torch.stack(
            (
                time_idx.view(1, time_steps).expand(freq_bins, -1),
                freq_idx.view(freq_bins, 1).expand(-1, time_steps),
            ),
            dim=-1,
        ).reshape(freq_bins * time_steps, 2)

        freq_pos = torch.stack(
            (
                time_idx.view(time_steps, 1).expand(-1, freq_bins),
                freq_idx.view(1, freq_bins).expand(time_steps, -1),
            ),
            dim=-1,
        ).reshape(time_steps * freq_bins, 2)

        return time_pos.to(device), freq_pos.to(device)

    def forward(self, x: Tensor, shape: tuple[int, ...] | None = None) -> Tensor:
        if shape is None:
            raise ValueError("2D RoPE requires an explicit (time, frequency) shape")
        if len(shape) != 2:
            raise ValueError(f"Expected 2D shape, got {shape}")

        time_steps, freq_bins = shape
        time_pos, freq_pos = self.build_grid_positions(time_steps, freq_bins, x.device)
        pos = torch.stack((time_pos[:, 0], freq_pos[:, 1]), dim=-1)
        return self.apply_nd(x, pos)

    def apply_nd(self, x: Tensor, pos: LongTensor) -> Tensor:
        B, L, N, H = x.shape

        if pos.ndim == 2:
            pos = pos.unsqueeze(0).expand(B, -1, -1)
        elif pos.ndim != 3:
            raise ValueError(f"Expected pos to have 2 or 3 dims, got shape {pos.shape}")

        if pos.shape[0] != B or pos.shape[1] != L:
            raise ValueError(
                f"Position shape {tuple(pos.shape)} does not match input batch/length {(B, L)}"
            )

        if H != self.head_dim:
            raise ValueError(f"Expected head dim {self.head_dim}, got {H}")

        needed_len = int(pos.max().item()) + 1
        self._maybe_update_cache(needed_len, x.device)

        x_time, x_freq = rearrange(x, "b l n (k d) -> k b l n d", k=2)
        out = []
        for axis_idx, x_axis in enumerate((x_time, x_freq)):
            axis_pos = pos[:, :, axis_idx].clamp(max=self.max_seq_len_cached - 1)
            cos = self.cos_cached[axis_pos].to(dtype=x.dtype)[:, :, None, :]
            sin = self.sin_cached[axis_pos].to(dtype=x.dtype)[:, :, None, :]
            x_axis = rearrange(x_axis, "b l n (h c) -> b l n h c", c=2)
            out.append(
                torch.stack(
                    [
                        x_axis[..., 0] * cos - x_axis[..., 1] * sin,
                        x_axis[..., 0] * sin + x_axis[..., 1] * cos,
                    ],
                    dim=-1,
                ).flatten(-2)
            )

        return torch.cat(out, dim=-1)


class RoPE1D(nn.Module):
    r"""Vanilla cached 1D RoPE for frequency attention."""

    def __init__(self, head_dim: int, max_len: int = 8192, base: int = 10000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be even for 1D RoPE")

        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base
        theta = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", theta, persistent=True)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self.max_seq_len_cached = 0
        self._set_cache(max_len, device=torch.device("cpu"))

    def _set_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device=device))
        self.cos_cached = freqs.cos()
        self.sin_cached = freqs.sin()
        self.max_seq_len_cached = seq_len

    def _maybe_update_cache(self, needed_len: int, device: torch.device) -> None:
        if needed_len <= self.max_seq_len_cached and self.cos_cached.device == device:
            return
        new_len = max(
            needed_len,
            self.max_seq_len_cached * 2 if self.max_seq_len_cached else needed_len,
        )
        self._set_cache(new_len, device=device)

    def forward(self, x: Tensor) -> Tensor:
        b, l, n, h = x.shape
        self._maybe_update_cache(l, x.device)

        x = rearrange(x, "b l n (h c) -> b l n h c", c=2)
        cos = self.cos_cached[:l].to(dtype=x.dtype)[None, :, None, :]
        sin = self.sin_cached[:l].to(dtype=x.dtype)[None, :, None, :]
        out = torch.stack(
            [
                x[..., 0] * cos - x[..., 1] * sin,
                x[..., 0] * sin + x[..., 1] * cos,
            ],
            dim=-1,
        )
        return rearrange(out, "b l n h c -> b l n (h c)")


# ============== Fourier ==============


class Fourier(nn.Module):
    def __init__(
        self, n_fft=2048, hop_length=480, return_complex=True, normalized=True
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.return_complex = return_complex
        self.normalized = normalized

        self.register_buffer(name="window", tensor=torch.hann_window(self.n_fft))

    def stft(self, waveform: Tensor) -> Tensor:
        r"""Compute STFT of waveforms.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: freq_bins

        Args:
            waveform: (b, c, l)

        Returns:
            complex_sp: (b, c, t, f)
        """

        B, C, T = waveform.shape

        x = rearrange(waveform, "b c l -> (b c) l")  # (b*c, l)

        x = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            normalized=self.normalized,
            return_complex=self.return_complex,
        )  # (b*c, f, t)

        complex_sp = rearrange(x, "(b c) f t -> b c t f", b=B, c=C)  # (b, c, t, f)
        return complex_sp

    def istft(self, complex_sp: Tensor) -> Tensor:
        r"""Reconstruct waveforms from STFT.

        b: batch_size
        c: channels_num
        t: frames_num
        f: freq_bins
        l: audio_samples

        Args:
            complex_sp: (b, c, t, f)

        Returns:
            out: (b, c, l)
        """

        B, C, T, F = complex_sp.shape

        x = rearrange(complex_sp, "b c t f -> (b c) f t")  # (b*c, f, t)

        x = torch.istft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            normalized=self.normalized,
        )  # (b*c, l)

        out = rearrange(x, "(b c) l -> b c l", b=B, c=C)  # (b, c, l)

        return out


# ============== BandSplit ==============


class BandSplit(nn.Module):
    def __init__(
        self,
        sr: float,
        n_fft: int,
        n_bands: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        r"""Band split STFT to mel scale STFT.

        f: stft bins
        k: mel bins
        w: band width

        Args:
            sr: Sample rate
            n_fft: FFT size
            n_bands: Number of frequency bands
            in_channels: Number of input channels (2 for real/imag)
            out_channels: Number of output channels
        """

        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.n_bands = n_bands
        self.in_channels = in_channels
        self.out_channels = out_channels

        F = n_fft // 2 + 1

        melbanks_init, ola_window_init = self.init_melbanks()
        melbanks_init = Tensor(melbanks_init)
        ola_window_init = Tensor(ola_window_init)

        self.register_buffer("melbanks", melbanks_init)
        self.register_buffer("ola_window", ola_window_init)

        self.pre_w = nn.Parameter(torch.zeros((in_channels, F, out_channels)))
        self.post_w = nn.Parameter(torch.zeros((out_channels, in_channels, F)))
        self.pre_b = nn.Parameter(torch.zeros((n_bands, out_channels)))
        self.post_b = nn.Parameter(torch.zeros((in_channels, F)))

        gain = np.sqrt(
            np.linalg.pinv(in_channels * melbanks_init.cpu().numpy() ** 2)
            @ np.ones(n_bands)
        )
        gain = torch.from_numpy(gain[None, :, None])

        nn.init.uniform_(self.pre_w, -1, 1)
        with torch.no_grad():
            self.pre_w *= gain

        bound = 1 / math.sqrt(out_channels)
        nn.init.uniform_(self.post_w, -bound, bound)

        nonzero_indexes = []
        nonzero_melbanks = []

        for f in range(n_bands):
            idxes = torch.nonzero(melbanks_init[f].abs() > 1e-6, as_tuple=True)[0]
            nonzero_indexes.append(idxes)
            nonzero_melbanks.append(melbanks_init[f, idxes])

        Q = 4
        self.Q = Q
        cumsum = np.cumsum([len(idxes) for idxes in nonzero_indexes])
        total = cumsum[-1]
        subbands = []

        for q in range(Q):
            subband = []
            for i in range(n_bands):
                if q / Q * total <= cumsum[i] < (q + 1) / Q * total:
                    subband.append(i)
            subbands.append(subband)
        subbands[-1].append(i)

        for q in range(Q):
            sb_idxes = []
            sb_melbanks = []
            for f in subbands[q]:
                sb_idxes.append(nonzero_indexes[f])
                sb_melbanks.append(nonzero_melbanks[f])

            sb_idxes = pad_sequence(
                sequences=sb_idxes, batch_first=True, padding_value=-1
            )
            sb_masks = (sb_idxes != -1).float()
            sb_idxes[sb_idxes == -1] = n_fft // 2
            sb_melbanks = pad_sequence(
                sequences=sb_melbanks, batch_first=True, padding_value=0
            )

            self.register_buffer(name=f"sb_idxes_{q}", tensor=sb_idxes)
            self.register_buffer(name=f"sb_melbanks_{q}", tensor=sb_melbanks)
            self.register_buffer(name=f"sb_masks_{q}", tensor=sb_masks)
            self.register_buffer(
                name=f"sb_subbands_{q}", tensor=LongTensor(subbands[q])
            )

    def init_melbanks(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Initialize mel bins from librosa.

        f: stft bins
        k: mel bins

        Args:
            None

        Returns:
            melbanks: (k, f)
            ola_window: (f,)
        """

        melbanks = librosa.filters.mel(
            sr=self.sr, n_fft=self.n_fft, n_mels=self.n_bands - 2, norm=None
        )  # shape: (k, f)

        F = self.n_fft // 2 + 1

        # The zeroth bank, e.g., [1., 0.66, 0.32, 0, ..., 0.]
        melbank_0 = np.zeros(F)
        idx = np.argmax(melbanks[0])
        melbank_0[0:idx] = 1.0 - melbanks[0, 0:idx]  # (f,)

        # The last bank, e.g., [0., ..., 0., 0.18, 0.87, 1.]
        melbank_last = np.zeros(F)
        idx = np.argmax(melbanks[-1])
        melbank_last[idx:] = 1.0 - melbanks[-1, idx:]  # (f,)

        # Concatenate
        melbanks = np.concatenate(
            [melbank_0[None, :], melbanks, melbank_last[None, :]], axis=0
        )  # (n_mels, f)

        # Calculate overlap-add window
        ola_window = np.sum(melbanks, axis=0)  # overlap add window
        assert ola_window.max() >= 0.5

        return melbanks, ola_window

    def transform(self, x: Tensor) -> Tensor:
        r"""Convert STFT to mel scale STFT.

        b: batch_size
        c: channels_num
        d: latent_dim
        t: frames_num
        k: mel_bins
        w: max_bank_width

        Args:
            x: (b, c, t, f, i)

        Returns:
            x: (b, c, t, k, o)
        """
        return self._transform_sparse(x)

    def _transform_sparse(self, x: Tensor) -> Tensor:
        B, C, T, F_, I = x.shape
        O_ = self.out_channels
        out = torch.zeros((B, C, T, self.n_bands, O_), device=x.device)

        for q in range(self.Q):
            subbands = getattr(self, f"sb_subbands_{q}")
            idxes = getattr(self, f"sb_idxes_{q}")
            melbanks = getattr(self, f"sb_melbanks_{q}")
            masks = getattr(self, f"sb_masks_{q}")

            _x = x[..., idxes, :] * melbanks[..., :, :, None] * masks[..., :, :, None]
            _w = self.pre_w[:, idxes, :] * masks[None, :, :, None]
            _b = self.pre_b[subbands, :]
            _y = torch.einsum("bctswi,iswo->bctso", _x, _w) + _b
            out[:, :, :, subbands, :] = _y

        return out

    def inverse_transform(self, x: Tensor) -> Tensor:
        r"""Convert mel scale STFT to STFT.

        b: batch_size
        c: channels_num
        d: latent_dim
        t: frames_num
        k: mel_bins
        w: max_bank_width

        Args:
            x: (b, c, t, k, o)

        Outputs:
            y: (b, c, t, f, i)
        """
        return self._inverse_transform_sparse(x)

    def _inverse_transform_sparse(self, x: Tensor) -> Tensor:
        B, C, T, F, O_ = x.shape
        I = self.in_channels
        out = torch.zeros((B, C, T, self.n_fft // 2 + 1, I), device=x.device)

        for q in range(self.Q):
            subbands = getattr(self, f"sb_subbands_{q}")
            idxes = getattr(self, f"sb_idxes_{q}")
            melbanks = getattr(self, f"sb_melbanks_{q}")
            masks = getattr(self, f"sb_masks_{q}")

            _x = x[:, :, :, subbands, :]
            _w = self.post_w[:, :, idxes] * masks
            _b = self.post_b[:, idxes] * masks
            _b = rearrange(_b, "i s w -> s w i")
            _y = torch.einsum("bctso,oisw->bctswi", _x, _w) + _b
            _y = _y * melbanks[..., :, :, None] * masks[..., :, :, None]
            out.scatter_add_(
                dim=3,
                index=idxes.flatten()[None, None, None, :, None].repeat(B, C, T, 1, I),
                src=_y.flatten(3, 4),
            )

        out = out / self.ola_window[..., :, None].clamp(min=1e-8)
        return out


# ============== Attention ==============

# Optional flash_attn import
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# Optional flash linear attention import
try:
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    FLASH_LINEAR_ATTENTION_AVAILABLE = True
except ImportError:
    chunk_gated_delta_rule = None
    FusedRMSNormGated = None
    FLASH_LINEAR_ATTENTION_AVAILABLE = False


class AttentionBlock(nn.Module):
    r"""Attention block with GatedAttention and SwiGLU MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int | None = None,
        attention_dropout: float = 0.0,
        n_bands: int | None = None,
        qk_norm_n_bands: int | None = None,
        norm_layout: Literal["batch_band", "seq_band"] = "batch_band",
        qk_norm_layout: Literal["batch_band", "seq_band"] = "batch_band",
    ) -> None:
        super().__init__()

        Norm = (
            (lambda d: BandGatedRMSNorm(d, n_bands)) if n_bands is not None else RMSNorm
        )
        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)
        self.norm_layout = norm_layout

        self.attn = GatedAttention(
            hidden_size=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            qk_norm_n_bands=qk_norm_n_bands,
            qk_norm_layout=qk_norm_layout,
        )

        self.ffn = GatedMLP(dim, intermediate_dim=dim * 4)

    def forward(
        self,
        x: Tensor,
        rope: RoPE | None = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        r"""Attention block forward.

        Args:
            x: (b, l, d)
            rope: RoPE module for position embeddings
            pos: (l, 2) or (b, l, 2) position indices for 2D RoPE

        Outputs:
            out: (b, l, d)
        """
        norm1 = (
            self.norm1(x, layout=self.norm_layout)
            if isinstance(self.norm1, BandGatedRMSNorm)
            else self.norm1(x)
        )
        x = x + self.attn(norm1, rope=rope, pos=pos)

        norm2 = (
            self.norm2(x, layout=self.norm_layout)
            if isinstance(self.norm2, BandGatedRMSNorm)
            else self.norm2(x)
        )
        x = x + self.ffn(norm2)
        return x


class DeltaNetBlock(nn.Module):
    r"""DeltaNet block with GatedDeltaNet and SwiGLU MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int = 64,
        conv_kernel_size: int = 7,
        n_bands: int | None = None,
    ) -> None:
        super().__init__()

        Norm = (
            (lambda d: BandGatedRMSNorm(d, n_bands)) if n_bands is not None else RMSNorm
        )
        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)

        self.deltanet = GatedDeltaNet(
            hidden_size=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            conv_kernel_size=conv_kernel_size,
            activation="silu",
            n_bands=n_bands,
        )

        self.ffn = GatedMLP(dim, intermediate_dim=dim * 4)

    def forward(self, x: Tensor) -> Tensor:
        r"""DeltaNet block forward.

        Args:
            x: (b, l, d)

        Outputs:
            out: (b, l, d)
        """
        norm1 = (
            self.norm1(x, layout="batch_band")
            if isinstance(self.norm1, BandGatedRMSNorm)
            else self.norm1(x)
        )
        x = x + self.deltanet(norm1)

        norm2 = (
            self.norm2(x, layout="batch_band")
            if isinstance(self.norm2, BandGatedRMSNorm)
            else self.norm2(x)
        )
        x = x + self.ffn(norm2)
        return x


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization.

    Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""RMSNorm.

        Args:
            x: (b, t, d)

        Outputs:
            x: (b, t, d)
        """

        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.scale
        return output


class BandRMSNorm(nn.Module):
    r"""Band-aware RMSNorm with per-band learnable scale.

    Standard RMSNorm uses a single scale vector shared across all bands.
    This variant learns a separate scale per frequency band, allowing each
    band to adapt its normalization independently.

    When used in time attention blocks, the input has shape (b*f, t, d)
    where the band dimension f is folded into the batch. We unfold it,
    apply per-band scale, then refold.
    """

    def __init__(self, dim: int, n_bands: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.n_bands = n_bands
        self.scale = nn.Parameter(torch.ones(n_bands, dim))  # (f, d)

    def forward(self, x: Tensor) -> Tensor:
        r"""Band-aware RMSNorm.

        Supports two input layouts:
            3D: (b*f, t, d) — time attention path, bands folded into batch
            4D: (b, d, t, f) — conv/spatial path, bands as last dim

        Args:
            x: (b*f, t, d) or (b, d, t, f)

        Outputs:
            out: same shape as input
        """
        if x.ndim == 3:
            # (b*f, t, d)
            BF, T, D = x.shape
            F = self.n_bands
            B = BF // F

            norm_x = torch.mean(x**2, dim=-1, keepdim=True)
            x = x * torch.rsqrt(norm_x + self.eps)

            x = x.view(B, F, T, D)  # (b, f, t, d)
            x = x * self.scale[None, :, None, :]  # (1, f, 1, d)
            x = x.view(BF, T, D)
        else:
            # (b, d, t, f)
            norm_x = torch.mean(x**2, dim=1, keepdim=True)  # norm over d
            x = x * torch.rsqrt(norm_x + self.eps)

            x = x * self.scale.t()[None, :, None, :]  # scale (d, f) -> (1, d, 1, f)

        return x


class GatedRMSNorm(nn.Module):
    r"""Gated RMSNorm with silu gating.

    This is a variant of RMSNorm that applies a gated activation.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor, gate: Tensor | None = None) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)

        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


class BandGatedRMSNorm(nn.Module):
    r"""Band-aware Gated RMSNorm with per-band learnable scale.

    Similar to BandRMSNorm but with silu gating for use in GatedDeltaNet.
    Supports both (b*f, t, d) and (b*t, f, d) layouts by folding the band
    axis into the batch dimension.
    """

    def __init__(self, hidden_size: int, n_bands: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_bands = n_bands
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_bands, hidden_size))

    def forward(
        self,
        hidden_states: Tensor,
        gate: Tensor | None = None,
        layout: Literal["batch_band", "seq_band"] = "batch_band",
    ) -> Tensor:
        r"""Band-aware GatedRMSNorm.

        Args:
            hidden_states: (b*f, t, d) or (b*t, f, d)
            gate: optional gate tensor matching hidden_states
            layout: "batch_band" for (b*f, t, d), "seq_band" for (b*t, f, d)

        Returns:
            out: (b*f, t, d)
        """
        if hidden_states.ndim == 4:
            if hidden_states.shape[-1] != self.n_bands:
                raise ValueError(
                    f"Expected last dim {self.n_bands} for 4D input, got {hidden_states.shape}"
                )
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            hidden_states = hidden_states * self.weight.t()[None, :, None, :]
            if gate is not None:
                hidden_states = hidden_states * F.silu(gate.to(torch.float32))
            return hidden_states.to(input_dtype)

        if hidden_states.ndim != 3:
            raise ValueError(f"Expected 3D or 4D input, got {hidden_states.shape}")

        n_bands = self.n_bands
        leading = hidden_states.shape[0]
        if layout == "batch_band":
            if leading % n_bands != 0:
                raise ValueError(
                    f"Input leading dimension {leading} is not divisible by n_bands={n_bands}"
                )
            batch = leading // n_bands
            seq_len = hidden_states.shape[1]
            hidden_states = hidden_states.view(
                batch, n_bands, seq_len, self.hidden_size
            )
        elif layout == "seq_band":
            if hidden_states.shape[1] != n_bands:
                raise ValueError(
                    f"Expected band dimension {n_bands} in seq_band layout, got {hidden_states.shape}"
                )
        else:
            raise ValueError(f"Unknown layout: {layout}")

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if layout == "batch_band":
            hidden_states = hidden_states * self.weight[None, :, None, :]
            hidden_states = hidden_states.view(
                leading, hidden_states.shape[2], self.hidden_size
            )
        else:
            hidden_states = hidden_states * self.weight[None, :, :]
        hidden_states = hidden_states.to(input_dtype)

        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    r"""L2 normalization along specified dimension."""
    norm = x.norm(p=2, dim=dim, keepdim=True)
    return x / (norm + eps)


def torch_chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    chunk_size: int = 64,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Tensor | None]:
    r"""Pure PyTorch chunked gated delta rule (CPU fallback).

    b: batch_size
    t: seq_len
    n: num_heads
    h: head_dim

    Args:
        query: (b, t, n, h)
        key: (b, t, n, h)
        value: (b, t, n, h)
        g: (b, t, n) - gate values (log space)
        beta: (b, t, n) - delta scaling
        chunk_size: chunk size for chunked computation
        initial_state: Initial recurrent state
        output_final_state: Whether to return final state
        use_qk_l2norm_in_kernel: Whether to apply L2 norm to Q and K

    Returns:
        core_attn_out: (b, t, n, h)
        last_recurrent_state: Final recurrent state or None
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class GatedDeltaNet(nn.Module):
    r"""Gated Delta Net attention layer.

    This is a recurrent attention mechanism using gated delta rule.
    Uses bidirectional processing for non-causal attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int | None = None,
        head_dim: int = 64,
        conv_kernel_size: int = 4,
        activation: str = "silu",
        eps: float = 1e-6,
        n_bands: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_bands = n_bands

        # Default num_heads to hidden_size // head_dim if not specified
        if num_heads is None:
            num_heads = hidden_size // head_dim

        self.num_v_heads = num_heads
        self.num_k_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = conv_kernel_size
        self.activation = activation
        self.act = F.silu if activation == "silu" else F.gelu
        self.layer_norm_epsilon = eps

        # QKV projection dimension
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Conv1d for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Time step projection (discretization)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        # A parameter for delta rule
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # FusedRMSNormGated only works without band-aware norm
        if FusedRMSNormGated is not None and n_bands is None:
            self.fused_norm = FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
            )
        else:
            self.fused_norm = None

        self.fallback_norm = (
            BandGatedRMSNorm(self.head_v_dim, n_bands, eps)
            if n_bands is not None
            else GatedRMSNorm(self.head_v_dim, eps)
        )

        # Output projection
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # chunk_gated_delta_rule (fla/Triton) for CUDA; torch fallback for CPU
        self._chunk_gated_delta_rule = chunk_gated_delta_rule
        self._torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule

        # Input projections
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def _apply_output_norm(self, core_attn_out: Tensor, z: Tensor) -> Tensor:
        if self.fused_norm is not None and core_attn_out.is_cuda:
            core_attn_out = self.fused_norm(
                core_attn_out.reshape(-1, self.head_v_dim),
                z.reshape(-1, self.head_v_dim),
            )
            return core_attn_out.reshape(
                core_attn_out.shape[0] // (z.shape[1] * self.num_v_heads),
                z.shape[1],
                -1,
            )

        if isinstance(self.fallback_norm, BandGatedRMSNorm):
            if self.n_bands is None or core_attn_out.shape[0] % self.n_bands != 0:
                raise ValueError(
                    f"Cannot apply BandGatedRMSNorm with batch={core_attn_out.shape[0]} and n_bands={self.n_bands}"
                )
            batch = core_attn_out.shape[0] // self.n_bands
            core_attn_out = rearrange(
                core_attn_out,
                "(b f) t n h -> (b n f) t h",
                b=batch,
                f=self.n_bands,
                n=self.num_v_heads,
            )
            z = rearrange(
                z,
                "(b f) t n h -> (b n f) t h",
                b=batch,
                f=self.n_bands,
                n=self.num_v_heads,
            )
            core_attn_out = self.fallback_norm(
                core_attn_out, gate=z, layout="batch_band"
            )
            return rearrange(
                core_attn_out,
                "(b n f) t h -> (b f) t (n h)",
                b=batch,
                f=self.n_bands,
                n=self.num_v_heads,
            )

        core_attn_out = self.fallback_norm(
            core_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        return core_attn_out.reshape(
            core_attn_out.shape[0] // (z.shape[1] * self.num_v_heads), z.shape[1], -1
        )

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        r"""Forward pass for GatedDeltaNet.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            output: (batch, seq_len, hidden_size)
        """
        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        # Project to QKV
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        # Z projection for gating
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # Beta and alpha projections
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Convolution sequence transformation
        if self.conv1d is not None:
            mixed_qkv = self.act(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        # Reshape to (batch, seq_len, heads, head_dim)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # Select chunk delta rule: fla/Triton on CUDA, pure PyTorch on CPU
        chunk_fn = (
            self._chunk_gated_delta_rule
            if (self._chunk_gated_delta_rule is not None and hidden_states.is_cuda)
            else self._torch_chunk_gated_delta_rule
        )

        # Bidirectional: process forward and backward, then combine
        # Forward pass
        core_attn_out_fwd, _ = chunk_fn(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        # Backward pass: flip sequence
        query_rev = torch.flip(query, dims=[1])
        key_rev = torch.flip(key, dims=[1])
        value_rev = torch.flip(value, dims=[1])
        g_rev = torch.flip(g, dims=[1])
        beta_rev = torch.flip(beta, dims=[1])

        core_attn_out_bwd, _ = chunk_fn(
            query_rev,
            key_rev,
            value_rev,
            g=g_rev,
            beta=beta_rev,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        # Flip backward output back
        core_attn_out_bwd = torch.flip(core_attn_out_bwd, dims=[1])

        # Combine forward and backward (average)
        core_attn_out = (core_attn_out_fwd + core_attn_out_bwd) * 0.5

        core_attn_out = self._apply_output_norm(core_attn_out, z)

        # Output projection
        output = self.out_proj(core_attn_out)
        return output


class GatedAttention(nn.Module):
    r"""Multi-headed attention with gated output.

    This is a standard multi-head attention (no group query attention).
    Each head has its own Q, K, V projections.
    Automatically uses flash_attn if available.
    RoPE is applied externally via the rope module.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int | None = None,
        head_dim: int | None = None,
        attention_dropout: float = 0.0,
        bias: bool = False,
        qk_norm_n_bands: int | None = None,
        qk_norm_layout: Literal["batch_band", "seq_band"] = "batch_band",
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_norm_n_bands = qk_norm_n_bands
        self.qk_norm_layout = qk_norm_layout

        # Default num_heads to hidden_size // head_dim if not specified
        if num_heads is None:
            num_heads = hidden_size // (head_dim or 64)
        if head_dim is None:
            head_dim = hidden_size // num_heads

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout

        # Q, K, V projections (no group query - same num_heads for all)
        self.q_proj = nn.Linear(
            hidden_size, num_heads * head_dim * 2, bias=bias
        )  # *2 for gate
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

        self.q_norm = (
            BandGatedRMSNorm(head_dim, qk_norm_n_bands, eps=1e-6)
            if qk_norm_n_bands is not None
            else RMSNorm(head_dim, eps=1e-6)
        )
        self.k_norm = (
            BandGatedRMSNorm(head_dim, qk_norm_n_bands, eps=1e-6)
            if qk_norm_n_bands is not None
            else RMSNorm(head_dim, eps=1e-6)
        )

    def _apply_qk_norm(self, norm: nn.Module, x: Tensor, layout: str) -> Tensor:
        if isinstance(norm, BandGatedRMSNorm):
            b, l, n, h = x.shape
            x = rearrange(x, "b l n h -> (b n) l h")
            x = norm(x, layout=layout)
            return rearrange(x, "(b n) l h -> b l n h", b=b, n=n)
        return norm(x)

    def forward(
        self,
        x: Tensor,
        rope: nn.Module | None = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        r"""Forward pass for GatedAttention.

        Args:
            x: (b, l, d) - Input tensor
            rope: RoPE module for position embeddings
            pos: (l, 2) or (b, l, 2) position indices for 2D RoPE

        Returns:
            x: (b, l, d) - Output tensor
        """
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q_gate = self.q_proj(x).view(*input_shape, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(q_gate, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        query_states = self._apply_qk_norm(
            self.q_norm, query_states.view(hidden_shape), self.qk_norm_layout
        )
        key_states = self._apply_qk_norm(
            self.k_norm, self.k_proj(x).view(hidden_shape), self.qk_norm_layout
        )
        value_states = self.v_proj(x).view(hidden_shape)

        if rope is not None:
            if pos is not None:
                query_states = rope.apply_nd(query_states, pos)
                key_states = rope.apply_nd(key_states, pos)
            else:
                query_states = rope(query_states)
                key_states = rope(key_states)

        # Compute attention using flash_attn if available, else PyTorch SDPA
        # FlashAttention only supports fp16 and bf16 on CUDA
        is_half = query_states.dtype in (torch.float16, torch.bfloat16)
        is_autocast = torch.is_autocast_enabled()
        use_flash_attn = (
            FLASH_ATTN_AVAILABLE and query_states.is_cuda and (is_half or is_autocast)
        )

        if use_flash_attn:
            # Cast to autocast dtype if needed (RMSNorm may have upcasted to fp32)
            if not is_half and is_autocast:
                target_dtype = torch.get_autocast_gpu_dtype()
                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)
            # flash_attn expects (b, l, n, h) layout
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )  # (b, l, n, h)
        else:
            # PyTorch SDPA
            query_states = rearrange(query_states, "b l n h -> b n l h")
            key_states = rearrange(key_states, "b l n h -> b n l h")
            value_states = rearrange(value_states, "b l n h -> b n l h")

            attn_output = F.scaled_dot_product_attention(
                query=query_states,
                key=key_states,
                value=value_states,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.scaling,
            )  # (b, n, l, h)
            attn_output = rearrange(attn_output, "b n l h -> b l n h")

        # Apply gating - keep shapes matching: attn_output (b, l, n, h), gate (b, l, n, h)
        gate = gate.view(*input_shape, self.num_heads, self.head_dim)  # (b, l, n, h)
        attn_output = attn_output * torch.sigmoid(gate)

        # Reshape and project
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # (b, l, n*h)
        attn_output = self.o_proj(attn_output)

        return attn_output


class GatedMLP(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()

        intermediate_dim = intermediate_dim * 2 // 3

        self.gate_proj = nn.Linear(dim, intermediate_dim)
        self.up_proj = nn.Linear(dim, intermediate_dim)
        self.down_proj = nn.Linear(intermediate_dim, dim)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor) -> Tensor:
        r"""Gated MLP.

        Args:
            x: (b, l, d)

        Outputs:
            out: (b, l, d)
        """

        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ============== BSRoformer56b Modules ==============


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

    def __init__(self, dim, dim_mult=2, n_bands: int | None = None):
        super().__init__()

        # 1. Normalization (Pre-norm) - separate norms for decoder and encoder
        # Use BandRMSNorm when n_bands is provided, otherwise use standard RMSNorm2d
        if n_bands is not None:
            self.norm_dec = BandGatedRMSNorm(dim, n_bands)
            self.norm_enc = BandGatedRMSNorm(dim, n_bands)
        else:
            self.norm_dec = RMSNorm2d(dim)
            self.norm_enc = RMSNorm2d(dim)

        # 2. Projection
        # We take 2 * dim inputs (enc + dec)
        # We project to 2 * (dim * dim_mult) because GEGLU requires splitting
        hidden_dim = int(dim * dim_mult)
        self.proj_in = nn.Conv2d(dim * 2, hidden_dim * 2, kernel_size=1)

        # 3. Output projection
        self.proj_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x_dec, x_enc):
        """
        Args:
            x_dec: Decoder features (B, C, H, W) -> Acts as the residual backbone
            x_enc: Encoder features (B, C, H', W')
        """
        # 1. Spatial Alignment (Crucial for U-Nets)
        if x_enc.shape[-2:] != x_dec.shape[-2:]:
            x_enc = F.interpolate(
                x_enc, size=x_dec.shape[-2:], mode="bilinear", align_corners=False
            )

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


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class BandGRN(nn.Module):
    r"""Band-aware Global Response Normalization with per-band learnable scale.

    Similar to GRN but learns separate gamma and beta per frequency band,
    allowing each band to adapt its normalization independently.

    When used in time attention blocks, the input has shape (b*f, t, d)
    where the band dimension f is folded into the batch.
    """

    def __init__(self, dim: int, n_bands: int):
        super().__init__()
        self.dim = dim
        self.n_bands = n_bands
        self.gamma = nn.Parameter(torch.zeros(n_bands, dim))  # (f, d)
        self.beta = nn.Parameter(torch.zeros(n_bands, dim))  # (f, d)

    def forward(self, x: Tensor) -> Tensor:
        r"""Band-aware GRN.

        Args:
            x: (b*f, t, d) - bands folded into batch

        Returns:
            out: (b*f, t, d)
        """
        BF, T, D = x.shape
        F = self.n_bands
        B = BF // F

        # Compute global response: Gx = ||x||_2 along channel dim
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)  # (b*f, t, 1)

        # Normalize: Nx = Gx / mean(Gx) across time dimension
        Gx = Gx.view(B, F, T, 1)  # (b, f, t, 1)
        Nx = Gx / (Gx.mean(dim=2, keepdim=True) + 1e-6)  # (b, f, t, 1)

        # Apply per-band gamma and beta
        gamma = self.gamma[None, :, None, :]  # (1, f, 1, d)
        beta = self.beta[None, :, None, :]  # (1, f, 1, d)

        x = x.view(B, F, T, D)  # (b, f, t, d)
        out = gamma * (x * Nx) + beta + x  # (b, f, t, d)
        out = out.view(BF, T, D)  # (b*f, t, d)

        return out


class TimeMixBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, dim_mult=4, n_bands: int | None = None):
        super().__init__()
        hidden_dim = int(dim * dim_mult * 2 // 3)
        padding = kernel_size // 2
        self.pwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )

        # Use BandRMSNorm when n_bands is provided, otherwise use standard RMSNorm
        if n_bands is not None:
            self.norm = BandGatedRMSNorm(dim, n_bands)
        else:
            self.norm = RMSNorm(dim)

        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.act = nn.GELU()

        # Use BandGRN when n_bands is provided, otherwise use standard GRN
        if n_bands is not None:
            self.grn = BandGRN(hidden_dim, n_bands)
        else:
            self.grn = GRN(hidden_dim)

        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        h = self.pwconv(x.transpose(1, 2)).transpose(1, 2)
        h = (
            self.norm(h, layout="batch_band")
            if isinstance(self.norm, BandGatedRMSNorm)
            else self.norm(h)
        )
        gate, value = self.proj(h).chunk(2, dim=-1)
        gate = self.act(gate)
        gate = self.grn(gate)
        h = gate * value
        h = self.proj_out(h)
        return x + h


class BSRoformerBlock(nn.Module):
    """Band-Split RoFormer block with time and/or frequency attention.

    Uses either DeltaNetBlock or AttentionBlock for time processing.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        axis: str = "tf",
        t_block_type: str = "delta",
        head_dim: int = 64,
        n_bands: int | None = None,
        conv_kernel_size: int = 7,
    ):
        super().__init__()
        assert axis in ["t", "f", "tf", "ft"], (
            "Axis must be one of 't', 'f', 'tf', or 'ft'."
        )
        self.axis = axis
        self.n_bands = n_bands
        self.use_time_rope = "t" in axis
        self.use_freq_rope = "f" in axis

        time_qk_norm_bands = n_bands if "t" in axis else None
        freq_qk_norm_bands = n_bands if "f" in axis else None

        if "t" in axis:
            if t_block_type == "delta":
                self.time_block = DeltaNetBlock(
                    dim=dim,
                    num_heads=n_heads,
                    head_dim=head_dim,
                    conv_kernel_size=conv_kernel_size,
                    n_bands=n_bands,
                )
            elif t_block_type == "time_mix":
                self.time_block = TimeMixBlock(dim=dim, n_bands=n_bands)
            elif t_block_type == "attn":
                self.time_block = AttentionBlock(
                    dim=dim,
                    num_heads=n_heads,
                    head_dim=head_dim,
                    n_bands=n_bands,
                    qk_norm_n_bands=time_qk_norm_bands,
                    norm_layout="batch_band",
                    qk_norm_layout="batch_band",
                )
            else:
                raise ValueError(f"Unknown time block type: {t_block_type}")
        else:
            self.time_block = None

        if "f" in axis:
            self.freq_block = AttentionBlock(
                dim=dim,
                num_heads=n_heads,
                head_dim=head_dim,
                n_bands=n_bands,
                qk_norm_n_bands=freq_qk_norm_bands,
                norm_layout="seq_band",
                qk_norm_layout="seq_band",
            )
        else:
            self.freq_block = None

    def forward(
        self,
        x: Tensor,
        time_rope: nn.Module | None = None,
        freq_rope: nn.Module | None = None,
        time_pos: LongTensor | None = None,
    ) -> Tensor:
        """
        Apply time and/or frequency attention.

        Args:
            x: Input tensor of shape (b, d, t, f).
            time_rope: 2D rotary position embedding for time attention.
            freq_rope: 1D rotary position embedding for frequency attention.
            time_pos: (b*f, t, 2) time-axis positions with frequency index retained.

        Returns:
            Output tensor of shape (b, d, t, f).
        """
        B = x.shape[0]

        if self.time_block is not None:
            x = rearrange(x, "b d t f -> (b f) t d")
            if isinstance(self.time_block, AttentionBlock):
                x = self.time_block(
                    x,
                    rope=time_rope if self.use_time_rope else None,
                    pos=time_pos,
                )
            else:
                x = self.time_block(x)
            x = rearrange(x, "(b f) t d -> b d t f", b=B)

        if self.freq_block is not None:
            x = rearrange(x, "b d t f -> (b t) f d")
            x = self.freq_block(x, rope=freq_rope if self.use_freq_rope else None)
            x = rearrange(x, "(b t) f d -> b d t f", b=B)

        return x


class BSRoformer(Fourier):
    """Band-Split RoFormer for music source separation."""

    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_fft=2048,
        hop_length=480,
        n_bands=256,
        dim_sp=96,
        dim=384,
        dim_head=32,
        patch_size=[4, 4],
        n_layers=12,
        n_pre_layers=1,
        n_post_layers=1,
        pre_time_block: bool = False,
        post_time_block: bool = False,
        use_checkpoint: bool = False,
        delta_attn_ratio=3,
        rope_len=8192,
        conv_kernel_size=7,
        **kwargs,
    ) -> None:
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, return_complex=True, normalized=True
        )

        self.ac = audio_channels
        self.patch_size = patch_size

        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate,
            n_fft=n_fft,
            n_bands=n_bands,
            in_channels=2,  # real + imag
            out_channels=dim_sp // 2,
        )

        # RoPE - always use 2D position encoding
        self.time_rope = MultidimRoPE(head_dim=dim_head, n_dims=2, max_len=rope_len)
        self.freq_rope = RoPE1D(head_dim=dim_head, max_len=rope_len)
        self.pre_time_block = pre_time_block
        self.post_time_block = post_time_block
        self.use_checkpoint = use_checkpoint

        # Blocks
        self.down = nn.Conv2d(dim_sp, dim, kernel_size=patch_size, stride=patch_size)
        self.up = nn.ConvTranspose2d(
            dim, dim_sp, kernel_size=patch_size, stride=patch_size
        )

        # Band counts at each resolution for band-aware norm in time blocks
        n_bands_full = n_bands  # pre/post blocks operate at full band resolution
        n_bands_down = n_bands // patch_size[1]  # main blocks after downsampling

        self.fusion = GEGLUFusion(dim_sp, n_bands=n_bands_full)

        self.pre_blocks = nn.ModuleList(
            [
                BSRoformerBlock(
                    dim_sp,
                    dim_sp // dim_head,
                    axis="tf" if pre_time_block else "f",
                    t_block_type="time_mix" if pre_time_block else "attn",
                    head_dim=dim_head,
                    n_bands=n_bands_full,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(n_pre_layers)
            ]
        )
        self.post_blocks = nn.ModuleList(
            [
                BSRoformerBlock(
                    dim_sp,
                    dim_sp // dim_head,
                    axis="tf" if post_time_block else "f",
                    t_block_type="time_mix" if post_time_block else "attn",
                    head_dim=dim_head,
                    n_bands=n_bands_full,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(n_post_layers)
            ]
        )
        # Main blocks: 3 DeltaNet + 1 Attention pattern for every 4 blocks
        self.blocks = nn.ModuleList([])
        for i in range(n_layers):
            use_deltanet = (
                i % delta_attn_ratio + 1
            ) != delta_attn_ratio  # Use DeltaNet for 0,1,2; Attention for 3
            self.blocks.append(
                BSRoformerBlock(
                    dim,
                    dim // dim_head,
                    axis="tf",
                    t_block_type="delta" if use_deltanet else "attn",
                    head_dim=dim_head,
                    n_bands=n_bands_down,
                    conv_kernel_size=conv_kernel_size,
                )
            )

        self.final_norm = BandGatedRMSNorm(dim_sp, n_bands_full)
        self._time_rope_position_cache: dict[
            tuple[int, int], tuple[LongTensor, LongTensor]
        ] = {}
        self._freq_rope_position_cache: dict[int, LongTensor] = {}

    def _maybe_checkpoint_block(
        self,
        block: nn.Module,
        x: Tensor,
        *,
        time_rope: nn.Module,
        freq_rope: nn.Module,
        time_pos: LongTensor,
    ) -> Tensor:
        if not self.training or not self.use_checkpoint:
            return block(x, time_rope=time_rope, freq_rope=freq_rope, time_pos=time_pos)

        return checkpoint(
            lambda inp, tr, fr, tp: block(
                inp,
                time_rope=tr,
                freq_rope=fr,
                time_pos=tp,
            ),
            x,
            time_rope,
            freq_rope,
            time_pos,
            use_reentrant=False,
        )

    def _get_time_axis_positions(
        self, batch_size: int, time_steps: int, freq_bins: int, device: torch.device
    ) -> tuple[LongTensor, LongTensor]:
        cache_key = (time_steps, freq_bins)
        if cache_key not in self._time_rope_position_cache:
            self._time_rope_position_cache[cache_key] = (
                self.time_rope.build_grid_positions(
                    time_steps=time_steps,
                    freq_bins=freq_bins,
                    device=torch.device("cpu"),
                )
            )

        cached_time_pos, cached_freq_pos = self._time_rope_position_cache[cache_key]
        time_pos = cached_time_pos.to(device=device)
        freq_pos = cached_freq_pos.to(device=device)

        time_pos = time_pos.view(freq_bins, time_steps, 2)
        time_pos = time_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)
        time_pos = time_pos.reshape(batch_size * freq_bins, time_steps, 2)

        freq_pos = freq_pos.view(time_steps, freq_bins, 2)
        freq_pos = freq_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)
        freq_pos = freq_pos.reshape(batch_size * time_steps, freq_bins, 2)

        return time_pos, freq_pos

    def forward(self, audio: Tensor) -> Tensor:
        """
        Separate audio sources.

        Args:
            audio: Input audio of shape (b, c, l).

        Returns:
            Separated audio of shape (b, c, l).
        """
        complex_sp = self.stft(audio)
        T0 = complex_sp.shape[2]

        x = torch.view_as_real(complex_sp)
        x = self.pad_tensor(x)
        x = self.bandsplit.transform(x)

        x = rearrange(x, "b c t f i -> b (c i) t f")
        B = x.shape[0]

        T, F = x.shape[2], x.shape[3]
        time_pos_full, _ = self._get_time_axis_positions(B, T, F, x.device)

        for block in self.pre_blocks:
            x = self._maybe_checkpoint_block(
                block,
                x,
                time_rope=self.time_rope,
                freq_rope=self.freq_rope,
                time_pos=time_pos_full,
            )
        h = x
        x = self.down(x)

        T_down, F_down = x.shape[2], x.shape[3]
        time_pos_down, _ = self._get_time_axis_positions(B, T_down, F_down, x.device)

        for block in self.blocks:
            x = block(
                x,
                time_rope=self.time_rope,
                freq_rope=self.freq_rope,
                time_pos=time_pos_down,
            )

        x = self.up(x)
        x = self.fusion(x, h)
        for block in self.post_blocks:
            x = self._maybe_checkpoint_block(
                block,
                x,
                time_rope=self.time_rope,
                freq_rope=self.freq_rope,
                time_pos=time_pos_full,
            )

        x = self.final_norm(x)
        x = rearrange(x, "b (c i) t f -> b c t f i", c=self.ac)
        x = self.bandsplit.inverse_transform(x)

        x = x[:, :, 0:T0, :, :].contiguous()
        mask = torch.view_as_complex(x)
        sep_stft = mask * complex_sp
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
    model = BSRoformer(use_checkpoint=True)
    dummy_audio = torch.randn(1, 2, 48000 * 2)
    output = model(dummy_audio)
    print(output.shape)
