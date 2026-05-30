from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, Tensor
import math

import librosa
import numpy as np
from torch.nn.utils.rnn import pad_sequence


# ============== RoPE ==============


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 8192, base: int = 10000):
        r"""Rotary position embedding.

        [1] Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary
        position embedding." Neurocomputing, 2024

        h: head_dim
        l: seq_len
        """
        super().__init__()

        self.head_dim = head_dim

        # Calculate θ = 1 / 10000**(2i/h)
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))  # (h/2,)

        # Matrix pθ
        pos_theta = torch.outer(torch.arange(max_len), theta).float()  # (l, h/2)

        # Rotation matrix
        w = torch.stack(
            [torch.cos(pos_theta), torch.sin(pos_theta)], dim=-1
        )  # (l, h/2, 2)
        self.register_buffer(name="w", tensor=w)

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply RoPE.

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim

        Args:
            x: (b, l, n, h)

        Outputs:
            out: (b, l, n, h)
        """

        L = x.shape[1]
        x = rearrange(x, "b l n (h c) -> b l n h c", c=2)  # (b, l, n, h/2, 2)
        w = self.w[0:L][None, :, None, :, :]  # (1, l, 1, h/2, 2)
        x = self.rotate(x, w)  # (b, l, n, h/2, 2)
        x = rearrange(x, "b l n h c -> b l n (h c)")  # (b, l, n, h)

        return x

    def rotate(self, x: Tensor, w: Tensor) -> Tensor:
        r"""Rotate x.

        x0 = cos(θp)·x0 - sin(θp)·x1
        x1 = sin(θp)·x0 + cos(θp)·x1

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim

        Args:
            x: (b, l, n, h/2, 2)
            w: (1, l, 1, h/2, 2)

        Outputs:
            out: (b, l, n, h/2, 2)
        """

        out = torch.stack(
            [
                w[..., 0] * x[..., 0] - w[..., 1] * x[..., 1],
                w[..., 0] * x[..., 1] + w[..., 1] * x[..., 0],
            ],
            dim=-1,
        )  # (b, l, n, h/2, 2)

        return out

    def apply_nd(self, x: Tensor, pos: LongTensor) -> Tensor:
        r"""Apply Nd RoPE with sparse positions.

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim
        k: data dim

        Args:
            x: (b, l, n, h)
            pos: (l, k)
            n_dim: int

        Outputs:
            out: (b, l, n, h)
        """

        B, L, N, H = x.shape
        K = pos.shape[1]  # rope_dim
        assert H == K * self.head_dim

        x = rearrange(
            x, "b l n (k h c) -> k b l n h c", k=K, c=2
        )  # (k, b, l, n, h/k/2, 2)
        out = torch.zeros_like(x, device=x.device)

        for i in range(K):
            p = pos[:, i]  # (l,)
            w = self.w[p][None, :, None, :, :]  # (1, l, 1, h/k/2, 2)
            out[i] = self.rotate(x[i], w)

        return rearrange(x, "k b l n h c -> b l n (k h c)")  # (b, l, n, h)


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
        self, sr: float, n_fft: int, n_bands: int, in_channels: int, out_channels: int
    ) -> None:
        r"""Band split STFT to mel scale STFT.

        f: stft bins
        k: mel bins
        w: band width
        """

        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.n_bands = n_bands
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Init mel banks
        melbanks, ola_window = self.init_melbanks()
        # melbanks = np.pad(melbanks, pad_width=(0, 1))
        self.register_buffer(name="melbanks", tensor=Tensor(melbanks))  # (k, f)
        self.register_buffer(name="ola_window", tensor=Tensor(ola_window))  # (f,)

        self.pre_w = nn.Parameter(
            torch.zeros((in_channels, n_fft // 2 + 1, out_channels))
        )  # (i, f, o)
        self.post_w = nn.Parameter(
            torch.zeros((out_channels, in_channels, n_fft // 2 + 1))
        )  # (o, i, f)
        self.pre_b = nn.Parameter(torch.zeros((self.n_bands, out_channels)))  # (k, o)
        self.post_b = nn.Parameter(torch.zeros((in_channels, n_fft // 2 + 1)))  # (i, f)

        # bound = 1 / math.sqrt(in_channels * (n_fft // 2 + 1))
        gain = np.sqrt(
            np.linalg.pinv(self.in_channels * melbanks**2) @ np.ones(self.n_bands)
        )  # (f)
        gain = torch.from_numpy(gain[None, :, None])

        nn.init.uniform_(self.pre_w, -1, 1)
        with torch.no_grad():
            self.pre_w *= gain

        bound = 1 / math.sqrt(out_channels)
        nn.init.uniform_(self.post_w, -bound, bound)

        nonzero_indexes = []  # (k, w)
        nonzero_melbanks = []  # (k, w)

        for f in range(self.n_bands):
            idxes = torch.nonzero(self.melbanks[f].abs() > 1e-6, as_tuple=True)[
                0
            ]  # shape: (w,)
            nonzero_indexes.append(idxes)
            nonzero_melbanks.append(self.melbanks[f, idxes])

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

        #
        for q in range(Q):
            sb_idxes = []
            sb_melbanks = []
            for f in subbands[q]:
                sb_idxes.append(nonzero_indexes[f])
                sb_melbanks.append(nonzero_melbanks[f])

            sb_idxes = pad_sequence(
                sequences=sb_idxes, batch_first=True, padding_value=-1
            )  # (k, w)
            sb_masks = (sb_idxes != -1).float()
            sb_idxes[sb_idxes == -1] = n_fft // 2
            sb_melbanks = pad_sequence(
                sequences=sb_melbanks, batch_first=True, padding_value=0
            )  # (k, w)

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
            x: (b, c, t, f)

        Returns:
            x: (b, d, t, k)
        """

        B, C, T, F_, I = x.shape
        O_ = self.out_channels
        out = torch.zeros(
            (B, C, T, self.n_bands, O_), device=x.device
        )  # (b, c, t, f, o)

        for q in range(self.Q):
            subbands = getattr(self, f"sb_subbands_{q}")  # (s,)
            idxes = getattr(self, f"sb_idxes_{q}")  # (s, w)
            melbanks = getattr(self, f"sb_melbanks_{q}")  # (s, w)
            masks = getattr(self, f"sb_masks_{q}")  # (s, w)

            _x = (
                x[..., idxes, :] * melbanks[..., :, :, None] * masks[..., :, :, None]
            )  # (b, c, t, s, w, i)
            _w = self.pre_w[:, idxes, :] * masks[None, :, :, None]  # (i, s, w, o)
            _b = self.pre_b[subbands, :]  # (s, o)
            _y = torch.einsum("bctswi,iswo->bctso", _x, _w) + _b  # (b, c, t, s, o)
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
            x: (b, c, t, f, o)

        Outputs:
            y: (b, c, t, f)
        """

        B, C, T, F, O_ = x.shape
        I = self.in_channels
        out = torch.zeros((B, C, T, self.n_fft // 2 + 1, I), device=x.device)

        for q in range(self.Q):
            subbands = getattr(self, f"sb_subbands_{q}")  # (s,)
            idxes = getattr(self, f"sb_idxes_{q}")  # (s, w)
            melbanks = getattr(self, f"sb_melbanks_{q}")  # (s, w)
            masks = getattr(self, f"sb_masks_{q}")  # (s, w)

            _x = x[:, :, :, subbands, :]  # (b, c, t, s, o)
            _w = self.post_w[:, :, idxes] * masks  # (o, i, s, w)
            _b = self.post_b[:, idxes] * masks  # (i, s, w)
            _b = rearrange(_b, "i s w -> s w i")
            _y = torch.einsum("bctso,oisw->bctswi", _x, _w) + _b  # (b, c, t, s, w, i)
            _y = _y * melbanks[..., :, :, None] * masks[..., :, :, None]
            out.scatter_add_(
                dim=3,
                index=idxes.flatten()[None, None, None, :, None].repeat(B, C, T, 1, I),
                src=_y.flatten(3, 4),
            )

        # Divide overlap add window
        out /= self.ola_window[..., :, None]

        return out


# ============== Attention ==============

# Optional flash_attn import
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class AbsolutePositionEmbedding(nn.Module):
    r"""Absolute Position Embedding.

    This module implements learnable absolute position embeddings that are added
    to the query and key projections in the attention mechanism. Unlike RoPE which
    applies rotations, absolute embeddings directly add positional information.

    Absolute position embeddings are simpler and can be more stable for certain tasks,
    especially when:
    - Sequence lengths are relatively fixed during training and inference
    - The model needs to learn specific positional patterns
    - Computational efficiency is critical (no rotation operations needed)

    Args:
        max_seq_len: Maximum sequence length to support
        dim: Embedding dimension (should match the model's hidden dimension)

    Shape:
        - Input: (batch_size, seq_len, num_heads, head_dim)
        - Output: (batch_size, seq_len, num_heads, head_dim)
    """

    def __init__(self, max_seq_len: int, dim: int) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.dim = dim

        # Learnable position embeddings
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, dim) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply absolute position embeddings.

        Args:
            x: (b, l, n, h) - Input tensor

        Returns:
            out: (b, l, n, h) - Tensor with position embeddings added
        """
        B, L, N, H = x.shape

        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds maximum supported length {self.max_seq_len}. "
                f"Please increase max_seq_len in the configuration."
            )

        # Get position embeddings for current sequence length
        pos_emb = self.pos_emb[:L]  # (l, d)

        # Reshape to match input dimensions: (l, d) -> (1, l, 1, d)
        pos_emb = pos_emb.view(1, L, 1, -1)

        # Add position embeddings to input
        out = x + pos_emb

        return out

    def apply_nd(self, x: Tensor, pos: LongTensor) -> Tensor:
        r"""Apply N-dimensional absolute position embeddings with sparse positions.

        This method supports multi-dimensional positional encoding (e.g., for images or videos)
        by indexing into the position embedding table using provided position indices.

        Args:
            x: (b, l, n, h) - Input tensor
            pos: (l, k) - Position indices for each dimension

        Returns:
            out: (b, l, n, h) - Tensor with position embeddings added
        """
        B, L, N, H = x.shape
        K = pos.shape[1]  # Number of dimensions

        # For N-D positions, we sum embeddings from each dimension
        # Each dimension gets H // K features
        dim_per_axis = self.dim // K

        pos_emb_sum = torch.zeros(L, self.dim, device=x.device, dtype=x.dtype)

        for i in range(K):
            p = pos[:, i]  # (l,)

            # Check bounds
            if torch.any(p >= self.max_seq_len):
                raise ValueError(
                    f"Position index {p.max().item()} exceeds maximum supported length {self.max_seq_len}"
                )

            # Get embeddings for this dimension
            start_idx = i * dim_per_axis
            end_idx = start_idx + dim_per_axis if i < K - 1 else self.dim
            pos_emb_sum[:, start_idx:end_idx] = self.pos_emb[p, start_idx:end_idx]

        # Reshape and add to input
        pos_emb_sum = pos_emb_sum.view(1, L, 1, -1)
        out = x + pos_emb_sum

        return out


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


class SelfAttention(nn.Module):
    r"""Self-Attention module with configurable position embedding and attention backend.

    This module supports two types of position embeddings:

    1. RoPE (Rotary Position Embedding) - Default
       - Applies rotations to query and key vectors
       - Better extrapolation to longer sequences
       - No learnable parameters for position encoding
       - Recommended for variable-length sequences

    2. Absolute Position Embedding
       - Adds learnable position embeddings to query and key
       - Simpler and more stable for fixed-length sequences
       - Requires specifying max_seq_len
       - Can be more efficient computationally

    This module also supports two attention backends:

    1. torch - Uses PyTorch's scaled_dot_product_attention
       - Supports windowed attention via attention mask

    2. flash_attn - Uses flash_attn library for efficient attention
       - Supports windowed attention via window_size argument

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        position_embedding_type: Type of position embedding ("rope" or "absolute")
        max_seq_len: Maximum sequence length (required for absolute embeddings)
        attn_backend: Attention backend ("torch" or "flash_attn")
        window_size: Window size for local attention (None for global attention)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        position_embedding_type: str = "rope",
        max_seq_len: int = 8192,
        attn_backend: Literal["torch", "flash_attn"] = "torch",
        window_size: int | None = None,
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.dim = dim
        self.window_size = window_size

        # Validate attention backend
        valid_backends = ["torch", "flash_attn"]
        if attn_backend not in valid_backends:
            raise ValueError(
                f"Invalid attn_backend: '{attn_backend}'. "
                f"Must be one of {valid_backends}"
            )

        # Check flash_attn availability
        if attn_backend == "flash_attn" and not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "flash_attn backend requested but flash_attn is not installed. "
                "Please install it with: pip install flash-attn --no-build-isolation"
            )

        self.attn_backend = attn_backend

        # Validate position embedding type
        valid_types = ["rope", "absolute"]
        if position_embedding_type not in valid_types:
            raise ValueError(
                f"Invalid position_embedding_type: '{position_embedding_type}'. "
                f"Must be one of {valid_types}"
            )

        self.position_embedding_type = position_embedding_type

        # Initialize position embedding module if using absolute embeddings
        if position_embedding_type == "absolute":
            self.pos_embedding = AbsolutePositionEmbedding(
                max_seq_len=max_seq_len,
                dim=self.head_dim,
            )
        else:
            self.pos_embedding = None

        self.qkv_linear = nn.Linear(dim, 3 * dim)
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)

        self.proj = nn.Linear(dim, dim)

    def _create_window_mask(self, seq_len: int, device: torch.device) -> Tensor:
        r"""Create attention mask for windowed attention.

        Args:
            seq_len: Sequence length
            device: Device to create the mask on

        Returns:
            mask: (seq_len, seq_len) - Boolean mask where True indicates positions to attend to
        """
        if self.window_size is None:
            return None

        # Create a mask where each position can attend to positions within the window
        # Window is centered on each position
        positions = torch.arange(seq_len, device=device)

        # Calculate distance matrix
        distance = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))

        # Create mask: True for positions within window, False otherwise
        mask = distance <= self.window_size // 2

        return mask

    def forward(
        self,
        x: Tensor,
        rope: nn.Module | None = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        r"""Self attention with configurable position embedding and backend.

        b: batch_size
        l: seq_len
        d: latent_dim
        n: n_head
        h: head_dim

        Args:
            x: (b, l, d) - Input tensor
            rope: RoPE module - Only used if position_embedding_type="rope"
            pos: (l, k) - Position indices for N-D embeddings (optional)

        Outputs:
            x: (b, l, d) - Output tensor
        """

        # Calculate query, key, values
        q, k, v = self.qkv_linear(x).chunk(chunks=3, dim=2)  # shapes: (b, l, d)
        q = rearrange(
            self.norm_q(q), "b l (n h) -> b l n h", h=self.head_dim
        )  # (b, l, n, h)
        k = rearrange(
            self.norm_k(k), "b l (n h) -> b l n h", h=self.head_dim
        )  # (b, l, n, h)
        v = rearrange(v, "b l (n h) -> b l n h", h=self.head_dim)  # (b, l, n, h)

        # Apply position embeddings based on type
        if self.position_embedding_type == "rope":
            # Use RoPE (Rotary Position Embedding)
            if rope is None:
                raise ValueError(
                    "RoPE module must be provided when position_embedding_type='rope'"
                )

            if pos is None:
                q = rope(q)  # (b, l, n, h)
                k = rope(k)  # (b, l, n, h)
            else:
                q = rope.apply_nd(q, pos)  # (b, l, n, h)
                k = rope.apply_nd(k, pos)  # (b, l, n, h)

        elif self.position_embedding_type == "absolute":
            # Use Absolute Position Embedding
            if pos is None:
                q = self.pos_embedding(q)  # (b, l, n, h)
                k = self.pos_embedding(k)  # (b, l, n, h)
            else:
                q = self.pos_embedding.apply_nd(q, pos)  # (b, l, n, h)
                k = self.pos_embedding.apply_nd(k, pos)  # (b, l, n, h)

        # Compute attention based on backend
        if self.attn_backend == "flash_attn":
            # Use flash_attn for efficient attention
            # flash_attn_func expects (b, l, n, h) layout and returns (b, l, n, h)
            if self.window_size is not None:
                # flash_attn uses (window_left, window_right) for sliding window attention
                # We use symmetric window: (window_size // 2, window_size // 2)
                window_size_half = self.window_size // 2
                window_size_tuple = (window_size_half, window_size_half)
            else:
                window_size_tuple = (-1, -1)  # (-1, -1) means global attention

            x = flash_attn_func(
                q,
                k,
                v,
                window_size=window_size_tuple,
            )  # (b, l, n, h)

        else:
            # Use PyTorch's scaled_dot_product_attention
            # Create attention mask for windowed attention if needed
            attn_mask = None
            if self.window_size is not None:
                attn_mask = self._create_window_mask(x.shape[1], x.device)
                # scaled_dot_product_attention expects mask where True = masked out
                # So we need to invert our mask
                attn_mask = ~attn_mask

            x = F.scaled_dot_product_attention(
                query=rearrange(q, "b l n h -> b n l h"),
                key=rearrange(k, "b l n h -> b n l h"),
                value=rearrange(v, "b l n h -> b n l h"),
                attn_mask=attn_mask,
                dropout_p=0.0,
            )  # (b, n, l, h)
            x = rearrange(x, "b n l h -> b l n h")

        x = rearrange(x, "b l n h -> b l (n h)")
        x = self.proj(x)  # (b, l, d)

        return x


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


class Block(nn.Module):
    r"""Self attention block.

    Ref:
        [1] https://github.com/facebookresearch/DiT/blob/main/models.py
        [2] https://huggingface.co/hpcai-tech/OpenSora-STDiT-v1-HQ-16x256x256/blob/main/layers.py
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        position_embedding_type: str = "rope",
        max_seq_len: int = 8192,
        attn_backend: Literal["torch", "flash_attn"] = "torch",
        window_size: int | None = None,
    ) -> None:
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.attn = SelfAttention(
            dim,
            num_heads,
            position_embedding_type=position_embedding_type,
            max_seq_len=max_seq_len,
            attn_backend=attn_backend,
            window_size=window_size,
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        x: Tensor,
        rope: RoPE | None = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            rope: (t, head_dim/2, 2) - Only used if position_embedding_type="rope"
            pos: (l, k) - Position indices for N-D embeddings (optional)

        Outputs:
            out: (b, l, d)
        """

        x = x + self.attn(self.norm1(x), rope, pos)
        x = x + self.ffn(self.norm2(x))

        return x


class BlockV2(nn.Module):
    """Self attention block with SwiGLU"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        position_embedding_type: str = "rope",
        max_seq_len: int = 8192,
        attn_backend: Literal["torch", "flash_attn"] = "torch",
        window_size: int | None = None,
    ) -> None:
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.attn = SelfAttention(
            dim,
            num_heads,
            position_embedding_type=position_embedding_type,
            max_seq_len=max_seq_len,
            attn_backend=attn_backend,
            window_size=window_size,
        )

        self.ffn = GatedMLP(dim, intermediate_dim=dim * 4)

    def forward(
        self,
        x: Tensor,
        rope: RoPE | None = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            rope: (t, head_dim/2, 2) - Only used if position_embedding_type="rope"
            pos: (l, k) - Position indices for N-D embeddings (optional)

        Outputs:
            out: (b, l, d)
        """

        x = x + self.attn(self.norm1(x), rope, pos)
        x = x + self.ffn(self.norm2(x))

        return x


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

    def __init__(self, dim, dim_mult=2):
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


class TimeMixBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, dim_mult=4):
        super().__init__()
        hidden_dim = int(dim * dim_mult * 2 // 3)
        padding = kernel_size // 2
        self.pwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )
        self.norm = RMSNorm(dim)

        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        h = self.pwconv(x.transpose(1, 2)).transpose(1, 2)
        h = self.norm(h)
        gate, value = self.proj(h).chunk(2, dim=-1)
        gate = self.act(gate)
        gate = self.grn(gate)
        h = gate * value
        h = self.proj_out(h)
        return x + h


class BSRoformerBlock(nn.Module):
    """Band-Split RoFormer block with time and/or frequency attention."""

    def __init__(self, dim, n_heads, axis="tf", use_time_mix=False):
        super().__init__()
        assert axis in ["t", "f", "tf", "ft"], (
            "Axis must be one of 't', 'f', 'tf', or 'ft'."
        )
        self.axis = axis
        self.use_time_mix = use_time_mix
        if "t" in axis:
            if use_time_mix:
                self.time_block = TimeMixBlock(dim)
            else:
                self.time_block = BlockV2(
                    dim, n_heads
                )  # Use BlockV2 (SwiGLU) like original
        self.freq_block = (
            BlockV2(dim, n_heads) if "f" in axis else None
        )  # Use BlockV2 (SwiGLU)

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
            x = rearrange(x, "b d t f -> (b f) t d")
            if self.use_time_mix:
                x = self.time_block(x)
            else:
                x = self.time_block(x, rope=rope, pos=None)
            x = rearrange(x, "(b f) t d -> b d t f", b=B)

        if self.freq_block is not None:
            # Frequency attention
            x = rearrange(x, "b d t f -> (b t) f d")
            x = self.freq_block(x, rope=rope, pos=None)
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
        band_dim=64,
        dim_sp=96,
        dim=384,
        dim_head=32,
        patch_size=[4, 4],
        n_layers=12,
        n_pre_layers=1,
        n_post_layers=1,
        rope_len=8192,
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
            out_channels=band_dim,
        )

        # RoPE
        self.rope = RoPE(head_dim=dim_head, max_len=rope_len)

        # Blocks
        self.patch = nn.Conv2d(band_dim * audio_channels, dim_sp, kernel_size=1)
        self.unpatch = nn.Conv2d(dim_sp, band_dim * audio_channels, kernel_size=1)
        self.down = nn.Conv2d(dim_sp, dim, kernel_size=patch_size, stride=patch_size)
        self.up = nn.ConvTranspose2d(
            dim, dim_sp, kernel_size=patch_size, stride=patch_size
        )
        self.fusion = GEGLUFusion(dim_sp)

        self.pre_blocks = nn.ModuleList(
            [
                BSRoformerBlock(
                    dim_sp, dim_sp // dim_head, axis="tf", use_time_mix=True
                )
                for _ in range(n_pre_layers)
            ]
        )
        self.post_blocks = nn.ModuleList(
            [
                BSRoformerBlock(
                    dim_sp, dim_sp // dim_head, axis="tf", use_time_mix=True
                )
                for _ in range(n_post_layers)
            ]
        )
        self.blocks = nn.ModuleList(
            [BSRoformerBlock(dim, dim // dim_head, axis="tf") for _ in range(n_layers)]
        )

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

        x = self.patch(rearrange(x, "b c t f i -> b (c i) t f"))

        for block in self.pre_blocks:
            x = block(x, rope=self.rope)
        h = x
        x = self.down(x)

        for block in self.blocks:
            x = block(x, rope=self.rope)

        x = self.up(x)
        x = self.fusion(x, h)
        for block in self.post_blocks:
            x = block(x, rope=self.rope)

        # --- Decode ---
        # Unpatchify
        x = self.unpatch(x)
        x = rearrange(x, "b (c i) t f -> b c t f i", c=self.ac)

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
    model = BSRoformer()
    dummy_audio = torch.randn(1, 2, 48000 * 2)
    output = model(dummy_audio)
    print(output.shape)
