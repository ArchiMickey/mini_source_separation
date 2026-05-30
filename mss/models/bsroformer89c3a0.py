from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, Tensor

from mss.models2.dsp3.banks import erb_linear_banks
from mss.models2.dsp3.subband_fast import SubbandFilter
from mss.utils import fast_sdr

try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class BSRoformer89c3a0(nn.Module):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_layers=12,
        n_heads=12,
        dim=768,
        rope_len=8192,
        hop_length=4,
        patch_size=(4, 1),
        time_compaction_size: int | None = None,
        patch_roformer_layers: int = 4,
        subband_n_bands: int = 112,
        max_bandwidth: int = 390,
        chunk_size: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()

        # Keep the legacy YAML `n_bands` key ignored for compatibility. Existing
        # 89c3a configs carried stale `n_bands: 256` while the model used 112.
        kwargs.pop("n_bands", None)
        kwargs.pop("band_dim", None)
        del kwargs

        self.audio_channels = audio_channels
        self.dim = dim
        self.subband_n_bands = subband_n_bands
        self.hop_length = int(hop_length)
        self.time_compaction_size = int(
            time_compaction_size if time_compaction_size is not None else patch_size[0]
        )
        if self.time_compaction_size < 1:
            raise ValueError("time_compaction_size must be >= 1")
        self.patch_size = (int(patch_size[0]), 1)
        self.patch_roformer_layers = int(patch_roformer_layers)
        if self.patch_roformer_layers < 0:
            raise ValueError("patch_roformer_layers must be >= 0")
        if self.patch_roformer_layers > n_layers:
            raise ValueError("patch_roformer_layers must be <= n_layers")

        factor = sample_rate // 400
        self.sb_factor = factor
        self._last_cnn_input_length: int | None = None

        banks = erb_linear_banks(
            sr=sample_rate,
            n_bands=subband_n_bands,
            max_bandwidth=max_bandwidth,
        )
        self.sb_filter = SubbandFilter(
            sample_rate,
            banks,
            factor,
            chunk_size=chunk_size,
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

        self.patch = Patch(dim, dim, self.patch_size)
        self.unpatch = UnPatch(dim, dim, self.patch_size)
        self.unpatch_skip_alpha = nn.Parameter(torch.tensor(0.0))
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)

        compacted_layers = n_layers - self.patch_roformer_layers
        self.patch_t_blocks = nn.ModuleList(
            RoformerBlock(dim, n_heads) for _ in range(self.patch_roformer_layers)
        )
        self.patch_k_blocks = nn.ModuleList(
            RoformerBlock(dim, n_heads) for _ in range(self.patch_roformer_layers)
        )
        self.t_blocks = nn.ModuleList(
            TimeCompactedBlock(dim, n_heads, self.time_compaction_size)
            for _ in range(compacted_layers)
        )
        self.k_blocks = nn.ModuleList(
            FreqCompactedBlock(dim, n_heads, self.time_compaction_size)
            for _ in range(compacted_layers)
        )

    def forward(self, audio: Tensor) -> Tensor:
        audio_length = audio.shape[-1]
        audio = self.pad_audio(audio, self.sb_factor)
        x = self.sb_filter.analysis(audio)
        features = self.cnn_analysis(x)

        if False:
            self.check_sdr(audio, features)
            os._exit(0)

        batch, _, time_steps, freq_bins = features.shape
        x = features

        if self.patch_roformer_layers > 0:
            x = self.pad_tensor(x, self.patch_size)
            x = self.patch(x)

            for t_block, k_block in zip(self.patch_t_blocks, self.patch_k_blocks):
                x = rearrange(x, "b d t f -> (b f) t d")
                x = t_block(x, rope=self.rope, pos=None)

                x = rearrange(x, "(b f) t d -> (b t) f d", b=batch)
                x = k_block(x, rope=self.rope, pos=None)

                x = rearrange(x, "(b t) f d -> b d t f", b=batch)

            x = self.unpatch(x)
            x = x[:, :, 0:time_steps, 0:freq_bins]
            x = x + self.unpatch_skip_alpha * features

        x = self.pad_tensor(x, self.time_compaction_size)

        for t_block, k_block in zip(self.t_blocks, self.k_blocks):
            x = t_block(x, rope=self.rope, pos=None)
            x = k_block(x, rope=self.rope, pos=None)

        x = x[:, :, 0:time_steps, :]
        mask = x
        x = features * mask

        x = self.cnn_synthesis(x)
        out = self.sb_filter.synthesis(x)

        return out[..., 0:audio_length]

    def cnn_analysis(self, x: Tensor) -> Tensor:
        B, C, K, L = x.shape
        self._last_cnn_input_length = L
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        x = torch.view_as_real(x.contiguous())
        x = rearrange(x, "b c k l ri -> b (c ri) l k")
        x = self._pad_for_cnn_analysis(x)
        return self.cnn_encoder(x)

    def cnn_synthesis(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        C = self.audio_channels
        K = x.shape[-1]
        target_length = self._last_cnn_input_length
        if target_length is None:
            raise RuntimeError(
                "cnn_analysis() must be called before cnn_synthesis() so synthesis length is known"
            )

        x = self.cnn_decoder(x)
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

    def pad_tensor(self, x: Tensor, patch_size) -> Tensor:
        if isinstance(patch_size, int):
            patch_t, patch_f = patch_size, 1
        else:
            patch_t, patch_f = patch_size
        pad_t = -x.shape[2] % patch_t
        pad_f = -x.shape[3] % patch_f
        return F.pad(x, pad=(0, pad_f, 0, pad_t))

    def pad_audio(self, audio: Tensor, multiple_l: int) -> Tensor:
        pad_l = -audio.shape[-1] % multiple_l
        return F.pad(audio, pad=(0, pad_l))

    def check_sdr(self, audio, complex_sp) -> None:
        y = self.cnn_synthesis(complex_sp)
        y = self.sb_filter.synthesis(y)
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"SDR: {sdr:.2f} dB")


class RoformerBlock(nn.Module):
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
        rope: "RoPE | None" = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), rope, pos)
        x = x + self.ffn(self.norm2(x))
        return x


class TimeCompactedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_compaction_size: int,
        position_embedding_type: str = "rope",
        max_seq_len: int = 8192,
        attn_backend: Literal["torch", "flash_attn"] = "torch",
        window_size: int | None = None,
    ) -> None:
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = TimeCompactedSelfAttention(
            dim,
            num_heads,
            time_compaction_size,
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
        rope: "RoPE | None" = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        x = x + self.attn(norm_4d(self.norm1, x), rope, pos)
        x = x + apply_4d(self.ffn, norm_4d(self.norm2, x))
        return x


class FreqCompactedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_compaction_size: int,
        position_embedding_type: str = "rope",
        max_seq_len: int = 8192,
        attn_backend: Literal["torch", "flash_attn"] = "torch",
        window_size: int | None = None,
    ) -> None:
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = FreqCompactedSelfAttention(
            dim,
            num_heads,
            time_compaction_size,
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
        rope: "RoPE | None" = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        x = x + self.attn(norm_4d(self.norm1, x), rope, pos)
        x = x + apply_4d(self.ffn, norm_4d(self.norm2, x))
        return x


class TimeCompactedSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_compaction_size: int,
        position_embedding_type: str = "rope",
        max_seq_len: int = 8192,
        attn_backend: Literal["torch", "flash_attn"] = "torch",
        window_size: int | None = None,
    ) -> None:
        super().__init__()
        self.time_compaction = TimeCompaction(time_compaction_size)
        self.attn = SelfAttention(
            dim,
            num_heads,
            position_embedding_type=position_embedding_type,
            max_seq_len=max_seq_len,
            attn_backend=attn_backend,
            window_size=window_size,
        )

    def forward(
        self,
        x: Tensor,
        rope: "RoPE | None" = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        batch, _, time_steps, freq_bins = x.shape
        compact = self.time_compaction.compress_4d(x)
        tokens = rearrange(compact, "b d t f -> (b f) t d")
        tokens = self.attn(tokens, rope, pos)
        compact = rearrange(tokens, "(b f) t d -> b d t f", b=batch, f=freq_bins)
        return self.time_compaction.decompress_4d(compact, time_steps)


class FreqCompactedSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_compaction_size: int,
        position_embedding_type: str = "rope",
        max_seq_len: int = 8192,
        attn_backend: Literal["torch", "flash_attn"] = "torch",
        window_size: int | None = None,
    ) -> None:
        super().__init__()
        self.time_compaction = TimeCompaction(time_compaction_size)
        self.attn = SelfAttention(
            dim,
            num_heads,
            position_embedding_type=position_embedding_type,
            max_seq_len=max_seq_len,
            attn_backend=attn_backend,
            window_size=window_size,
        )

    def forward(
        self,
        x: Tensor,
        rope: "RoPE | None" = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        batch, _, time_steps, freq_bins = x.shape
        compact = self.time_compaction.compress_4d(x)
        tokens = rearrange(compact, "b d t f -> (b t) f d")
        tokens = self.attn(tokens, rope, pos)
        compact = rearrange(tokens, "(b t) f d -> b d t f", b=batch, f=freq_bins)
        return self.time_compaction.decompress_4d(compact, time_steps)


class TimeCompaction(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        self.patch_size = patch_size
        self.compress = nn.Linear(patch_size, 1)
        self.decompress = nn.Linear(1, patch_size)

    def compress_4d(self, x: Tensor) -> Tensor:
        pad_t = -x.shape[2] % self.patch_size
        if pad_t > 0:
            x = F.pad(x, pad=(0, 0, 0, pad_t))
        x = rearrange(x, "b d (t p) f -> b d t f p", p=self.patch_size)
        return self.compress(x).squeeze(-1)

    def decompress_4d(self, x: Tensor, time_steps: int) -> Tensor:
        x = self.decompress(x.unsqueeze(-1))
        x = rearrange(x, "b d t f p -> b d (t p) f")
        return x[:, :, 0:time_steps, :]


class Patch(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: tuple[int, int]
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UnPatch(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: tuple[int, int]
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


def norm_4d(norm: nn.Module, x: Tensor) -> Tensor:
    return rearrange(norm(rearrange(x, "b d t f -> b t f d")), "b t f d -> b d t f")


def apply_4d(module: nn.Module, x: Tensor) -> Tensor:
    return rearrange(module(rearrange(x, "b d t f -> b t f d")), "b t f d -> b d t f")


class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, dim: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, dim) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        B, L, N, H = x.shape
        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds maximum supported length {self.max_seq_len}. "
                "Please increase max_seq_len in the configuration."
            )

        pos_emb = self.pos_emb[:L]
        pos_emb = pos_emb.view(1, L, 1, -1)
        return x + pos_emb

    def apply_nd(self, x: Tensor, pos: LongTensor) -> Tensor:
        B, L, N, H = x.shape
        K = pos.shape[1]
        dim_per_axis = self.dim // K
        pos_emb_sum = torch.zeros(L, self.dim, device=x.device, dtype=x.dtype)

        for i in range(K):
            p = pos[:, i]
            if torch.any(p >= self.max_seq_len):
                raise ValueError(
                    f"Position index {p.max().item()} exceeds maximum supported length {self.max_seq_len}"
                )

            start_idx = i * dim_per_axis
            end_idx = start_idx + dim_per_axis if i < K - 1 else self.dim
            pos_emb_sum[:, start_idx:end_idx] = self.pos_emb[p, start_idx:end_idx]

        pos_emb_sum = pos_emb_sum.view(1, L, 1, -1)
        return x + pos_emb_sum


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm_x + self.eps) * self.scale


class SelfAttention(nn.Module):
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

        valid_backends = ["torch", "flash_attn"]
        if attn_backend not in valid_backends:
            raise ValueError(
                f"Invalid attn_backend: '{attn_backend}'. Must be one of {valid_backends}"
            )

        if attn_backend == "flash_attn" and not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "flash_attn backend requested but flash_attn is not installed. "
                "Please install it with: pip install flash-attn --no-build-isolation"
            )

        self.attn_backend = attn_backend

        valid_types = ["rope", "absolute"]
        if position_embedding_type not in valid_types:
            raise ValueError(
                f"Invalid position_embedding_type: '{position_embedding_type}'. "
                f"Must be one of {valid_types}"
            )

        self.position_embedding_type = position_embedding_type
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
        if self.window_size is None:
            return None

        positions = torch.arange(seq_len, device=device)
        distance = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
        return distance <= self.window_size // 2

    def forward(
        self,
        x: Tensor,
        rope: nn.Module | None = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        q, k, v = self.qkv_linear(x).chunk(chunks=3, dim=2)
        q = rearrange(self.norm_q(q), "b l (n h) -> b l n h", h=self.head_dim)
        k = rearrange(self.norm_k(k), "b l (n h) -> b l n h", h=self.head_dim)
        v = rearrange(v, "b l (n h) -> b l n h", h=self.head_dim)

        if self.position_embedding_type == "rope":
            if rope is None:
                raise ValueError("RoPE module must be provided when position_embedding_type='rope'")

            if pos is None:
                q = rope(q)
                k = rope(k)
            else:
                q = rope.apply_nd(q, pos)
                k = rope.apply_nd(k, pos)
        elif self.position_embedding_type == "absolute":
            if pos is None:
                q = self.pos_embedding(q)
                k = self.pos_embedding(k)
            else:
                q = self.pos_embedding.apply_nd(q, pos)
                k = self.pos_embedding.apply_nd(k, pos)

        if self.attn_backend == "flash_attn":
            if self.window_size is not None:
                window_size_half = self.window_size // 2
                window_size_tuple = (window_size_half, window_size_half)
            else:
                window_size_tuple = (-1, -1)

            x = flash_attn_func(q, k, v, window_size=window_size_tuple)
        else:
            attn_mask = None
            if self.window_size is not None:
                attn_mask = self._create_window_mask(x.shape[1], x.device)
                attn_mask = ~attn_mask

            x = F.scaled_dot_product_attention(
                query=rearrange(q, "b l n h -> b n l h"),
                key=rearrange(k, "b l n h -> b n l h"),
                value=rearrange(v, "b l n h -> b n l h"),
                attn_mask=attn_mask,
                dropout_p=0.0,
            )
            x = rearrange(x, "b n l h -> b l n h")

        x = rearrange(x, "b l n h -> b l (n h)")
        return self.proj(x)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 8192, base: int = 10000):
        super().__init__()
        self.head_dim = head_dim

        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
        pos_theta = torch.outer(torch.arange(max_len), theta).float()
        w = torch.stack([torch.cos(pos_theta), torch.sin(pos_theta)], dim=-1)
        self.register_buffer(name="w", tensor=w)

    def forward(self, x: Tensor) -> Tensor:
        L = x.shape[1]
        x = rearrange(x, "b l n (h c) -> b l n h c", c=2)
        w = self.w[0:L][None, :, None, :, :]
        x = self.rotate(x, w)
        return rearrange(x, "b l n h c -> b l n (h c)")

    def rotate(self, x: Tensor, w: Tensor) -> Tensor:
        return torch.stack(
            [
                w[..., 0] * x[..., 0] - w[..., 1] * x[..., 1],
                w[..., 0] * x[..., 1] + w[..., 1] * x[..., 0],
            ],
            dim=-1,
        )

    def apply_nd(self, x: Tensor, pos: LongTensor) -> Tensor:
        B, L, N, H = x.shape
        K = pos.shape[1]
        assert H == K * self.head_dim

        x = rearrange(x, "b l n (k h c) -> k b l n h c", k=K, c=2)
        out = torch.zeros_like(x, device=x.device)

        for i in range(K):
            p = pos[:, i]
            w = self.w[p][None, :, None, :, :]
            out[i] = self.rotate(x[i], w)

        return rearrange(out, "k b l n h c -> b l n (k h c)")


if __name__ == "__main__":
    model = BSRoformer89c3a0()
    audio = torch.randn(2, 2, 48000 * 2)
    out = model(audio)
    print(out.shape)
