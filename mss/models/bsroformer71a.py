from __future__ import annotations

import math

from torch import Tensor, LongTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from mss.models.bsroformer66 import (
    BandGatedRMSNorm,
    BSRoformerBlock,
    GEGLUFusion,
    MultidimRoPE,
    RoPE1D,
)
from mss.models.fourier import Fourier
from mss.models2.dsp3.banks import erb_linear_banks
from mss.models2.dsp3.subband_fast import SubbandFilter


class LightweightCrossAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int):
        super().__init__()
        if dim % dim_head != 0:
            raise ValueError(f"dim ({dim}) must be divisible by dim_head ({dim_head})")

        self.n_heads = dim // dim_head
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        return rearrange(x, "b n (h d) -> b h n d", h=self.n_heads)

    def forward(self, q_input: Tensor, kv_input: Tensor) -> Tensor:
        q = self.q_proj(self.q_norm(q_input))
        kv = self.kv_norm(kv_input)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        out = F.scaled_dot_product_attention(
            self._reshape_heads(q),
            self._reshape_heads(k),
            self._reshape_heads(v),
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.out_proj(out)


class AttentionPool(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int,
        n_bands_full: int,
        pool_factor: int,
    ):
        super().__init__()
        self.pool_factor = max(1, int(pool_factor))
        self.n_registers = math.ceil(n_bands_full / self.pool_factor)
        self.band_pos = nn.Parameter(torch.zeros(1, n_bands_full, dim))
        self.pool_gate = nn.Parameter(torch.tensor(0.1))
        self.refine_attn = LightweightCrossAttention(dim=dim, dim_head=dim_head)

    def forward(self, x: Tensor) -> Tensor:
        b, _, t, freq_bins = x.shape
        if freq_bins > self.band_pos.shape[1]:
            raise ValueError(
                f"Input has {freq_bins} bands, but AttentionPool was built for {self.band_pos.shape[1]}"
            )

        band_tokens = rearrange(x, "b c t f -> (b t) f c")
        band_tokens_with_pos = band_tokens + self.band_pos[:, :freq_bins]

        pooled_seed = F.adaptive_avg_pool1d(
            rearrange(band_tokens, "bt f c -> bt c f"),
            self.n_registers,
        )
        pooled_seed = rearrange(pooled_seed, "bt c r -> bt r c")
        pooled_pos = F.adaptive_avg_pool1d(
            rearrange(self.band_pos[:, :freq_bins], "b f c -> b c f"),
            self.n_registers,
        )
        pooled_pos = rearrange(pooled_pos, "b c r -> b r c")
        pooled_queries = pooled_seed + pooled_pos
        pooled_delta = self.refine_attn(pooled_queries, band_tokens_with_pos)
        pooled = pooled_seed + self.pool_gate.tanh() * pooled_delta
        return rearrange(pooled, "(b t) r c -> b c t r", b=b, t=t)


class AttentionUpsample(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int,
        n_bands_full: int,
        n_registers: int,
    ):
        super().__init__()
        self.n_registers = n_registers
        self.band_pos = nn.Parameter(torch.zeros(1, n_bands_full, dim))
        self.attn = LightweightCrossAttention(dim=dim, dim_head=dim_head)
        self.attn_gate = nn.Parameter(torch.tensor(0.1))
        self.query_gate_proj = nn.Linear(dim * 2, dim)
        self.skip_gate_proj = nn.Linear(dim * 2, dim)
        nn.init.zeros_(self.query_gate_proj.weight)
        nn.init.zeros_(self.skip_gate_proj.weight)
        nn.init.constant_(self.query_gate_proj.bias, -2.0)
        nn.init.constant_(self.skip_gate_proj.bias, -2.0)

    def forward(self, x: Tensor, band_query: Tensor) -> Tensor:
        b, _, t, n_registers = x.shape
        freq_bins = band_query.shape[3]

        if n_registers != self.n_registers:
            raise ValueError(
                f"Expected {self.n_registers} pooled registers, but got {n_registers}"
            )

        if freq_bins > self.band_pos.shape[1]:
            raise ValueError(
                f"Band query has {freq_bins} bands, but AttentionUpsample was built for {self.band_pos.shape[1]}"
            )

        query_tokens = rearrange(band_query, "b c t f -> (b t) f c")
        band_pos = self.band_pos[:, :freq_bins]

        register_tokens = rearrange(x, "b c t r -> (b t) r c")
        register_pos = F.adaptive_avg_pool1d(
            rearrange(band_pos, "b f c -> b c f"),
            n_registers,
        )
        register_pos = rearrange(register_pos, "b c r -> b r c")
        registers = register_tokens + register_pos

        if n_registers == freq_bins:
            mix_features = torch.cat([register_tokens, query_tokens], dim=-1)
            query_gate = torch.sigmoid(self.query_gate_proj(mix_features))
            base_queries = register_tokens + query_gate * (
                query_tokens - register_tokens
            )
            queries = base_queries + band_pos
        else:
            queries = query_tokens + band_pos

        upsampled = self.attn(queries, registers)
        if n_registers == freq_bins:
            mix_features = torch.cat([register_tokens, query_tokens], dim=-1)
            skip_gate = torch.sigmoid(self.skip_gate_proj(mix_features))
            output_tokens = (
                register_tokens
                + skip_gate * (query_tokens - register_tokens)
                + self.attn_gate.tanh() * upsampled
            )
        else:
            output_tokens = query_tokens + self.attn_gate.tanh() * upsampled
        return rearrange(output_tokens, "(b t) f c -> b c t f", b=b, t=t)


class BSRoformer(Fourier):
    r"""BSRoformer with direct subband patching and attention pool/upsample."""

    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_bands=128,
        dim_sp=96,
        dim=384,
        dim_head=32,
        patch_size=[4, 4],
        freq_pool_factor=None,
        n_layers=12,
        n_pre_layers=1,
        n_post_layers=1,
        pre_time_block: bool = False,
        post_time_block: bool = False,
        use_checkpoint: bool = False,
        delta_attn_ratio=3,
        rope_len=8192,
        conv_kernel_size=7,
        frame_rate: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(n_fft=64, hop_length=8, return_complex=True, normalized=True)

        max_bandwidth = 400
        factor = sample_rate // max_bandwidth
        chunk_size = 16

        self.ac = audio_channels
        if isinstance(patch_size, (list, tuple)):
            patch_size = tuple(patch_size)
        else:
            patch_size = (patch_size, 1)
        if len(patch_size) == 1:
            patch_size = (patch_size[0], 1)

        # Direct subband patching always keeps frequency kernel/stride at 1.
        # patch_size[1] is retained only as the legacy default for freq_pool_factor.
        self.patch_size = patch_size
        self.patch_size_t = self.patch_size[0]
        self.freq_pool_factor = (
            self.patch_size[1] if freq_pool_factor is None else freq_pool_factor
        )
        self.use_checkpoint = use_checkpoint
        self.frame_rate = frame_rate
        self.subband_rate = sample_rate / factor
        self.frame_hop = max(1, round(self.subband_rate / self.frame_rate))

        banks = erb_linear_banks(
            sr=sample_rate, n_bands=n_bands, max_bandwidth=max_bandwidth
        )
        self.sb_filter = SubbandFilter(
            sr=sample_rate, banks=banks, factor=factor, chunk_size=chunk_size
        )

        self.rope = MultidimRoPE(head_dim=dim_head, n_dims=2, max_len=rope_len)
        self.freq_rope = RoPE1D(head_dim=dim_head, max_len=rope_len)

        in_channels = audio_channels * 2
        self.patchify = nn.Conv2d(
            in_channels,
            dim_sp,
            kernel_size=(self.frame_hop, 1),
            stride=(self.frame_hop, 1),
        )
        self.unpatchify = nn.ConvTranspose2d(
            dim_sp,
            in_channels,
            kernel_size=(self.frame_hop, 1),
            stride=(self.frame_hop, 1),
        )

        self.down = nn.Conv2d(
            dim_sp,
            dim,
            kernel_size=(self.patch_size_t, 1),
            stride=(self.patch_size_t, 1),
        )
        self.up = nn.ConvTranspose2d(
            dim,
            dim_sp,
            kernel_size=(self.patch_size_t, 1),
            stride=(self.patch_size_t, 1),
        )

        n_bands_full = len(banks)
        n_bands_down = math.ceil(n_bands_full / max(1, self.freq_pool_factor))
        self.freq_pool = AttentionPool(
            dim=dim,
            dim_head=dim_head,
            n_bands_full=n_bands_full,
            pool_factor=self.freq_pool_factor,
        )
        self.freq_upsample = AttentionUpsample(
            dim=dim,
            dim_head=dim_head,
            n_bands_full=n_bands_full,
            n_registers=n_bands_down,
        )

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

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            use_deltanet = (i % delta_attn_ratio + 1) != delta_attn_ratio
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

        self.final_norm = BandGatedRMSNorm(dim_sp, n_bands=n_bands_full)
        self._time_rope_position_cache: dict[
            tuple[int, int], tuple[LongTensor, LongTensor]
        ] = {}

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
            lambda inp, tr, fr, tp: block(inp, time_rope=tr, freq_rope=fr, time_pos=tp),
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
            self._time_rope_position_cache[cache_key] = self.rope.build_grid_positions(
                time_steps=time_steps,
                freq_bins=freq_bins,
                device=torch.device("cpu"),
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

    def pad_tensor(self, x: Tensor, patch_size_t: int) -> Tensor:
        pad_t = -x.shape[2] % patch_size_t
        return F.pad(x, pad=(0, 0, 0, pad_t))

    def forward(self, audio: Tensor) -> Tensor:
        subbands = self.sb_filter.analysis(audio)
        t0 = subbands.shape[3]

        x = rearrange(torch.view_as_real(subbands), "b c k t i -> b (c i) t k")
        x = self.pad_tensor(x, self.frame_hop)
        x = self.patchify(x)

        b = x.shape[0]
        t_full, freq_full = x.shape[2], x.shape[3]
        time_pos_full, _ = self._get_time_axis_positions(b, t_full, freq_full, x.device)

        for block in self.pre_blocks:
            x = self._maybe_checkpoint_block(
                block,
                x,
                time_rope=self.rope,
                freq_rope=self.freq_rope,
                time_pos=time_pos_full,
            )

        h = x
        x = self.pad_tensor(x, self.patch_size_t)
        x = self.down(x)
        band_query = x
        x = self.freq_pool(x)

        t_down, f_down = x.shape[2], x.shape[3]
        time_pos_down, _ = self._get_time_axis_positions(b, t_down, f_down, x.device)

        for block in self.blocks:
            x = block(
                x,
                time_rope=self.rope,
                freq_rope=self.freq_rope,
                time_pos=time_pos_down,
            )

        x = self.freq_upsample(x, band_query)
        x = self.up(x)
        x = x[:, :, : h.shape[2], : h.shape[3]]
        x = self.fusion(x, h)

        for block in self.post_blocks:
            x = self._maybe_checkpoint_block(
                block,
                x,
                time_rope=self.rope,
                freq_rope=self.freq_rope,
                time_pos=time_pos_full,
            )

        x = self.final_norm(x)
        x = self.unpatchify(x)
        x = x[:, :, :t0, :]
        x = rearrange(x, "b (c i) t k -> b c k t i", c=self.ac, i=2).contiguous()

        mask = torch.view_as_complex(x.float())
        out = mask * subbands
        return self.sb_filter.synthesis(out)


if __name__ == "__main__":
    model = BSRoformer(use_checkpoint=True)
    dummy_audio = torch.randn(1, 2, 48000 * 2)
    output = model(dummy_audio)
    print(output.shape)
