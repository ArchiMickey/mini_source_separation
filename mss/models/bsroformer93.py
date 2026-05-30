from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, Tensor

from mss.models.attention import Block
from mss.models.rope import RoPE
from mss.models2.dsp3.banks import erb_linear_banks
from mss.models2.dsp3.subband_fast import SubbandFilter
from mss.utils import fast_sdr


class BSRoformer93(nn.Module):
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
        pre_post_variant: Literal[
            "89c2",
            "90a",
            "91",
            "92",
            "92a",
            "92b",
            "92c",
            "92a_resglu",
        ] = "92",
        layer_scale_init: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()

        unexpected_kwargs = set(kwargs) - {"name"}
        if unexpected_kwargs:
            unexpected = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"Unsupported BSRoformer93 model config keys: {unexpected}")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_size_t = patch_size_t
        self.total_stride_t = self.patch_size_t
        pre_post_variant = str(pre_post_variant)
        layer_scale_init = float(layer_scale_init)

        self.pre_post_variant = pre_post_variant
        self.layer_scale_init = layer_scale_init

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
        self.pre_block = self._make_pre_post_block(
            variant=pre_post_variant,
            channels=in_channels,
            depth=pre_post_depth,
            n_heads=pre_post_heads,
            layer_scale_init=layer_scale_init,
            use_skip=False,
            role="pre",
        )
        self.patch = Patch(in_channels, dim, (self.patch_size_t, 1))
        self.unpatch = UnPatch(dim, in_channels, (self.patch_size_t, 1))
        self.post_block = self._make_pre_post_block(
            variant=pre_post_variant,
            channels=in_channels,
            depth=pre_post_depth,
            n_heads=pre_post_heads,
            layer_scale_init=layer_scale_init,
            use_skip=pre_post_variant != "89c2",
            role="post",
        )

        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)
        self.t_blocks = nn.ModuleList(
            LayerScaleBlock(dim, n_heads, layer_scale_init=layer_scale_init)
            for _ in range(n_layers)
        )
        self.k_blocks = nn.ModuleList(
            LayerScaleBlock(dim, n_heads, layer_scale_init=layer_scale_init)
            for _ in range(n_layers)
        )

    def _make_pre_post_block(
        self,
        variant: str,
        channels: int,
        depth: int,
        n_heads: int,
        layer_scale_init: float,
        use_skip: bool,
        role: Literal["pre", "post"],
    ) -> nn.Module:
        if variant == "89c2":
            return IdentityPrePostBlock()
        if variant == "92c" and role == "pre":
            return IdentityPrePostBlock()
        if variant == "90a":
            return TimeTransformerBlock(
                channels=channels,
                depth=depth,
                n_heads=n_heads,
                layer_scale_init=layer_scale_init,
                use_skip=use_skip,
            )
        if variant == "91":
            return FreqTransformerBlock(
                channels=channels,
                depth=depth,
                n_heads=n_heads,
                layer_scale_init=layer_scale_init,
                use_skip=use_skip,
            )
        if variant == "92b":
            return ParallelTimeFreqTransformerBlock(
                channels=channels,
                depth=depth,
                n_heads=n_heads,
                layer_scale_init=layer_scale_init,
                use_skip=use_skip,
            )
        if variant in {"92", "92a", "92c", "92a_resglu"}:
            return TimeFreqTransformerBlock(
                channels=channels,
                depth=depth,
                n_heads=n_heads,
                layer_scale_init=layer_scale_init,
                use_skip=use_skip,
                use_axis_skips=variant in {"92a", "92a_resglu"},
                residual_axis_skips=variant == "92a_resglu",
            )
        raise ValueError(f"Unsupported pre_post_variant: {variant}")

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
        x = self.pad_tensor(x, self.total_stride_t)

        if self.pre_post_variant == "89c2":
            skip = None
            x = self.patch(x)
        else:
            skip = self.pre_block(x)
            x = self.patch(skip)

        for t_block, k_block in zip(self.t_blocks, self.k_blocks):
            x = rearrange(x, "b d t f -> (b f) t d")
            x = t_block(x, rope=self.rope, pos=None)

            x = rearrange(x, "(b f) t d -> (b t) f d", b=batch_size)
            x = k_block(x, rope=self.rope, pos=None)

            x = rearrange(x, "(b t) f d -> b d t f", b=batch_size)

        x = self.unpatch(x)
        if skip is not None:
            x = self.post_block(x, skip)
        x = x[:, :, 0:frames_num, :]
        x = rearrange(x, "b (c f x) t k -> b c k t f x", c=audio_channels, x=2)
        mask = torch.view_as_complex(x.contiguous().float())
        sep_stft = complex_sp * mask

        x = self.istft(sep_stft)
        out = self.sb_filter.synthesis(x)
        return out[..., 0:audio_length]

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

    def pad_tensor(self, x: Tensor, multiple_t: int) -> Tensor:
        pad_t = -x.shape[2] % multiple_t
        return F.pad(x, pad=(0, 0, 0, pad_t))

    def pad_audio(self, audio: Tensor, multiple_l: int) -> Tensor:
        pad_l = -audio.shape[-1] % multiple_l
        return F.pad(audio, pad=(0, pad_l))

    def check_sdr(self, audio: Tensor, complex_sp: Tensor) -> None:
        y = self.istft(complex_sp)
        y = self.sb_filter.synthesis(y)
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"SDR: {sdr:.2f} dB")


class LayerScaleBlock(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        position_embedding_type: str = "rope",
        max_seq_len: int = 8192,
        attn_backend: Literal["torch", "flash_attn"] = "torch",
        window_size: int | None = None,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            position_embedding_type=position_embedding_type,
            max_seq_len=max_seq_len,
            attn_backend=attn_backend,
            window_size=window_size,
        )
        self.attn_layerscale = nn.Parameter(torch.ones(dim) * layer_scale_init)
        self.ffn_layerscale = nn.Parameter(torch.ones(dim) * layer_scale_init)

    def forward(
        self,
        x: Tensor,
        rope: RoPE | None = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        x = x + self.attn_layerscale * self.attn(self.norm1(x), rope, pos)
        x = x + self.ffn_layerscale * self.ffn(self.norm2(x))
        return x


class IdentityPrePostBlock(nn.Module):
    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        del skip
        return x


class TimeTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        n_heads: int,
        layer_scale_init: float,
        use_skip: bool = False,
    ) -> None:
        super().__init__()
        self.skip = GLUSkipConnection(channels) if use_skip else None
        self.rope = RoPE(head_dim=channels // n_heads, max_len=8192)
        self.blocks = nn.ModuleList(
            LayerScaleBlock(
                channels,
                n_heads,
                position_embedding_type="rope",
                max_seq_len=8192,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(depth)
        )

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        if self.skip is not None:
            if skip is None:
                raise ValueError("skip tensor is required when skip connection is enabled")
            x = self.skip(x, skip)

        batch_size, _, _, freq_bins = x.shape
        for block in self.blocks:
            x = rearrange(x, "b c t k -> (b k) t c")
            x = block(x, rope=self.rope, pos=None)
            x = rearrange(x, "(b k) t c -> b c t k", b=batch_size, k=freq_bins)
        return x


class FreqTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        n_heads: int,
        layer_scale_init: float,
        use_skip: bool = False,
    ) -> None:
        super().__init__()
        self.skip = GLUSkipConnection(channels) if use_skip else None
        self.rope = RoPE(head_dim=channels // n_heads, max_len=8192)
        self.blocks = nn.ModuleList(
            LayerScaleBlock(
                channels,
                n_heads,
                position_embedding_type="rope",
                max_seq_len=8192,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(depth)
        )

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        if self.skip is not None:
            if skip is None:
                raise ValueError("skip tensor is required when skip connection is enabled")
            x = self.skip(x, skip)

        batch_size, _, time_steps, _ = x.shape
        for block in self.blocks:
            x = rearrange(x, "b c t k -> (b t) k c")
            x = block(x, rope=self.rope, pos=None)
            x = rearrange(x, "(b t) k c -> b c t k", b=batch_size, t=time_steps)
        return x


class TimeFreqTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        n_heads: int,
        layer_scale_init: float,
        use_skip: bool = False,
        use_axis_skips: bool = False,
        residual_axis_skips: bool = False,
    ) -> None:
        super().__init__()
        self.skip = GLUSkipConnection(channels) if use_skip else None
        axis_skip_cls = ResidualGLUSkipConnection if residual_axis_skips else GLUSkipConnection
        self.time_axis_skips = (
            nn.ModuleList(axis_skip_cls(channels) for _ in range(depth))
            if use_axis_skips
            else None
        )
        self.freq_axis_skips = (
            nn.ModuleList(axis_skip_cls(channels) for _ in range(depth))
            if use_axis_skips
            else None
        )
        self.rope = RoPE(head_dim=channels // n_heads, max_len=8192)
        self.time_blocks = nn.ModuleList(
            LayerScaleBlock(
                channels,
                n_heads,
                position_embedding_type="rope",
                max_seq_len=8192,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(depth)
        )
        self.freq_blocks = nn.ModuleList(
            LayerScaleBlock(
                channels,
                n_heads,
                position_embedding_type="rope",
                max_seq_len=8192,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(depth)
        )

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        if self.skip is not None:
            if skip is None:
                raise ValueError("skip tensor is required when skip connection is enabled")
            x = self.skip(x, skip)

        batch_size, _, _, freq_bins = x.shape
        for idx, (time_block, freq_block) in enumerate(
            zip(self.time_blocks, self.freq_blocks)
        ):
            time_skip = x
            x = rearrange(x, "b c t k -> (b k) t c")
            x = time_block(x, rope=self.rope, pos=None)
            x = rearrange(x, "(b k) t c -> b c t k", b=batch_size, k=freq_bins)
            if self.time_axis_skips is not None:
                x = self.time_axis_skips[idx](x, time_skip)

            freq_skip = x
            x = rearrange(x, "b c t k -> (b t) k c")
            x = freq_block(x, rope=self.rope, pos=None)
            x = rearrange(x, "(b t) k c -> b c t k", b=batch_size)
            if self.freq_axis_skips is not None:
                x = self.freq_axis_skips[idx](x, freq_skip)
        return x


class ParallelTimeFreqTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        n_heads: int,
        layer_scale_init: float,
        use_skip: bool = False,
    ) -> None:
        super().__init__()
        self.skip = GLUSkipConnection(channels) if use_skip else None
        self.rope = RoPE(head_dim=channels // n_heads, max_len=8192)
        self.time_blocks = nn.ModuleList(
            LayerScaleBlock(
                channels,
                n_heads,
                position_embedding_type="rope",
                max_seq_len=8192,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(depth)
        )
        self.freq_blocks = nn.ModuleList(
            LayerScaleBlock(
                channels,
                n_heads,
                position_embedding_type="rope",
                max_seq_len=8192,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(depth)
        )
        self.time_residual_scale = nn.Parameter(torch.zeros(depth, channels, 1, 1))
        self.freq_residual_scale = nn.Parameter(torch.zeros(depth, channels, 1, 1))

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        if self.skip is not None:
            if skip is None:
                raise ValueError("skip tensor is required when skip connection is enabled")
            x = self.skip(x, skip)

        batch_size, _, time_steps, freq_bins = x.shape
        for idx, (time_block, freq_block) in enumerate(
            zip(self.time_blocks, self.freq_blocks)
        ):
            identity = x

            time_out = rearrange(identity, "b c t k -> (b k) t c")
            time_out = time_block(time_out, rope=self.rope, pos=None)
            time_out = rearrange(
                time_out, "(b k) t c -> b c t k", b=batch_size, k=freq_bins
            )

            freq_out = rearrange(identity, "b c t k -> (b t) k c")
            freq_out = freq_block(freq_out, rope=self.rope, pos=None)
            freq_out = rearrange(
                freq_out, "(b t) k c -> b c t k", b=batch_size, t=time_steps
            )

            x = (
                identity
                + self.time_residual_scale[idx] * (time_out - identity)
                + self.freq_residual_scale[idx] * (freq_out - identity)
            )
        return x


class GLUSkipConnection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels * 2, channels * 2, kernel_size=1)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        gate, value = torch.chunk(self.proj(torch.cat((x, skip), dim=1)), 2, dim=1)
        gate = F.gelu(gate)
        return gate * value


class ResidualGLUSkipConnection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.glu = GLUSkipConnection(channels)
        self.residual_scale = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return x + self.residual_scale * self.glu(x, skip)


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
