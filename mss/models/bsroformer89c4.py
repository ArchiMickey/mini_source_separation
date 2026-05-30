from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models.bsroformer56b import BSRoformerBlock, GEGLUFusion
from mss.models.rope import RoPE
from mss.models2.dsp3.banks import (
    erb_linear_banks,
    erb_linear_banks_overlap,
    linear_banks,
    mel_linear_banks,
)
from mss.models2.dsp3.subband_fast import SubbandFilter


class BSRoformer89c4(nn.Module):
    """56b RoFormer body with the 89c3a ERB-linear subband filter frontend."""

    def __init__(
        self,
        audio_channels: int = 2,
        sample_rate: int = 48000,
        n_bands: int | None = None,
        subband_n_bands: int | None = None,
        dim_sp: int = 96,
        dim: int = 384,
        dim_head: int = 32,
        patch_size: list[int] | tuple[int, int] = (4, 4),
        n_layers: int = 12,
        n_pre_layers: int = 1,
        n_post_layers: int = 1,
        rope_len: int = 8192,
        hop_length: int = 4,
        bank_type: str = "erb_linear",
        subband_factor: int | None = None,
        max_bandwidth: int = 390,
        chunk_size: int = 16,
        bandpass_filter_len: int = 48000,
        upsample_filter_len: int = 12000,
        debug_print_shapes: bool = False,
        use_fusion: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # Kept only for config compatibility with 56b YAMLs.
        kwargs.pop("n_fft", None)
        kwargs.pop("band_dim", None)
        del kwargs

        self.audio_channels = audio_channels
        self.patch_size = tuple(int(v) for v in patch_size)
        self.subband_n_bands = int(subband_n_bands or n_bands or 112)
        self.hop_length = int(hop_length)
        self.bank_type = bank_type
        self.sb_factor = int(subband_factor or max(sample_rate // 400, 1))
        self._last_cnn_input_length: int | None = None
        self.debug_print_shapes = debug_print_shapes
        self.use_fusion = bool(use_fusion)

        banks = self._build_banks(sample_rate, self.subband_n_bands, max_bandwidth)
        self.sb_filter = SubbandFilter(
            sample_rate,
            banks,
            self.sb_factor,
            chunk_size=chunk_size,
            bandpass_filter_len=bandpass_filter_len,
            upsample_filter_len=upsample_filter_len,
        )

        self.rope = RoPE(head_dim=dim_head, max_len=rope_len)

        self.cnn_encoder = nn.Conv2d(
            audio_channels * 2,
            dim_sp,
            kernel_size=(self.hop_length, 1),
            stride=(self.hop_length, 1),
            bias=False,
        )
        self.cnn_decoder = nn.ConvTranspose2d(
            dim_sp,
            audio_channels * 2,
            kernel_size=(self.hop_length, 1),
            stride=(self.hop_length, 1),
            bias=False,
        )
        self.down = nn.Conv2d(dim_sp, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.up = nn.ConvTranspose2d(dim, dim_sp, kernel_size=self.patch_size, stride=self.patch_size)
        self.fusion = GEGLUFusion(dim_sp) if self.use_fusion else None

        self.pre_blocks = nn.ModuleList(
            [
                BSRoformerBlock(
                    dim_sp,
                    dim_sp // dim_head,
                    axis="tf",
                    use_time_mix=True,
                )
                for _ in range(n_pre_layers)
            ]
        )
        self.post_blocks = nn.ModuleList(
            [
                BSRoformerBlock(
                    dim_sp,
                    dim_sp // dim_head,
                    axis="tf",
                    use_time_mix=True,
                )
                for _ in range(n_post_layers)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                BSRoformerBlock(dim, dim // dim_head, axis="tf")
                for _ in range(n_layers)
            ]
        )

    def _build_banks(
        self,
        sample_rate: int,
        n_bands: int,
        max_bandwidth: int,
    ) -> list[tuple[float, float]]:
        if self.bank_type == "erb_linear":
            return erb_linear_banks(sample_rate, n_bands, max_bandwidth)
        if self.bank_type == "erb_linear_overlap":
            return erb_linear_banks_overlap(sample_rate, n_bands, max_bandwidth)
        if self.bank_type == "mel_linear":
            return mel_linear_banks(sample_rate, n_bands, max_bandwidth)
        if self.bank_type == "linear":
            return linear_banks(sample_rate, n_bands)
        raise ValueError(f"Unsupported bank_type: {self.bank_type}")

    def forward(self, audio: Tensor) -> Tensor:
        audio_length = audio.shape[-1]
        audio = self.pad_audio(audio, self.sb_factor)
        subbands = self.sb_filter.analysis(audio)
        features = self.cnn_analysis(subbands)
        frames_num = features.shape[2]
        bands_num = features.shape[3]

        x = self.pad_tensor(features)
        if self.debug_print_shapes:
            print(f"down_before: {tuple(x.shape)}")
        x = features = x

        for block in self.pre_blocks:
            x = block(x, rope=self.rope)
        h = x
        x = self.down(x)
        if self.debug_print_shapes:
            print(f"down_after: {tuple(x.shape)}")


        for block in self.blocks:
            x = block(x, rope=self.rope)

        x = self.up(x)
        if self.fusion is not None:
            x = self.fusion(x, h)
        for block in self.post_blocks:
            x = block(x, rope=self.rope)

        if self.debug_print_shapes:
            print(f"mask_before_crop: {tuple(x.shape)}")
        mask = x[:, :, :frames_num, :bands_num]
        x = features[:, :, :frames_num, :bands_num] * mask
        x = self.cnn_synthesis(x)
        out = self.sb_filter.synthesis(x)
        return self.trim_or_pad_output(out, audio_length)

    def cnn_analysis(self, x: Tensor) -> Tensor:
        self._last_cnn_input_length = x.shape[-1]
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        x = torch.view_as_real(x.contiguous())
        x = rearrange(x, "b c f t ri -> b (c ri) t f")
        x = self._pad_for_cnn_analysis(x)
        if self.debug_print_shapes:
            print(f"cnn_analysis_before: {tuple(x.shape)}")
        x = self.cnn_encoder(x)
        if self.debug_print_shapes:
            print(f"cnn_analysis_after: {tuple(x.shape)}")
        return x

    def cnn_synthesis(self, x: Tensor) -> Tensor:
        target_length = self._last_cnn_input_length
        if target_length is None:
            raise RuntimeError("cnn_analysis() must be called before cnn_synthesis() so synthesis length is known")

        if self.debug_print_shapes:
            print(f"cnn_synthesis_before: {tuple(x.shape)}")
        x = self.cnn_decoder(x)
        x = x[:, :, :target_length, :]
        if x.shape[2] < target_length:
            x = F.pad(x, pad=(0, 0, 0, target_length - x.shape[2]))
        if self.debug_print_shapes:
            print(f"cnn_synthesis_after: {tuple(x.shape)}")
        x = rearrange(
            x,
            "b (c ri) t f -> b c f t ri",
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

    def pad_tensor(self, x: Tensor) -> Tensor:
        pad_t = -x.shape[2] % self.patch_size[0]
        pad_f = -x.shape[3] % self.patch_size[1]
        return F.pad(x, pad=(0, pad_f, 0, pad_t))

    def pad_audio(self, audio: Tensor, multiple_l: int) -> Tensor:
        pad_l = -audio.shape[-1] % multiple_l
        return F.pad(audio, pad=(0, pad_l))

    def trim_or_pad_output(self, out: Tensor, audio_length: int) -> Tensor:
        out = out[..., :audio_length]
        if out.shape[-1] < audio_length:
            out = F.pad(out, pad=(0, audio_length - out.shape[-1]))
        return out


if __name__ == "__main__":
    model = BSRoformer89c4()
    audio = torch.randn(1, 2, 48000)
    out = model(audio)
    print(out.shape)
