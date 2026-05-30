from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models.bsroformer56b import BSRoformerBlock, GEGLUFusion
from mss.models.rope import RoPE
from mss.models2.dsp3.banks import mel_linear_banks_triangle
from mss.models2.dsp3.subband_fast_triangle import SubbandFilter


class BSRoformer89c4(nn.Module):
    """56b RoFormer body with the 115d triangle subband frontend."""

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
        analysis_type: str = "cnn",
        cnn_mask_mode: str = "feature",
        n_fft: int = 32,
        stft_hop_length: int = 8,
        bank_type: str = "mel_linear_triangle",
        subband_factor: int | None = None,
        max_bandwidth: int = 390,
        max_half_bandwidth: int | None = None,
        chunk_size: int = 16,
        bandpass_filter_len: int = 48000,
        upsample_filter_len: int = 12000,
        debug_print_shapes: bool = False,
        use_fusion: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # Kept only for config compatibility with 56b YAMLs.
        kwargs.pop("band_dim", None)
        del kwargs

        self.audio_channels = audio_channels
        self.patch_size = tuple(int(v) for v in patch_size)
        self.subband_n_bands = int(subband_n_bands or n_bands or 118)
        self.hop_length = int(hop_length)
        self.analysis_type = self._normalize_analysis_type(analysis_type)
        self.cnn_mask_mode = self._normalize_cnn_mask_mode(cnn_mask_mode)
        self.n_fft = int(n_fft)
        self.stft_hop_length = int(stft_hop_length)
        self.bank_type = bank_type
        self.sb_factor = int(subband_factor or max(sample_rate // 800, 1))
        self._last_cnn_input_length: int | None = None
        self._last_stft_input_length: int | None = None
        self.debug_print_shapes = debug_print_shapes
        self.use_fusion = bool(use_fusion)

        banks = self._build_banks(
            sample_rate,
            self.subband_n_bands,
            int(max_half_bandwidth or max_bandwidth),
        )
        self.sb_filter = SubbandFilter(
            sample_rate,
            banks,
            self.sb_factor,
            chunk_size=chunk_size,
            bandpass_filter_len=bandpass_filter_len,
            upsample_filter_len=upsample_filter_len,
        )

        self.rope = RoPE(head_dim=dim_head, max_len=rope_len)
        self.feature_dim = dim_sp if self.analysis_type == "cnn" else audio_channels * self.n_fft * 2
        if self.feature_dim % dim_head != 0:
            raise ValueError(f"feature_dim={self.feature_dim} must be divisible by dim_head={dim_head}")

        if self.analysis_type == "cnn":
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
        else:
            self.cnn_encoder = None
            self.cnn_decoder = None

        self.down = nn.Conv2d(self.feature_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.up = nn.ConvTranspose2d(dim, self.feature_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.fusion = GEGLUFusion(self.feature_dim) if self.use_fusion else None

        self.pre_blocks = nn.ModuleList(
            [
                BSRoformerBlock(
                    self.feature_dim,
                    self.feature_dim // dim_head,
                    axis="tf",
                    use_time_mix=True,
                )
                for _ in range(n_pre_layers)
            ]
        )
        self.post_blocks = nn.ModuleList(
            [
                BSRoformerBlock(
                    self.feature_dim,
                    self.feature_dim // dim_head,
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

    def _normalize_analysis_type(self, analysis_type: str) -> str:
        analysis_type = analysis_type.lower()
        if analysis_type not in {"cnn", "stft"}:
            raise ValueError(f"Unsupported analysis_type: {analysis_type}")
        return analysis_type

    def _normalize_cnn_mask_mode(self, cnn_mask_mode: str) -> str:
        aliases = {
            "feature": "feature",
            "features": "feature",
            "cnn": "feature",
            "cnn_feature": "feature",
            "cnn_features": "feature",
            "subband": "subband",
            "subbands": "subband",
            "subband_analysis": "subband",
        }
        try:
            return aliases[cnn_mask_mode.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported cnn_mask_mode: {cnn_mask_mode}") from exc

    def _build_banks(
        self,
        sample_rate: int,
        n_bands: int,
        max_half_bandwidth: int,
    ) -> list[tuple[float, float]]:
        if self.bank_type in {"mel_linear_triangle", "mel_triangle", "115d"}:
            return mel_linear_banks_triangle(sample_rate, n_bands, max_half_bandwidth)
        raise ValueError(f"Unsupported bank_type: {self.bank_type}")

    def forward(self, audio: Tensor) -> Tensor:
        audio_length = audio.shape[-1]
        audio = self.pad_audio(audio, self.sb_factor)
        subbands = self.sb_filter.analysis(audio)
        features = self.analysis(subbands)
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
        features = features[:, :, :frames_num, :bands_num]
        x = self.apply_mask_and_synthesize(subbands, features, mask)
        out = self.sb_filter.synthesis(x)
        return self.trim_or_pad_output(out, audio_length)

    def analysis(self, x: Tensor) -> Tensor:
        if self.analysis_type == "cnn":
            return self.cnn_analysis(x)
        if self.analysis_type == "stft":
            return self.stft_analysis(x)
        raise RuntimeError(f"Unhandled analysis_type: {self.analysis_type}")

    def apply_mask_and_synthesize(self, subbands: Tensor, features: Tensor, mask: Tensor) -> Tensor:
        if self.analysis_type == "stft":
            complex_sp = self.stft_channels_to_complex(features)
            complex_mask = self.stft_channels_to_complex(mask)
            return self.istft(complex_sp * complex_mask)

        if self.cnn_mask_mode == "feature":
            return self.cnn_synthesis(features * mask)
        if self.cnn_mask_mode == "subband":
            return subbands * self.cnn_synthesis(mask)
        raise RuntimeError(f"Unhandled cnn_mask_mode: {self.cnn_mask_mode}")

    def stft_analysis(self, x: Tensor) -> Tensor:
        self._last_stft_input_length = x.shape[-1]
        complex_sp = self.stft(x)
        return self.complex_stft_to_channels(complex_sp)

    def complex_stft_to_channels(self, x: Tensor) -> Tensor:
        return rearrange(torch.view_as_real(x), "b c k t f ri -> b (c f ri) t k")

    def stft_channels_to_complex(self, x: Tensor) -> Tensor:
        x = rearrange(
            x,
            "b (c f ri) t k -> b c k t f ri",
            c=self.audio_channels,
            f=self.n_fft,
            ri=2,
        )
        return torch.view_as_complex(x.contiguous().float())

    def stft(self, x: Tensor) -> Tensor:
        B, C = x.shape[0:2]
        x = rearrange(x, "b c k l -> (b c k) l")
        x = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.stft_hop_length,
            window=torch.hann_window(self.n_fft, device=x.device),
            normalized=True,
            onesided=False,
            return_complex=True,
        )
        return rearrange(x, "(b c k) f t -> b c k t f", b=B, c=C)

    def istft(self, x: Tensor) -> Tensor:
        B, C = x.shape[0:2]
        x = rearrange(x, "b c k t f -> (b c k) f t")
        x = torch.istft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.stft_hop_length,
            window=torch.hann_window(self.n_fft, device=x.device),
            normalized=True,
            onesided=False,
            return_complex=True,
            length=self._last_stft_input_length,
        )
        return rearrange(x, "(b c k) l -> b c k l", b=B, c=C)

    def cnn_analysis(self, x: Tensor) -> Tensor:
        if self.cnn_encoder is None:
            raise RuntimeError("cnn_analysis() is only available when analysis_type='cnn'")
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
        if self.cnn_decoder is None:
            raise RuntimeError("cnn_synthesis() is only available when analysis_type='cnn'")
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
        if self.cnn_encoder is None:
            raise RuntimeError("_pad_for_cnn_analysis() is only available when analysis_type='cnn'")
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
