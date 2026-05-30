from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from torch import Tensor

from mss.models.bsroformer91 import BSRoformer91


class BSRoformer91NoMain(BSRoformer91):
    def __init__(self, *args, **kwargs) -> None:
        kwargs = dict(kwargs)
        kwargs["n_layers"] = 0
        super().__init__(*args, **kwargs)
        self.t_blocks = nn.ModuleList()
        self.k_blocks = nn.ModuleList()

    def forward(self, audio: Tensor) -> Tensor:
        audio_length = audio.shape[-1]
        audio = self.pad_audio(audio, self.sb_factor)

        x = self.sb_filter.analysis(audio)
        complex_sp = self.stft(x)

        batch_size, audio_channels, _, frames_num = complex_sp.shape[0:4]
        x = rearrange(
            torch.view_as_real(complex_sp),
            "b c k t f x -> b (c f x) t k",
        )
        x = self.pad_tensor(x, self.total_stride_t)

        skip = self.pre_block(x)
        x = self.patch(skip)
        x = self.unpatch(x)
        x = self.post_block(x, skip)
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
