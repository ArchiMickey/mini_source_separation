from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mss.models.bsroformer91_cnn import BSRoformer91Cnn


class BSRoformer91CnnOverlap(BSRoformer91Cnn):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        overlap_kernel_t = self.hop_length * 2

        self.cnn_encoder = nn.Conv2d(
            self.audio_channels * 2,
            self.dim,
            kernel_size=(overlap_kernel_t, 1),
            stride=(self.hop_length, 1),
            bias=False,
        )
        self.cnn_decoder = nn.ConvTranspose2d(
            self.dim,
            self.audio_channels * 2,
            kernel_size=(overlap_kernel_t, 1),
            stride=(self.hop_length, 1),
            bias=False,
        )

    def _pad_for_cnn_analysis(self, x: Tensor) -> Tensor:
        length = x.shape[-2]
        kernel_size = self.cnn_encoder.kernel_size[0]
        stride = self.cnn_encoder.stride[0]
        frames = (length + stride - 1) // stride
        target_length = (frames - 1) * stride + kernel_size
        pad_right = target_length - length
        if pad_right > 0:
            x = F.pad(x, pad=(0, 0, 0, pad_right))
        return x
