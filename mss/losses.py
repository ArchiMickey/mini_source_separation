from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


class MultiResolutionSTFTLoss(nn.Module):
    r"""Multi-resolution STFT loss."""

    def __init__(
        self,
        window_sizes: List[int] = [4096, 2048, 1024, 512, 256],
        hop_size=147,
        stft_n_fft=2048,
        normalized=False,
        window_fn=torch.hann_window,
    ) -> None:
        super(MultiResolutionSTFTLoss, self).__init__()

        self.window_sizes = window_sizes
        self.hop_size = hop_size
        self.stft_n_fft = stft_n_fft
        self.normalized = normalized
        self.window_fn = window_fn
  
        self.multi_stft_kwargs = dict(
            hop_length = hop_size,
            normalized = normalized,
        )

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss = 0.
        
        for window_size in self.window_sizes:
            stft_kwargs = dict(
                n_fft = max(window_size, self.stft_n_fft),
                win_length = window_size,
                return_complex = True,
                window = self.window_fn(window_size).to(output.device),
                **self.multi_stft_kwargs
            )
            
            output_Y = torch.stft(rearrange(output, "... s t -> (... s) t"), **stft_kwargs)
            target_Y = torch.stft(rearrange(target, "... s t -> (... s) t"), **stft_kwargs)
            
            loss = loss + F.l1_loss(output_Y, target_Y)
        
        return loss

class MultiLoss(nn.Module):
    r"""Combine multiple loss functions with weights."""

    def __init__(self, losses: List[tuple[str, nn.Module, float]]) -> None:
        super(MultiLoss, self).__init__()
        self.losses = losses

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        total_loss = 0.
        loss_info = {}
        for loss_name, loss_fn, weight in self.losses:
            total_loss = total_loss + weight * loss_fn(output, target)
            loss_info[loss_name] = total_loss.item()
        return total_loss, loss_info

def get_loss_fn(configs):
    loss = configs["train"]["loss"]
    if isinstance(loss, str):
        loss = [loss, 1.0]
    
    losses = []
    for loss_name, weight in loss:
        if loss_name == "l1":
            losses.append((loss_name, torch.nn.L1Loss(), weight))
        elif loss_name == "mse":
            losses.append((loss_name, torch.nn.MSELoss(), weight))
        elif loss_name == "mrstft":
            losses.append((loss_name, MultiResolutionSTFTLoss(hop_size=configs["model"]["hop_length"]), weight))
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")
    return MultiLoss(losses)