from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch import LongTensor, Tensor


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 8192, base: int = 10000, context_len: int = 0):
        r"""Rotary position embedding.

        [1] Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary 
        position embedding." Neurocomputing, 2024

        h: head_dim
        l: seq_len
        """
        super().__init__()

        self.head_dim = head_dim
        self.context_len = context_len

        # Calculate θ = 1 / 10000**(2i/h)
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))  # (h/2,)

        # Matrix pθ
        pos_theta = torch.outer(torch.arange(max_len), theta).float()  # (l, h/2)

        # Rotation matrix
        freqs_cos = torch.cos(pos_theta)  # (l, h/2)
        freqs_sin = torch.sin(pos_theta)  # (l, h/2)
        
        w = torch.stack([freqs_cos, freqs_sin], dim=-1)  # (l, h/2, 2)
        self.register_buffer(name="w", tensor=w)

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply RoPE.

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim

        Args:
            x: (b, n, l, h)

        Outputs:
            out: (b, n, l, h)
        """
        
        if self.context_len > 0:
            r, x = x[:, :, :self.context_len, :], x[:, :, self.context_len:, :]
        
        L = x.shape[2]
        x = rearrange(x, 'b n l (h c) -> b n l h c', c=2)  # (b, n, l, h/2, 2)
        w = self.w[0 : L][None, None, :, :, :]  # (1, 1, l, h/2, 2)
        x = self.rotate(x, w)  # (b, n, l, h/2, 2)
        x = rearrange(x, 'b n l h c -> b n l (h c)')  # (b, n, l, h)
        
        if self.context_len > 0:
            x = torch.cat([r, x], dim=2)
        
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
            x: (b, n, l, h/2, 2)
            w: (1, 1, l, h/2, 2)

        Outputs:
            out: (b, n, l, h/2, 2)
        """
        dtype = x.dtype
        out = torch.stack([
            w[..., 0] * x[..., 0] - w[..., 1] * x[..., 1],
            w[..., 0] * x[..., 1] + w[..., 1] * x[..., 0]
            ],
            dim=-1,
        )  # (b, n, l, h/2, 2)

        return out.to(dtype)

    def apply_nd(self, x: Tensor, pos: LongTensor) -> Tensor:
        r"""Apply Nd RoPE with sparse positions.

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim
        k: data dim

        Args:
            x: (b, n, l, h)
            pos: (l, k)
            n_dim: int

        Outputs:
            out: (b, n, l, h)
        """
        
        B, N, L, H = x.shape
        K = pos.shape[1]  # rope_dim
        assert H == K * self.head_dim

        if self.context_len > 0:
            r, x = x[:, :, :self.context_len, :], x[:, :, self.context_len:, :]
        
        x = rearrange(x, 'b n l (k h c) -> k b n l h c', k=K, c=2)  # (k, b, n, l, h/2, 2)
        x = x.contiguous()

        # Gather the rotation matrices for all positions at once
        w = self.w[pos]  # (l, k, h/2, 2)
        w = w.permute(1, 0, 2, 3)  # (k, l, h/2, 2)
        w = w[:, None, None, :, :, :]  # (k, 1, 1, l, h/2, 2)
        
        # Apply rotation to all dimensions simultaneously
        x = self.rotate(x, w)  # (k, b, n, l, h/2, 2)

        out = rearrange(x, 'k b n l h c -> b n l (k h c)')  # (b, n, l, h)
        
        if self.context_len > 0:
            out = torch.cat([r, out], dim=2)
        
        return out


if __name__ == '__main__':

    torch.manual_seed(1234)

    B = 4  # batch_size
    N = 8  # n_head
    L = 100  # time_steps
    H = 24  # head_dim
    n_embd = N * H

    print("Example 1: RoPE (1D)")
    rope = RoPE(head_dim=H)
    x = torch.rand((B, N, L, H))  # (b, n, l, h)
    out = rope(x)  # (b, n, l, h)
    print(out.shape)

    print("Example 2: RoPE (1D) with sparse positions")
    rope = RoPE(head_dim=H)
    x = torch.rand((B, N, 4, H))  # (b, n, l, h)
    pos = torch.LongTensor([[0], [3], [7], [8]])  # (l, 1)
    out = rope.apply_nd(x, pos)  # (b, n, l, h)
    print(out.shape)

    print("Example 3: RoPE (2D image) with sparse positions")
    data_dim = 2
    rope = RoPE(head_dim=H // data_dim)
    x = torch.rand((B, N, 4, H))
    pos = torch.LongTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    out = rope.apply_nd(x, pos)  # (b, n, l, h)
    print(out.shape)

    print("Example 4: RoPE (3D video) with sparse positions")
    data_dim = 3
    rope = RoPE(head_dim=H // data_dim)
    x = torch.rand((B, N, 4, H))
    pos = torch.LongTensor([[0, 0, 0], [1, 3, 4], [2, 2, 2], [5, 4, 3]])
    out3 = rope.apply_nd(x, pos)  # (b, n, l, h)
    print(out3.shape)

    # Visualization of RoPE weights
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].matshow(rope.w[:, :, 0].data.cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(rope.w[:, :, 1].data.cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("rope.png")
    print("Write out to rope.png")