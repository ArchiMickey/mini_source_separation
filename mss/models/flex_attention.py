from __future__ import annotations

from functools import lru_cache
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, Tensor

from torch.nn.attention.flex_attention import (
    flex_attention as _flex_attention,
    BlockMask,
)
torch._dynamo.config.cache_size_limit = 5000
flex_attention_compiled = torch.compile(_flex_attention, dynamic=False)
from mss.models.rope import RoPE


class Block(nn.Module):
    r"""Self attention block.

    Ref: 
        [1] https://github.com/facebookresearch/DiT/blob/main/models.py
        [2] https://huggingface.co/hpcai-tech/OpenSora-STDiT-v1-HQ-16x256x256/blob/main/layers.py
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.attn = SelfAttention(dim, num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), 
            nn.GELU(approximate='tanh'),
            nn.Linear(dim * 4, dim)
        )

    def forward(
        self,
        x: Tensor,
        rope: RoPE,
        pos: LongTensor | None = None,
        attn_mask: Tensor | BlockMask | None = None,
    ) -> Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            rope: (t, head_dim/2, 2)

        Outputs:
            out: (b, l, d)
        """

        x = x + self.attn(self.norm1(x), rope, pos, attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))

        return x

class BlockV2(nn.Module):
    """Self attention block with SwiGLU"""
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.attn = SelfAttention(dim, num_heads)
        
        self.ffn = GatedMLP(dim, intermediate_dim=dim * 4)

    def forward(
        self,
        x: Tensor,
        rope: RoPE,
        pos: LongTensor | None = None,
        attn_mask: Tensor | BlockMask | None = None,
    ) -> Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            rope: (t, head_dim/2, 2)

        Outputs:
            out: (b, l, d)
        """

        x = x + self.attn(self.norm1(x), rope, pos, attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))

        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class DecoderBlock(BlockV2):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__(dim, num_heads)
        
        self.mod = nn.Linear(dim, dim * 6)
    
    def forward(
        self,
        x: Tensor,
        c: Tensor,
        rope: RoPE,
        pos: LongTensor | None = None,
        attn_mask: Tensor | BlockMask | None = None,
    ) -> Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            rope: (t, head_dim/2, 2)

        Outputs:
            out: (b, l, d)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.mod(c).chunk(6, dim=-1)

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope, pos, attn_mask=attn_mask)
        x = x + gate_mlp * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

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
        dtype = x.dtype
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.scale
        return output.to(dtype)


def generate_mask_mod(window_size: tuple[int, int]):
    """
    Generates a mask_mod for sliding window attention:
      - Each query position q_idx can attend to keys at kv_idx
        if (q_idx - kv_idx) is between -lookahead and +lookback inclusive.
    Args:
        lookback: how many positions to look backwards (kv_idx < q_idx).
        lookahead: how many positions ahead (kv_idx > q_idx) are allowed.
    Returns:
        function mask_mod(b, h, q_idx, kv_idx) -> bool
    """
    def mask_mod(b, h, q_idx, kv_idx):
        # compute delta
        delta = q_idx - kv_idx
        window_mask = (delta >= -window_size[1]) & (delta <= window_size[0])
        return window_mask

    mask_mod.__name__ = f"sliding_window_{window_size[0]}_{window_size[1]}"
    return mask_mod

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads) -> None:
        super().__init__()

        assert dim % num_heads == 0
        self.head_dim = dim // num_heads

        self.qkv_linear = nn.Linear(dim, 3 * dim)
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)

        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        rope: RoPE,
        pos: LongTensor | None = None,
        attn_mask: BlockMask | None = None,
    ) -> Tensor:
        r"""Causal self attention.

        b: batch_size
        l: seq_len
        d: latent_dim
        n: n_head
        h: head_dim

        Args:
            x: (b, l, d)
            rope: (l, head_dim/2, 2)
            mask: (1, 1)

        Outputs:
            x: (b, l, d)
        """

        # Calculate query, key, values
        q, k, v = self.qkv_linear(x).chunk(chunks=3, dim=-1)  # shapes: (b, l, d)
        q = rearrange(self.norm_q(q), 'b l (n h) -> b n l h', h=self.head_dim)  # (b, n, l, h)
        k = rearrange(self.norm_k(k), 'b l (n h) -> b n l h', h=self.head_dim)  # (b, n, l, h)
        v = rearrange(v, 'b l (n h) -> b n l h', h=self.head_dim)  # (b, n, l, h)

        # Apply RoPE
        if pos is None:
            q = rope(q)  # (b, n, l, h)
            k = rope(k)  # (b, n, l, h)
        else:
            q = rope.apply_nd(q, pos)  # (b, n, l, h)
            k = rope.apply_nd(k, pos)  # (b, n, l, h)

        x = flex_attention_compiled(
            query=q,
            key=k,
            value=v,
            block_mask=attn_mask,
        )

        x = rearrange(x, 'b n l h -> b l (n h)')
        x = self.proj(x)  # (b, l, d)

        return x

class GatedMLP(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        
        self.gate_proj = nn.Linear(dim, intermediate_dim)
        self.up_proj = nn.Linear(dim, intermediate_dim)
        self.down_proj = nn.Linear(intermediate_dim, dim)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        r"""Gated MLP.

        Args:
            x: (b, l, d)

        Outputs:
            out: (b, l, d)
        """

        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))