from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, Tensor

from mss.models.rope import RoPE

# Optional flash_attn import
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class AbsolutePositionEmbedding(nn.Module):
    r"""Absolute Position Embedding.
    
    This module implements learnable absolute position embeddings that are added
    to the query and key projections in the attention mechanism. Unlike RoPE which
    applies rotations, absolute embeddings directly add positional information.
    
    Absolute position embeddings are simpler and can be more stable for certain tasks,
    especially when:
    - Sequence lengths are relatively fixed during training and inference
    - The model needs to learn specific positional patterns
    - Computational efficiency is critical (no rotation operations needed)
    
    Args:
        max_seq_len: Maximum sequence length to support
        dim: Embedding dimension (should match the model's hidden dimension)
        
    Shape:
        - Input: (batch_size, seq_len, num_heads, head_dim)
        - Output: (batch_size, seq_len, num_heads, head_dim)
    """
    
    def __init__(self, max_seq_len: int, dim: int) -> None:
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.dim = dim
        
        # Learnable position embeddings
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, dim) * 0.02)
        
    def forward(self, x: Tensor) -> Tensor:
        r"""Apply absolute position embeddings.
        
        Args:
            x: (b, l, n, h) - Input tensor
            
        Returns:
            out: (b, l, n, h) - Tensor with position embeddings added
        """
        B, L, N, H = x.shape
        
        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds maximum supported length {self.max_seq_len}. "
                f"Please increase max_seq_len in the configuration."
            )
        
        # Get position embeddings for current sequence length
        pos_emb = self.pos_emb[:L]  # (l, d)
        
        # Reshape to match input dimensions: (l, d) -> (1, l, 1, d)
        pos_emb = pos_emb.view(1, L, 1, -1)
        
        # Add position embeddings to input
        out = x + pos_emb
        
        return out
    
    def apply_nd(self, x: Tensor, pos: LongTensor) -> Tensor:
        r"""Apply N-dimensional absolute position embeddings with sparse positions.
        
        This method supports multi-dimensional positional encoding (e.g., for images or videos)
        by indexing into the position embedding table using provided position indices.
        
        Args:
            x: (b, l, n, h) - Input tensor
            pos: (l, k) - Position indices for each dimension
            
        Returns:
            out: (b, l, n, h) - Tensor with position embeddings added
        """
        B, L, N, H = x.shape
        K = pos.shape[1]  # Number of dimensions
        
        # For N-D positions, we sum embeddings from each dimension
        # Each dimension gets H // K features
        dim_per_axis = self.dim // K
        
        pos_emb_sum = torch.zeros(L, self.dim, device=x.device, dtype=x.dtype)
        
        for i in range(K):
            p = pos[:, i]  # (l,)
            
            # Check bounds
            if torch.any(p >= self.max_seq_len):
                raise ValueError(
                    f"Position index {p.max().item()} exceeds maximum supported length {self.max_seq_len}"
                )
            
            # Get embeddings for this dimension
            start_idx = i * dim_per_axis
            end_idx = start_idx + dim_per_axis if i < K - 1 else self.dim
            pos_emb_sum[:, start_idx:end_idx] = self.pos_emb[p, start_idx:end_idx]
        
        # Reshape and add to input
        pos_emb_sum = pos_emb_sum.view(1, L, 1, -1)
        out = x + pos_emb_sum
        
        return out


class Block(nn.Module):
    r"""Self attention block.

    Ref: 
        [1] https://github.com/facebookresearch/DiT/blob/main/models.py
        [2] https://huggingface.co/hpcai-tech/OpenSora-STDiT-v1-HQ-16x256x256/blob/main/layers.py
    """

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
            nn.GELU(approximate='tanh'),
            nn.Linear(dim * 4, dim)
        )

    def forward(
        self,
        x: Tensor,
        rope: RoPE | None = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            rope: (t, head_dim/2, 2) - Only used if position_embedding_type="rope"
            pos: (l, k) - Position indices for N-D embeddings (optional)

        Outputs:
            out: (b, l, d)
        """

        x = x + self.attn(self.norm1(x), rope, pos)
        x = x + self.ffn(self.norm2(x))

        return x

class BlockV2(nn.Module):
    """Self attention block with SwiGLU"""
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
        
        self.ffn = GatedMLP(dim, intermediate_dim=dim * 4)

    def forward(
        self,
        x: Tensor,
        rope: RoPE | None = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            rope: (t, head_dim/2, 2) - Only used if position_embedding_type="rope"
            pos: (l, k) - Position indices for N-D embeddings (optional)

        Outputs:
            out: (b, l, d)
        """

        x = x + self.attn(self.norm1(x), rope, pos)
        x = x + self.ffn(self.norm2(x))

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
        
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.scale
        return output


class SelfAttention(nn.Module):
    r"""Self-Attention module with configurable position embedding and attention backend.
    
    This module supports two types of position embeddings:
    
    1. RoPE (Rotary Position Embedding) - Default
       - Applies rotations to query and key vectors
       - Better extrapolation to longer sequences
       - No learnable parameters for position encoding
       - Recommended for variable-length sequences
       
    2. Absolute Position Embedding
       - Adds learnable position embeddings to query and key
       - Simpler and more stable for fixed-length sequences
       - Requires specifying max_seq_len
       - Can be more efficient computationally
    
    This module also supports two attention backends:
    
    1. torch - Uses PyTorch's scaled_dot_product_attention
       - Supports windowed attention via attention mask
       
    2. flash_attn - Uses flash_attn library for efficient attention
       - Supports windowed attention via window_size argument
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        position_embedding_type: Type of position embedding ("rope" or "absolute")
        max_seq_len: Maximum sequence length (required for absolute embeddings)
        attn_backend: Attention backend ("torch" or "flash_attn")
        window_size: Window size for local attention (None for global attention)
    """
    
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
        
        # Validate attention backend
        valid_backends = ["torch", "flash_attn"]
        if attn_backend not in valid_backends:
            raise ValueError(
                f"Invalid attn_backend: '{attn_backend}'. "
                f"Must be one of {valid_backends}"
            )
        
        # Check flash_attn availability
        if attn_backend == "flash_attn" and not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "flash_attn backend requested but flash_attn is not installed. "
                "Please install it with: pip install flash-attn --no-build-isolation"
            )
        
        self.attn_backend = attn_backend
        
        # Validate position embedding type
        valid_types = ["rope", "absolute"]
        if position_embedding_type not in valid_types:
            raise ValueError(
                f"Invalid position_embedding_type: '{position_embedding_type}'. "
                f"Must be one of {valid_types}"
            )
        
        self.position_embedding_type = position_embedding_type
        
        # Initialize position embedding module if using absolute embeddings
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
        r"""Create attention mask for windowed attention.
        
        Args:
            seq_len: Sequence length
            device: Device to create the mask on
            
        Returns:
            mask: (seq_len, seq_len) - Boolean mask where True indicates positions to attend to
        """
        if self.window_size is None:
            return None
        
        # Create a mask where each position can attend to positions within the window
        # Window is centered on each position
        positions = torch.arange(seq_len, device=device)
        
        # Calculate distance matrix
        distance = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
        
        # Create mask: True for positions within window, False otherwise
        mask = distance <= self.window_size // 2
        
        return mask

    def forward(
        self,
        x: Tensor,
        rope: nn.Module | None = None,
        pos: LongTensor | None = None,
    ) -> Tensor:
        r"""Self attention with configurable position embedding and backend.

        b: batch_size
        l: seq_len
        d: latent_dim
        n: n_head
        h: head_dim

        Args:
            x: (b, l, d) - Input tensor
            rope: RoPE module - Only used if position_embedding_type="rope"
            pos: (l, k) - Position indices for N-D embeddings (optional)

        Outputs:
            x: (b, l, d) - Output tensor
        """

        # Calculate query, key, values
        q, k, v = self.qkv_linear(x).chunk(chunks=3, dim=2)  # shapes: (b, l, d)
        q = rearrange(self.norm_q(q), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        k = rearrange(self.norm_k(k), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        v = rearrange(v, 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)

        # Apply position embeddings based on type
        if self.position_embedding_type == "rope":
            # Use RoPE (Rotary Position Embedding)
            if rope is None:
                raise ValueError(
                    "RoPE module must be provided when position_embedding_type='rope'"
                )
            
            if pos is None:
                q = rope(q)  # (b, l, n, h)
                k = rope(k)  # (b, l, n, h)
            else:
                q = rope.apply_nd(q, pos)  # (b, l, n, h)
                k = rope.apply_nd(k, pos)  # (b, l, n, h)
                
        elif self.position_embedding_type == "absolute":
            # Use Absolute Position Embedding
            if pos is None:
                q = self.pos_embedding(q)  # (b, l, n, h)
                k = self.pos_embedding(k)  # (b, l, n, h)
            else:
                q = self.pos_embedding.apply_nd(q, pos)  # (b, l, n, h)
                k = self.pos_embedding.apply_nd(k, pos)  # (b, l, n, h)

        # Compute attention based on backend
        if self.attn_backend == "flash_attn":
            # Use flash_attn for efficient attention
            # flash_attn_func expects (b, l, n, h) layout and returns (b, l, n, h)
            if self.window_size is not None:
                # flash_attn uses (window_left, window_right) for sliding window attention
                # We use symmetric window: (window_size // 2, window_size // 2)
                window_size_half = self.window_size // 2
                window_size_tuple = (window_size_half, window_size_half)
            else:
                window_size_tuple = (-1, -1)  # (-1, -1) means global attention
            
            x = flash_attn_func(
                q, k, v,
                window_size=window_size_tuple,
            )  # (b, l, n, h)
            
        else:
            # Use PyTorch's scaled_dot_product_attention
            # Create attention mask for windowed attention if needed
            attn_mask = None
            if self.window_size is not None:
                attn_mask = self._create_window_mask(x.shape[1], x.device)
                # scaled_dot_product_attention expects mask where True = masked out
                # So we need to invert our mask
                attn_mask = ~attn_mask
            
            x = F.scaled_dot_product_attention(
                query=rearrange(q, 'b l n h -> b n l h'), 
                key=rearrange(k, 'b l n h -> b n l h'), 
                value=rearrange(v, 'b l n h -> b n l h'), 
                attn_mask=attn_mask, 
                dropout_p=0.0
            )  # (b, n, l, h)
            x = rearrange(x, 'b n l h -> b l n h')

        x = rearrange(x, 'b l n h -> b l (n h)')
        x = self.proj(x)  # (b, l, d)
        
        return x
    
class GatedMLP(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        
        intermediate_dim = intermediate_dim * 2 // 3
        
        self.gate_proj = nn.Linear(dim, intermediate_dim)
        self.up_proj = nn.Linear(dim, intermediate_dim)
        self.down_proj = nn.Linear(intermediate_dim, dim)
        self.act_fn = nn.GELU(approximate='tanh')
        
    
    def forward(self, x: Tensor) -> Tensor:
        r"""Gated MLP.

        Args:
            x: (b, l, d)

        Outputs:
            out: (b, l, d)
        """

        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
