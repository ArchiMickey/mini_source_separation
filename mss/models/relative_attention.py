from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, Tensor

from mss.models.alibi import ALiBi


class ALiBiEnhanced(nn.Module):
    """Enhanced Attention with Linear Biases (ALiBi).
    
    This enhanced implementation provides:
    - Linear bias computation based on relative distances
    - Geometric sequence of slopes for different attention heads
    - Support for both causal and bidirectional attention masks
    - Efficient matrix operations for adding biases to attention scores
    - Proper handling of batch dimensions and multiple attention heads
    - Flash attention compatibility
    
    Ref:
        Press, Ofir, Noah A. Smith, and Mike Lewis. "Train short, test long: 
        Attention with linear biases enables input length extrapolation." 
        arXiv preprint arXiv:2108.12409 (2021).
    """
    
    def __init__(
        self, 
        num_heads: int,
        causal: bool = False,
        symmetric: bool = True
    ) -> None:
        """Initialize Enhanced ALiBi.
        
        Args:
            num_heads: Number of attention heads
            causal: If True, use causal masking (only attend to past positions)
            symmetric: If True, use symmetric distances |i-j|, else use signed distances
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.causal = causal
        self.symmetric = symmetric
        
        # Calculate slopes for each head using geometric sequence
        slopes = self._get_slopes(num_heads)
        
        # Register slopes as buffer (num_heads,)
        self.register_buffer("slopes", slopes)
        
    def _get_slopes(self, num_heads: int) -> Tensor:
        """Calculate slopes for each attention head using geometric sequence.
        
        For n heads, slopes follow: 2^(-8/n * i) for i in [1, n]
        This creates a geometric sequence that provides different decay rates
        for different heads.
        
        Args:
            num_heads: Number of attention heads
            
        Returns:
            slopes: (num_heads,) tensor of slope values
        """
        def get_slopes_power_of_2(n: int) -> Tensor:
            """Get slopes when n is a power of 2."""
            start = 2 ** (-2 ** -(torch.arange(n, dtype=torch.float32) / n * 8))
            return start
        
        # Check if num_heads is a power of 2
        if math.log2(num_heads) % 1 == 0:
            return get_slopes_power_of_2(num_heads)
        else:
            # For non-power-of-2, use interpolation strategy
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes_a = get_slopes_power_of_2(int(closest_power_of_2))
            
            # Get additional slopes by taking every other slope from next power of 2
            slopes_b = self._get_slopes(int(2 * closest_power_of_2))[0::2]
            slopes_b = slopes_b[:num_heads - int(closest_power_of_2)]
            
            return torch.cat([slopes_a, slopes_b])
    
    def _create_position_bias(
        self, 
        query_len: int, 
        key_len: int, 
        device: torch.device
    ) -> Tensor:
        """Create position bias matrix on-the-fly.
        
        Args:
            query_len: Query sequence length
            key_len: Key sequence length
            device: Device to create tensor on
            
        Returns:
            position_bias: (query_len, key_len) tensor of relative position biases
        """
        # Create position indices
        context_position = torch.arange(query_len, device=device, dtype=torch.float32)[:, None]
        memory_position = torch.arange(key_len, device=device, dtype=torch.float32)[None, :]
        
        # Compute relative positions: memory_position - context_position
        relative_position = memory_position - context_position  # (query_len, key_len)
        
        if self.symmetric:
            # Use absolute distances: -|i - j|
            position_bias = -torch.abs(relative_position)
        else:
            # Use signed distances: -(i - j) for causal, or raw distances
            if self.causal:
                position_bias = -torch.clamp(relative_position, min=0)
            else:
                position_bias = -torch.abs(relative_position)
        
        return position_bias
    
    def forward(
        self, 
        query_len: int, 
        key_len: Optional[int] = None,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Get ALiBi bias for the given sequence lengths.
        
        Computes bias on-the-fly to save memory.
        
        Args:
            query_len: Query sequence length
            key_len: Key sequence length (defaults to query_len if None)
            attn_mask: Optional attention mask to combine with ALiBi bias
            
        Returns:
            bias: (num_heads, query_len, key_len) tensor of attention biases
        """
        if key_len is None:
            key_len = query_len
        
        # Create position bias on-the-fly
        position_bias = self._create_position_bias(
            query_len, key_len, self.slopes.device
        )  # (query_len, key_len)
        
        # Apply slopes: (num_heads, 1, 1) * (1, query_len, key_len)
        alibi_bias = self.slopes[:, None, None] * position_bias[None, :, :]
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(query_len, key_len, device=self.slopes.device, dtype=torch.bool),
                diagonal=key_len - query_len + 1
            )
            alibi_bias = alibi_bias.masked_fill(causal_mask, float('-inf'))
        
        # Combine with additional attention mask if provided
        if attn_mask is not None:
            alibi_bias = alibi_bias + attn_mask
        
        return alibi_bias
    
    def get_bias(
        self, 
        query_len: int, 
        key_len: int,
        batch_size: Optional[int] = None
    ) -> Tensor:
        """Get ALiBi bias with optional batch dimension.
        
        Args:
            query_len: Query sequence length
            key_len: Key sequence length
            batch_size: Optional batch size to expand bias
            
        Returns:
            bias: (num_heads, query_len, key_len) or 
                  (batch_size, num_heads, query_len, key_len)
        """
        bias = self.forward(query_len, key_len)
        
        if batch_size is not None:
            bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return bias


class RelativePositionalBias(nn.Module):
    """Learnable Relative Positional Bias for attention mechanisms.
    
    This implementation provides:
    - Learnable relative position embeddings with configurable maximum distance
    - Clipping mechanism for positions beyond maximum distance
    - Separate bias parameters for each head or shared across heads
    - Efficient indexing to retrieve bias values based on query-key pairs
    - Support for 1D sequence positions and optional 2D spatial positions
    - Bucketing strategy for relative positions to reduce parameter count
    - Proper initialization of bias parameters
    
    Ref:
        Raffel, Colin, et al. "Exploring the limits of transfer learning with a 
        unified text-to-text transformer." JMLR 21.1 (2020): 5485-5551.
    """
    
    def __init__(
        self,
        num_heads: int,
        max_distance: int = 128,
        num_buckets: Optional[int] = None,
        bidirectional: bool = True,
        shared_heads: bool = False,
        causal: bool = False
    ) -> None:
        """Initialize Relative Positional Bias.
        
        Args:
            num_heads: Number of attention heads
            max_distance: Maximum relative distance to consider
            num_buckets: Number of buckets for bucketing strategy (None = no bucketing)
            bidirectional: If True, use bidirectional relative positions
            shared_heads: If True, share bias parameters across all heads
            causal: If True, use causal masking
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.num_buckets = num_buckets if num_buckets is not None else max_distance * 2 + 1
        self.bidirectional = bidirectional
        self.shared_heads = shared_heads
        self.causal = causal
        
        # Adjust num_buckets for bidirectional case
        if not bidirectional:
            self.num_buckets = self.num_buckets // 2
        
        # Create learnable bias parameters
        # Shape: (num_heads, num_buckets) or (1, num_buckets) if shared
        num_bias_heads = 1 if shared_heads else num_heads
        self.relative_attention_bias = nn.Parameter(
            torch.zeros(num_bias_heads, self.num_buckets)
        )
        
        # Initialize bias parameters
        self._init_bias()
        
    def _init_bias(self) -> None:
        """Initialize bias parameters with small random values."""
        nn.init.normal_(self.relative_attention_bias, mean=0.0, std=0.02)
    
    def _relative_position_bucket(
        self,
        relative_position: Tensor,
        num_buckets: int,
        max_distance: int
    ) -> Tensor:
        """Map relative positions to buckets.
        
        Uses a bucketing strategy that allocates more buckets to smaller distances
        and fewer buckets to larger distances (logarithmic bucketing).
        
        Args:
            relative_position: (query_len, key_len) tensor of relative positions
            num_buckets: Number of buckets
            max_distance: Maximum distance
            
        Returns:
            buckets: (query_len, key_len) tensor of bucket indices
        """
        relative_buckets = torch.zeros_like(relative_position)
        
        if self.bidirectional:
            num_buckets = num_buckets // 2
            # Positive distances get buckets [num_buckets, 2*num_buckets)
            relative_buckets = relative_buckets + (relative_position > 0).long() * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # For causal/unidirectional, only consider non-positive distances
            relative_position = -torch.clamp(relative_position, max=0)
        
        # Now relative_position is in range [0, inf)
        # Half of buckets for small distances (linear)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # Other half for large distances (logarithmic)
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / 
            math.log(max_distance / max_exact) * 
            (num_buckets - max_exact)
        ).long()
        relative_position_if_large = torch.clamp(
            relative_position_if_large, 
            max=num_buckets - 1
        )
        
        relative_buckets = relative_buckets + torch.where(
            is_small,
            relative_position,
            relative_position_if_large
        )
        
        return relative_buckets
    
    def _compute_bias(
        self,
        query_len: int,
        key_len: int,
        device: torch.device
    ) -> Tensor:
        """Compute relative positional bias.
        
        Args:
            query_len: Query sequence length
            key_len: Key sequence length
            device: Device to create tensors on
            
        Returns:
            bias: (num_heads, query_len, key_len) tensor of biases
        """
        # Create position indices
        context_position = torch.arange(query_len, device=device, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_len, device=device, dtype=torch.long)[None, :]
        
        # Compute relative positions
        relative_position = memory_position - context_position  # (query_len, key_len)
        
        # Map to buckets
        relative_buckets = self._relative_position_bucket(
            relative_position,
            self.num_buckets,
            self.max_distance
        )  # (query_len, key_len)
        
        # Retrieve bias values using advanced indexing
        # relative_attention_bias: (num_heads, num_buckets)
        # We need to gather values for each position pair
        bias = self.relative_attention_bias[:, relative_buckets]  # (num_heads, query_len, key_len)
        
        # If shared heads, expand to all heads
        if self.shared_heads:
            bias = bias.expand(self.num_heads, -1, -1)
        
        return bias
    
    def forward(
        self,
        query_len: int,
        key_len: Optional[int] = None,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Compute relative positional bias.
        
        Args:
            query_len: Query sequence length
            key_len: Key sequence length (defaults to query_len if None)
            attn_mask: Optional attention mask to combine with bias
            
        Returns:
            bias: (num_heads, query_len, key_len) tensor of attention biases
        """
        if key_len is None:
            key_len = query_len
        
        # Compute bias
        bias = self._compute_bias(
            query_len, 
            key_len, 
            self.relative_attention_bias.device
        )
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(
                    query_len, key_len, 
                    device=self.relative_attention_bias.device, 
                    dtype=torch.bool
                ),
                diagonal=key_len - query_len + 1
            )
            bias = bias.masked_fill(causal_mask, float('-inf'))
        
        # Combine with additional attention mask if provided
        if attn_mask is not None:
            bias = bias + attn_mask
        
        return bias
    
    def get_bias(
        self,
        query_len: int,
        key_len: int,
        batch_size: Optional[int] = None
    ) -> Tensor:
        """Get relative positional bias with optional batch dimension.
        
        Args:
            query_len: Query sequence length
            key_len: Key sequence length
            batch_size: Optional batch size to expand bias
            
        Returns:
            bias: (num_heads, query_len, key_len) or 
                  (batch_size, num_heads, query_len, key_len)
        """
        bias = self.forward(query_len, key_len)
        
        if batch_size is not None:
            bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return bias


class RelativePositionalBias2D(nn.Module):
    """2D Relative Positional Bias for spatial attention (e.g., images, feature maps).
    
    Extends relative positional bias to 2D spatial dimensions, useful for
    vision transformers and spatial attention mechanisms.
    """
    
    def __init__(
        self,
        num_heads: int,
        max_height: int = 32,
        max_width: int = 32,
        shared_heads: bool = False
    ) -> None:
        """Initialize 2D Relative Positional Bias.
        
        Args:
            num_heads: Number of attention heads
            max_height: Maximum height for relative positions
            max_width: Maximum width for relative positions
            shared_heads: If True, share bias parameters across all heads
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.max_height = max_height
        self.max_width = max_width
        self.shared_heads = shared_heads
        
        # Create learnable bias parameters for height and width
        num_bias_heads = 1 if shared_heads else num_heads
        
        # Separate biases for height and width dimensions
        self.relative_bias_h = nn.Parameter(
            torch.zeros(num_bias_heads, 2 * max_height - 1)
        )
        self.relative_bias_w = nn.Parameter(
            torch.zeros(num_bias_heads, 2 * max_width - 1)
        )
        
        # Initialize
        nn.init.normal_(self.relative_bias_h, mean=0.0, std=0.02)
        nn.init.normal_(self.relative_bias_w, mean=0.0, std=0.02)
    
    def forward(
        self,
        height: int,
        width: int
    ) -> Tensor:
        """Compute 2D relative positional bias.
        
        Args:
            height: Spatial height
            width: Spatial width
            
        Returns:
            bias: (num_heads, height*width, height*width) tensor
        """
        device = self.relative_bias_h.device
        
        # Create 2D position grids
        coords_h = torch.arange(height, device=device)
        coords_w = torch.arange(width, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, H, W)
        coords_flatten = coords.reshape(2, -1)  # (2, H*W)
        
        # Compute relative positions
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, H*W, H*W)
        relative_coords = relative_coords.permute(1, 2, 0)  # (H*W, H*W, 2)
        
        # Shift to start from 0
        relative_coords[:, :, 0] += self.max_height - 1
        relative_coords[:, :, 1] += self.max_width - 1
        
        # Get bias values
        bias_h = self.relative_bias_h[:, relative_coords[:, :, 0]]  # (num_heads, H*W, H*W)
        bias_w = self.relative_bias_w[:, relative_coords[:, :, 1]]  # (num_heads, H*W, H*W)
        
        # Combine height and width biases
        bias = bias_h + bias_w
        
        # If shared heads, expand to all heads
        if self.shared_heads:
            bias = bias.expand(self.num_heads, -1, -1)
        
        return bias


class Block(nn.Module):
    r"""Self attention block with ALiBi positional encoding.

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
        alibi: ALiBi,
        pos: LongTensor | None = None,
    ) -> Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            alibi: ALiBi module for positional bias
            pos: Optional position tensor (kept for API compatibility, not used with ALiBi)

        Outputs:
            out: (b, l, d)
        """

        x = x + self.attn(self.norm1(x), alibi, pos)
        x = x + self.ffn(self.norm2(x))

        return x

class BlockV2(nn.Module):
    """Self attention block with SwiGLU and ALiBi positional encoding"""
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.attn = SelfAttention(dim, num_heads)
        
        self.ffn = GatedMLP(dim, intermediate_dim=dim * 4)

    def forward(
        self,
        x: Tensor,
        alibi: ALiBi,
        pos: LongTensor | None = None,
    ) -> Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            alibi: ALiBi module for positional bias
            pos: Optional position tensor (kept for API compatibility, not used with ALiBi)

        Outputs:
            out: (b, l, d)
        """

        x = x + self.attn(self.norm1(x), alibi, pos)
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
    def __init__(self, dim, num_heads) -> None:
        super().__init__()
        
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_linear = nn.Linear(dim, 3 * dim)
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)

        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        alibi: nn.Module,
        pos: LongTensor | None = None
    ) -> Tensor:
        r"""Self attention with ALiBi positional bias.

        b: batch_size
        l: seq_len
        d: latent_dim
        n: n_head
        h: head_dim

        Args:
            x: (b, l, d)
            alibi: ALiBi module for positional bias
            pos: Optional position tensor (kept for API compatibility, not used with ALiBi)

        Outputs:
            x: (b, l, d)
        """

        B, L, D = x.shape

        # Calculate query, key, values
        q, k, v = self.qkv_linear(x).chunk(chunks=3, dim=2)  # shapes: (b, l, d)
        q = rearrange(self.norm_q(q), 'b l (n h) -> b n l h', h=self.head_dim)  # (b, n, l, h)
        k = rearrange(self.norm_k(k), 'b l (n h) -> b n l h', h=self.head_dim)  # (b, n, l, h)
        v = rearrange(v, 'b l (n h) -> b n l h', h=self.head_dim)  # (b, n, l, h)

        # Get ALiBi bias for current sequence length
        # ALiBi bias shape: (num_heads, seq_len, seq_len)
        alibi_bias = alibi(seq_len=L)  # (n, l, l)
        
        # Expand bias for batch dimension
        alibi_bias = alibi_bias.unsqueeze(0)  # (1, n, l, l)

        # Efficient attention using Flash Attention CUDA kernels with ALiBi bias
        x = F.scaled_dot_product_attention(
            query=q,  # (b, n, l, h)
            key=k,    # (b, n, l, h)
            value=v,  # (b, n, l, h)
            attn_mask=alibi_bias,  # (1, n, l, l)
            dropout_p=0.0
        )  # (b, n, l, h)

        x = rearrange(x, 'b n l h -> b l (n h)')
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


class SelfAttentionWithRelativeBias(nn.Module):
    """Self attention with configurable relative positional bias.
    
    Supports both ALiBi and learnable relative positional bias mechanisms.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias_type: str = "alibi",  # "alibi", "relative", or "none"
        max_distance: int = 128,
        causal: bool = False,
        dropout: float = 0.0
    ) -> None:
        """Initialize self attention with relative bias.
        
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            bias_type: Type of positional bias ("alibi", "relative", or "none")
            max_distance: Maximum relative distance for learnable bias
            causal: Whether to use causal masking
            dropout: Dropout probability
        """
        super().__init__()
        
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.bias_type = bias_type
        self.dropout = dropout

        self.qkv_linear = nn.Linear(dim, 3 * dim)
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)
        self.proj = nn.Linear(dim, dim)
        
        # Initialize positional bias module based on type
        if bias_type == "alibi":
            self.pos_bias = ALiBiEnhanced(num_heads, causal=causal)
        elif bias_type == "relative":
            self.pos_bias = RelativePositionalBias(
                num_heads, 
                max_distance=max_distance,
                causal=causal
            )
        else:
            self.pos_bias = None
    
    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass with relative positional bias.
        
        Args:
            x: (b, l, d) input tensor
            attn_mask: Optional attention mask
            
        Returns:
            output: (b, l, d) output tensor
        """
        B, L, D = x.shape

        # Calculate query, key, values
        q, k, v = self.qkv_linear(x).chunk(chunks=3, dim=2)
        q = rearrange(self.norm_q(q), 'b l (n h) -> b n l h', h=self.head_dim)
        k = rearrange(self.norm_k(k), 'b l (n h) -> b n l h', h=self.head_dim)
        v = rearrange(v, 'b l (n h) -> b n l h', h=self.head_dim)

        # Get positional bias if enabled
        pos_bias = None
        if self.pos_bias is not None:
            pos_bias = self.pos_bias(L, L, attn_mask)  # (n, l, l)
            pos_bias = pos_bias.unsqueeze(0)  # (1, n, l, l)

        # Attention with positional bias
        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=pos_bias,
            dropout_p=self.dropout if self.training else 0.0
        )

        x = rearrange(x, 'b n l h -> b l (n h)')
        x = self.proj(x)
        
        return x
