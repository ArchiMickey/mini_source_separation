from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ALiBi(nn.Module):
    """Attention with Linear Biases (ALiBi).
    
    ALiBi adds position-dependent biases to attention scores instead of 
    adding positional embeddings to word embeddings.
    
    This implementation computes biases on-the-fly to minimize VRAM usage.
    
    Ref:
        Press, Ofir, Noah A. Smith, and Mike Lewis. "Train short, test long: 
        Attention with linear biases enables input length extrapolation." 
        arXiv preprint arXiv:2108.12409 (2021).
    """
    
    def __init__(self, num_heads: int):
        r"""Initialize ALiBi.
        
        Args:
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.num_heads = num_heads
        
        # Calculate slopes for each head - only store slopes, not full bias matrix
        # For n heads, slopes are: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
        slopes = self._get_slopes(num_heads)
        
        # Register slopes as buffer (num_heads,)
        self.register_buffer("slopes", slopes)
        
    def _get_slopes(self, num_heads: int) -> Tensor:
        """Calculate slopes for each attention head.
        
        Args:
            num_heads: Number of attention heads
            
        Returns:
            slopes: (num_heads,)
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(torch.arange(n).float() / n * 8))
            return start
        
        if torch.log2(torch.tensor(num_heads)).item() % 1 == 0:
            # num_heads is a power of 2
            return get_slopes_power_of_2(num_heads)
        else:
            # num_heads is not a power of 2
            # Get slopes for the closest power of 2
            closest_power_of_2 = 2 ** torch.floor(torch.log2(torch.tensor(num_heads)))
            slopes_a = get_slopes_power_of_2(int(closest_power_of_2))
            
            # Get additional slopes by interpolation
            slopes_b = self._get_slopes(int(2 * closest_power_of_2))[0::2][:num_heads - int(closest_power_of_2)]
            
            return torch.cat([slopes_a, slopes_b])
    
    def _create_position_bias(self, seq_len: int, device: torch.device) -> Tensor:
        """Create position bias matrix on-the-fly.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensor on
            
        Returns:
            position_bias: (seq_len, seq_len)
        """
        # Create a matrix where position_bias[i, j] = -(i - j)
        # This gives relative position distances
        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position = torch.arange(seq_len, device=device)[None, :]
        relative_position = memory_position - context_position  # (seq_len, seq_len)
        
        # For causal attention, we use negative distances
        # ALiBi uses -|i - j| but for causal we use -(i - j) for i >= j
        position_bias = -torch.abs(relative_position).float()
        
        return position_bias
    
    def forward(self, seq_len: int) -> Tensor:
        """Get ALiBi bias for the given sequence length.
        
        Computes bias on-the-fly to save VRAM.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            bias: (num_heads, seq_len, seq_len)
        """
        # Create position bias on-the-fly
        position_bias = self._create_position_bias(seq_len, self.slopes.device)  # (seq_len, seq_len)
        
        # Apply slopes: (num_heads, 1, 1) * (1, seq_len, seq_len) -> (num_heads, seq_len, seq_len)
        alibi_bias = self.slopes[:, None, None] * position_bias[None, :, :]
        
        return alibi_bias
    
    def get_bias(self, query_len: int, key_len: int) -> Tensor:
        """Get ALiBi bias for arbitrary query and key lengths.
        
        Computes bias on-the-fly to save VRAM.
        
        Args:
            query_len: Query sequence length
            key_len: Key sequence length
            
        Returns:
            bias: (num_heads, query_len, key_len)
        """
        # Create position bias on-the-fly for arbitrary dimensions
        context_position = torch.arange(query_len, device=self.slopes.device)[:, None]
        memory_position = torch.arange(key_len, device=self.slopes.device)[None, :]
        relative_position = memory_position - context_position
        position_bias = -torch.abs(relative_position).float()
        
        # Apply slopes
        alibi_bias = self.slopes[:, None, None] * position_bias[None, :, :]
        
        return alibi_bias


if __name__ == '__main__':
    
    print("Example 1: ALiBi for 8 heads")
    alibi = ALiBi(num_heads=8)
    bias = alibi(seq_len=10)
    print(f"Bias shape: {bias.shape}")  # (8, 10, 10)
    print(f"First head bias:\n{bias[0]}")
    
    print("\nExample 2: ALiBi for 12 heads (not power of 2)")
    alibi = ALiBi(num_heads=12)
    bias = alibi(seq_len=10)
    print(f"Bias shape: {bias.shape}")  # (12, 10, 10)
    
    print("\nExample 3: Different query and key lengths")
    alibi = ALiBi(num_heads=8)
    bias = alibi.get_bias(query_len=5, key_len=10)
    print(f"Bias shape: {bias.shape}")  # (8, 5, 10)
    
    print("\nExample 4: Memory efficiency test")
    import sys
    alibi = ALiBi(num_heads=8)
    print(f"ALiBi module size: {sys.getsizeof(alibi.slopes.storage()) / 1024:.2f} KB")
    print("Biases are computed on-the-fly, not stored!")
    
    # Visualization
    import matplotlib.pyplot as plt
    alibi = ALiBi(num_heads=8)
    bias = alibi(seq_len=50)
    
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(8):
        ax = axs[i // 4, i % 4]
        im = ax.matshow(bias[i].cpu().numpy(), cmap='RdBu_r', aspect='auto')
        ax.set_title(f'Head {i+1}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig("alibi.png")
    print("\nVisualization saved to alibi.png")
