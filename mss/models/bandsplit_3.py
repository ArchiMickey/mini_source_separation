from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class LearnableBandSplit(nn.Module):
    def __init__(
        self, 
        sr: float, 
        n_fft: int, 
        n_bands: int, 
        in_channels: int, 
        out_channels: int,
        requires_grad: bool = True
    ) -> None:
        r"""
        Band split STFT using Learnable Gaussian Filters.
        Replaces fixed hard-split indices with differentiable soft masks.
        """
        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.n_bands = n_bands
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1. Register Frequency Axis (Static)
        # 0Hz to Nyquist Frequency
        freq_axis = torch.linspace(0, sr / 2, self.n_bins)
        self.register_buffer('freq_axis', freq_axis)

        # 2. Initialize Learnable Parameters (Mu and Sigma)
        # We define them in "Hz" space for intuitive initialization
        self.mu, self.sigma = self._init_parameters(requires_grad)

        # 3. Projection Layers
        # Note: Input dimension is now (n_bins * in_channels) because we use 
        # soft masking over the full spectrum, not sparse indexing.
        self.pre_bandnet = BandLinear(
            n_bands=self.n_bands, 
            in_channels=self.n_bins * self.in_channels, 
            out_channels=self.out_channels
        )

        self.post_bandnet = BandLinear(
            n_bands=self.n_bands, 
            in_channels=self.out_channels, 
            out_channels=self.n_bins * self.in_channels
        )

    def _init_parameters(self, requires_grad: bool):
        """Initialize Mu and Sigma using Mel scale heuristics."""
        # Calculate Mel points
        mel_min = 0
        mel_max = 2595 * np.log10(1 + (self.sr / 2) / 700)
        mel_points = np.linspace(mel_min, mel_max, self.n_bands)
        
        # Convert back to Hz for Mu
        mu_hz = 700 * (10**(mel_points / 2595) - 1)
        
        # Initialize Sigma (bandwidth)
        # A good heuristic is the distance to the next band / 2
        # We ensure a minimum width to avoid vanishing gradients
        mu_tensor = torch.tensor(mu_hz, dtype=torch.float32)
        
        # Calculate diffs to estimate width
        diffs = torch.cat([mu_tensor[1:] - mu_tensor[:-1], torch.tensor([mu_tensor[-1] - mu_tensor[-2]])])
        sigma_hz = diffs  # Start with broad overlap
        
        mu = nn.Parameter(mu_tensor, requires_grad=requires_grad)
        sigma = nn.Parameter(sigma_hz, requires_grad=requires_grad)
        
        return mu, sigma

    def get_filterbank(self):
        """Generates the Gaussian masks based on current Mu and Sigma."""
        f = self.freq_axis.unsqueeze(0)   # (1, F)
        mu = self.mu.unsqueeze(1)         # (K, 1)
        sigma = self.sigma.unsqueeze(1)   # (K, 1)
        
        # -----------------------------------------------------------
        # REPLACEMENT: Use Softplus to enforce positivity
        # -----------------------------------------------------------
        # F.softplus(x) = log(1 + exp(x))
        # We add 1e-3 (or a small Hz value) to ensure sigma never hits true 0
        sigma = torch.nn.functional.softplus(sigma) + 1e-3
        
        # Gaussian Formula
        exponent = -0.5 * ((f - mu) / sigma) ** 2
        mask = torch.exp(exponent)
        
        return mask

    def transform(self, x: Tensor) -> Tensor:
        r"""
        Optimized Forward Pass using Weight Folding.
        """
        # x: (b, c, t, f)
        b, c, t, f = x.shape
        
        # 1. Get Gaussian Masks (k, f)
        mask = self.get_filterbank() 
        
        # 2. Get Linear Weights from the sub-module
        # shape: (k, in_feat, out_feat) where in_feat = c * f
        w = self.pre_bandnet.weight 
        bias = self.pre_bandnet.bias # (k, out_feat)

        # 3. Reshape Weights to separate Channels and Freq
        # We need to match the mask shape.
        # w: (k, c*f, d) -> (k, c, f, d)
        w_reshaped = rearrange(w, 'k (c f) d -> k c f d', c=c, f=f)
        
        # 4. FUSE MASK INTO WEIGHTS
        # Mask broadcasts over C and D dimensions
        # mask: (k, 1, f, 1)
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1)
        
        # The Magic Step: We weight the projection matrix, not the input.
        # This keeps the memory footprint tiny.
        w_effective = w_reshaped * mask_expanded # (k, c, f, d)
        
        # 5. Flatten effective weights back
        # w_effective: (k, c*f, d)
        w_effective = rearrange(w_effective, 'k c f d -> k (c f) d')
        
        # 6. Apply Linear Projection
        # Input x needs to be: (b, t, c*f)
        # We want output: (b, d, t, k)
        
        x_flat = rearrange(x, 'b c t f -> b t (c f)')
        
        # We use einsum for the projection
        # b: batch, t: time, i: input_dim (c*f), k: bands, o: output_dim (latent)
        # x: (b, t, i)
        # w: (k, i, o)
        # result: (b, t, k, o)
        
        # NOTE: This is still computationally heavy (dense matrix mult), 
        # but it saves massive amounts of VRAM and memory bandwidth.
        x_out = torch.einsum('bti, kio -> btko', x_flat, w_effective)
        
        # Add bias
        x_out = x_out + bias
        
        # Rearrange to final shape
        x_out = rearrange(x_out, 'b t k d -> b d t k')
        
        return x_out

    def inverse_transform(self, x: Tensor) -> Tensor:
        r"""
        Optimized Inverse Transform using Weight Folding.
        Performs Projection + Masking + Band Summation in one optimized Einsum.
        
        Input x: (b, d, t, k)
        Output:  (b, c, t, f)
        """
        # x: (b, d, t, k)
        b, d, t, k = x.shape
        c = self.in_channels
        f = self.n_bins
        
        # 1. Prepare Input: Move K and D to last positions for Einsum
        # (b, d, t, k) -> (b, t, k, d)
        x = rearrange(x, 'b d t k -> b t k d')

        # 2. Get Mask & Raw Weights
        mask = self.get_filterbank()   # (k, f)
        w = self.post_bandnet.weight   # (k, d_in, c*f)
        bias = self.post_bandnet.bias  # (k, c*f)

        # 3. Fuse Mask into Weights (Weight Folding)
        
        # --- PROCESS WEIGHTS (4D) ---
        # w_reshaped: (k, d, c, f)
        w_reshaped = rearrange(w, 'k d (c f) -> k d c f', c=c, f=f)
        
        # Expand mask for weights: (k, f) -> (k, 1, 1, f)
        mask_for_weights = mask.unsqueeze(1).unsqueeze(1)
        
        # Apply mask
        w_effective = w_reshaped * mask_for_weights  # (k, d, c, f)
        
        # Flatten back for Linear calculation: (k, d, c*f)
        w_effective = rearrange(w_effective, 'k d c f -> k d (c f)')

        # --- PROCESS BIAS (3D) ---
        # bias_reshaped: (k, c, f)
        bias_reshaped = rearrange(bias, 'k (c f) -> k c f', c=c, f=f)
        
        # Expand mask for bias: (k, f) -> (k, 1, f)
        # FIX: We only unsqueeze once for bias, so dimensions match (K, C, F)
        mask_for_bias = mask.unsqueeze(1) 
        
        # Apply mask
        bias_effective = bias_reshaped * mask_for_bias # (k, c, f)

        # 4. The Optimized Projection + Summation
        # x: (b, t, k, d)
        # w_effective: (k, d, o) -> where o = c*f
        # Output: (b, t, o) -> The 'k' dimension is contracted (summed) here
        x_out = torch.einsum('btkd, kdo -> bto', x, w_effective)
        
        # 5. Handle Bias
        # Sum the masked biases over the bands (Overlap-Add)
        # bias_effective: (k, c, f) -> Sum over k -> (c, f)
        bias_sum = bias_effective.sum(dim=0) 
        
        # Flatten to match x_out shape: (c*f)
        bias_sum = rearrange(bias_sum, 'c f -> (c f)')
        
        # Add bias
        x_out = x_out + bias_sum

        # 6. Reshape to Spectrogram Format
        x_out = rearrange(x_out, 'b t (c f) -> b c t f', c=c, f=f)
        
        # 7. Normalize (Divide by Overlap-Add Window)
        # We divide by sum(mask^2) because we applied the mask twice 
        # (once in analysis, once here in synthesis)
        ola_window = torch.sum(mask ** 2, dim=0) # (f,)
        
        # Reshape for broadcasting: (1, 1, 1, f)
        ola_window = ola_window.view(1, 1, 1, -1)
        
        # Apply normalization with epsilon
        x_out = x_out / (ola_window + 1e-7)
        
        return x_out


class BandLinear(nn.Module):
    """
    Same as original, but we rely on it handling larger input dimensions
    due to dense masking.
    """
    def __init__(self, n_bands: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_bands, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(n_bands, out_channels))
        
        bound = 1 / math.sqrt(in_channels)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # x: (b, t, k, i)
        # weight: (k, i, o)
        return torch.einsum('btki,kio->btko', x, self.weight) + self.bias

if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    
    # Example Params
    B, C, T, F_bins = 2, 2, 100, 1025
    n_bands = 256
    dim_latent = 384
    
    x = torch.randn(B, C, T, F_bins)

    # Instantiate New Module
    # Note: in_channels is effectively much larger now because we see the full freq axis
    bandsplit = LearnableBandSplit(
        sr=44100, 
        n_fft=2048, 
        n_bands=n_bands,
        in_channels=C, 
        out_channels=dim_latent
    )

    print(f"Original Input: {x.shape}")
    
    # 1. Forward
    y = bandsplit.transform(x)
    print(f"Latent Output: {y.shape}") # Should be (B, dim, T, n_bands)
    
    # 2. Check Gradients (Validation that it is learnable)
    loss = y.sum()
    loss.backward()
    print(f"Mu Gradients Exist: {bandsplit.mu.grad is not None}")
    print(f"Sigma Gradients Exist: {bandsplit.sigma.grad is not None}")

    # 3. Inverse
    x_hat = bandsplit.inverse_transform(y.detach())
    print(f"Reconstruction: {x_hat.shape}")
    
    # Simple reconstruction check (will not be perfect due to compression)
    rec_error = (x - x_hat).abs().mean()
    print(f"Reconstruction Error (untrained): {rec_error.item()}")