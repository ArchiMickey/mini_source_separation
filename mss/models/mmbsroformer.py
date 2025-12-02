from __future__ import annotations
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from mss.models.attention import Block
from mss.models.flex_attention import Block as FlexBlock
from mss.models.bandsplit import BandSplit
from mss.models.fourier import Fourier
from mss.models.rope import RoPE


def build_2d_time_window_mask(
    T: int,
    F: int,
    window_size: tuple,
    device="cpu",
    dtype=torch.bool
):
    """
    window_size = (left, right)

    left  >= 0: how many time steps BACKWARD (t < t_self)
    right >= 0: how many time steps FORWARD  (t > t_self)

    -1 means unbounded in that direction.

    True  = allowed to attend
    False = masked
    """

    left, right = window_size

    t_self = torch.arange(T, device=device).repeat_interleave(F) # (N,)
    t      = torch.arange(T, device=device).repeat_interleave(F) # (N,)

    t_self = t_self.unsqueeze(1) # (N,1)
    t      = t.unsqueeze(0)      # (1,N)

    dt = t - t_self              # positive = future, negative = past

    # Left (past)
    if left == -1:
        past_ok = torch.ones_like(dt, dtype=torch.bool)
    else:
        past_ok = (dt >= -left)

    # Right (future)
    if right == -1:
        future_ok = torch.ones_like(dt, dtype=torch.bool)
    else:
        future_ok = (dt <= right)

    allowed = past_ok & future_ok

    return allowed.to(dtype)

def generate_mask_mod(mask):
    def mask_mod(b, h, q_idx, kv_idx):
        return mask[q_idx, kv_idx]
    return mask_mod

class RoformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dim_in=None, stride=1, depth=1, window_size=(-1, -1), use_flex_attention=False):
        super().__init__()
        dim_in = dim if dim_in is None else dim_in
        self.stride = stride
        self.window_size = window_size
        self.use_flex_attention = use_flex_attention
        
        self.down = nn.Conv2d(dim_in, dim, kernel_size=(stride, 1), stride=(stride, 1)) if stride > 1 else nn.Identity()
        block_cls = FlexBlock if use_flex_attention else Block
        self.blocks = nn.ModuleList([block_cls(dim, n_heads) for _ in range(depth)])
        self.up = nn.ConvTranspose2d(dim, dim_in, kernel_size=(stride, 1), stride=(stride, 1)) if stride > 1 else nn.Identity()
        self.layerscale = nn.Parameter(torch.ones(1, dim, 1, 1))
        
        self.attn_mask_cache = None
    
    def get_pos(self, x: Tensor) -> Tensor:
        T, F = x.shape[2], x.shape[3]
        t = torch.arange(T, device=x.device)
        f = torch.arange(F, device=x.device)
        tt, ff = torch.meshgrid(t, f, indexing='ij')  # shape: (T, F)
        coords = torch.stack([tt, ff], dim=-1).view(-1, 2)
        return coords
    
    def forward(self, x, rope=None):
        # x: (b, d, t, f)
        inp = x
        x = self.down(x)
        
        if self.window_size != (-1, -1):
            L = x.shape[2] * x.shape[3]
            if self.attn_mask_cache is not None and self.attn_mask_cache.shape[2:] == (L, L):
                attn_mask = self.attn_mask_cache
            else:
                mask = build_2d_time_window_mask(
                    T=x.shape[2],
                    F=x.shape[3],
                    window_size=self.window_size,
                    device=x.device,
                )
                if self.use_flex_attention:
                    attn_mask = create_block_mask(generate_mask_mod(mask), None, None, L, L, x.device)
                else:
                    attn_mask = create_mask(generate_mask_mod(mask), None, None, L, L, x.device)
                self.attn_mask_cache = attn_mask
        else:
            attn_mask = None
        
        pos = self.get_pos(x)  # shape: (T*F, 2)
        T = x.shape[2]
        x = rearrange(x, 'b d t f -> b (t f) d')
        for block in self.blocks:
            x = block(x, rope=rope, pos=pos, attn_mask=attn_mask)  # shape: (b, t * f, d)
        x = rearrange(x, 'b (t f) d -> b d t f', t=T)  # shape: (b, d, t, f)
        
        x = self.up(x) * self.layerscale
        
        return x + inp
        

class BSRoformer(Fourier):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_fft=2048,
        hop_length=480,
        n_bands=256,
        band_dim=64,
        patch_size=[4, 4],
        dim=768,
        strides=[1, 2, 4, 2, 1],
        n_layers=[2, 2, 4, 4, 4],
        n_heads=12,
        rope_len=8192,
        window_size=(-1, -1),
        use_flex_attention=False,
        **kwargs
    ) -> None:
        super().__init__(
            n_fft=n_fft, 
            hop_length=hop_length, 
            return_complex=True, 
            normalized=True
        )

        self.patch_size = patch_size
        self.strides = strides
        self.window_size = window_size
        self.use_flex_attention = use_flex_attention

        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate,
            n_fft=n_fft, 
            n_bands=n_bands,
            in_channels=audio_channels * 2,  # real + imag
            out_channels=band_dim
        )

        self.patch = nn.Conv2d(band_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.unpatch = nn.ConvTranspose2d(dim, band_dim, kernel_size=patch_size, stride=patch_size)

        # RoPE
        self.rope = RoPE(head_dim=dim // n_heads // 2, max_len=rope_len)

        # Transformer blocks
        self.stages = nn.ModuleList()
        assert len(n_layers) == len(strides), "n_layers and strides must have the same length."
        for depth, stride in zip(n_layers, strides):
            stage_window_size = (
                -1 if self.window_size[0] == -1 else self.window_size[0] // stride,
                -1 if self.window_size[1] == -1 else self.window_size[1] // stride,
            )
            stage = RoformerBlock(
                dim=dim,
                n_heads=n_heads,
                stride=stride,
                depth=depth,
                window_size=stage_window_size,
                use_flex_attention=use_flex_attention
            )
            self.stages.append(stage)
        
    def forward(self, audio: Tensor) -> Tensor:
        r"""Separation model.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: freq_bins

        Args:
            audio: (b, c, t)

        Outputs:
            output: (b, c, t)
        """

        # --- 1. Encode ---
        # 1.1 Complex spectrum
        complex_sp = self.stft(audio)  # shape: (b, c, t, f)

        x = torch.view_as_real(complex_sp)  # shape: (b, c, t, f, 2)
        x = rearrange(x, 'b c t f k -> b (c k) t f')  # shape: (b, d, t, f)
        T0 = x.shape[2]

        # 1.2 Pad stft
        x = self.pad_tensor(x)  # x: (b, d, t, f)

        # 1.3 Convert STFT to mel scale
        x = self.bandsplit.transform(x)  # shape: (b, d, t, f)

        # 1.4 Patchify
        x = self.patch(x)  # shape: (b, d, t, f)
        T1 = x.shape[2]

        L = x.shape[2] * x.shape[3]
        # --- 2. Transformer ---
        for stage in self.stages:
            x = stage(x, rope=self.rope)  # shape: (b, t * f, d)

        # --- 3. Decode ---
        # 3.1 Unpatchify
        x = self.unpatch(x)  # shape: (b, d, t, f)

        # 3.2 Convert mel scale STFT to original STFT
        x = self.bandsplit.inverse_transform(x)  # shape: (b, d, t, f)

        # Unpad
        x = x[:, :, :T0, :]
        
        # 3.3 Get complex mask
        x = rearrange(x, 'b (c k) t f -> b c t f k', k=2).contiguous()
        mask = torch.view_as_complex(x)  # shape: (b, c, t, f)

        # 3.5 Calculate stft of separated audio
        sep_stft = mask * complex_sp  # shape: (b, c, t, f)

        # 3.6 ISTFT
        output = self.istft(sep_stft)  # shape: (b, c, l)

        return output

    def pad_tensor(self, x: Tensor) -> tuple[Tensor, int]:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f)
        
        Outpus:
            output: E.g., (b, c, t=204, f)
        """

        # Pad last frames, e.g., 201 -> 204
        max_stride = max(self.patch_size[0] * stride for stride in self.strides)
        pad_t = -x.shape[2] % max_stride  # Equals to p - (T % p)
        x = F.pad(x, pad=(0, 0, 0, pad_t))

        return x