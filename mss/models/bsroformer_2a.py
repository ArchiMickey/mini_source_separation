from __future__ import annotations
from functools import lru_cache

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from torch import Tensor

from mss.models.attention import Block
from mss.models.flex_attention import Block as FlexBlock
from mss.models.bandsplit import BandSplit
from mss.models.fourier import Fourier
from mss.models.rope import RoPE


@lru_cache
def create_block_mask_cached(mask_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(mask_mod, B, H, M, N, device=device)
    return block_mask

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
        n_layers=12,
        n_freq_layers=6,
        n_heads=12,
        dim=768,
        rope_len=8192,
        window_size=(-1, -1),
        num_freq_registers=8,
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
        self.use_flex_attention = use_flex_attention
        self.window_size = window_size
        self.num_freq_registers = num_freq_registers
        self.mask_mod = generate_mask_mod(window_size) if window_size != (-1, -1) else None

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
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)

        # Transformer blocks
        block_cls = FlexBlock if use_flex_attention else Block
        self.t_blocks = nn.ModuleList(block_cls(dim, n_heads) for _ in range(n_layers))

        self.freq_registers = nn.Parameter(torch.randn(num_freq_registers, dim))
        self.f_encoder_blocks = nn.ModuleList(block_cls(dim, n_heads) for _ in range(n_freq_layers))
        self.f_decoder_blocks = nn.ModuleList(block_cls(dim, n_heads) for _ in range(n_freq_layers))

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
        B = x.shape[0]
        T1 = x.shape[2]
        F1 = x.shape[3]

        x = rearrange(x, 'b d t f -> (b t) f d')
        r = rearrange(self.freq_registers, 'm d -> 1 m d').expand(x.shape[0], -1, -1)
        x, ps = pack([r, x], 'b * d')

        f_pos = torch.cat(
            [
                torch.arange(self.num_freq_registers, device=x.device) * (F1 - 1) / (self.num_freq_registers - 1),
                torch.arange(F1, device=x.device),
            ],
            dim=0,
        ).view(-1, 1).to(torch.long)

        for f_block in self.f_encoder_blocks:
            x = f_block(x, rope=self.rope, pos=f_pos)  # shape: (b*t, f+m, d)

        r, freq_feats = unpack(x, ps, 'b * d')
        x = rearrange(r, '(b t) m d -> (b m) t d', b=B)

        if self.window_size != (-1, -1):
            if self.use_flex_attention:
                attn_mask_t = create_block_mask_cached(
                    self.mask_mod, None, None, T1, T1, x.device
                )
            else:
                attn_mask_t = create_mask(
                    self.mask_mod, 1, 1, T1, T1, x.device
                )
        else:
            attn_mask_t = None

        # print(attn_mask_t, attn_mask_f)  # --- IGNORE ---

        # --- 2. Transformer along time and frequency axes ---
        for t_block in self.t_blocks:
            x = t_block(x, rope=self.rope, pos=None, attn_mask=attn_mask_t)  # shape: (b*f, t, d)

        x = rearrange(x, '(b m) t d -> (b t) m d', b=B)
        x, ps = pack([x, freq_feats], 'b * d')

        for f_block in self.f_decoder_blocks:
            x = f_block(x, rope=self.rope, pos=f_pos)  # shape: (b*t, f+m, d)

        _, x = unpack(x, ps, 'b * d')
        x = rearrange(x, '(b t) f d -> b d t f', b=B)

        # --- 3. Decode ---
        # 3.1 Unpatchify
        x = self.unpatch(x)  # shape: (b, d, t, f)

        # 3.2 Convert mel scale STFT to original STFT
        x = self.bandsplit.inverse_transform(x)  # shape: (b, d, t, f)

        # Unpad
        x = x[:, :, 0 : T0, :]

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
        pad_t = -x.shape[2] % self.patch_size[0]  # Equals to p - (T % p)
        x = F.pad(x, pad=(0, 0, 0, pad_t))

        return x
