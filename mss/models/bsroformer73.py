from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mss.models.bsroformer71 import BSRoformer as _BSRoformer
from mss.models.bsroformer71 import LightweightCrossAttention
from mss.models.bsroformer66 import BandGatedRMSNorm
from mss.models2.dsp3.banks import erb_linear_banks


class CrossAttentionFusion(nn.Module):
    """Cross-attention based skip fusion: decoder queries encoder features."""

    def __init__(self, dim, dim_head, n_bands):
        super().__init__()
        self.norm_dec = BandGatedRMSNorm(dim, n_bands)
        self.norm_enc = BandGatedRMSNorm(dim, n_bands)
        self.band_pos = nn.Parameter(torch.zeros(1, n_bands, dim))
        self.cross_attn = LightweightCrossAttention(dim=dim, dim_head=dim_head)
        self.gate = nn.Parameter(torch.tensor(0.1))
        hidden = int(dim * 2)
        self.ffn_proj_in = nn.Conv2d(dim, hidden * 2, 1)
        self.ffn_proj_out = nn.Conv2d(hidden, dim, 1)
        self.ffn_gate_param = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_dec, x_enc):
        if x_enc.shape[-2:] != x_dec.shape[-2:]:
            x_enc = F.interpolate(
                x_enc, size=x_dec.shape[-2:], mode="bilinear", align_corners=False
            )

        residual = x_dec
        b, c, t, f = x_dec.shape

        dec_normed = self.norm_dec(x_dec)
        enc_normed = self.norm_enc(x_enc)

        dec_tokens = rearrange(dec_normed, "b c t f -> (b t) f c")
        enc_tokens = rearrange(enc_normed, "b c t f -> (b t) f c")

        pos = self.band_pos[:, :f]
        dec_q = dec_tokens + pos
        enc_kv = enc_tokens + pos

        attn_out = self.cross_attn(dec_q, enc_kv)
        attn_out = rearrange(attn_out, "(b t) f c -> b c t f", b=b, t=t)

        x = residual + self.gate.tanh() * attn_out

        ffn_in = self.ffn_proj_in(x)
        gate_val, value = ffn_in.chunk(2, dim=1)
        ffn_out = self.ffn_proj_out(F.gelu(gate_val) * value)
        x = x + self.ffn_gate_param.tanh() * ffn_out

        return x


class BSRoformer(_BSRoformer):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_bands=64,
        dim_sp=96,
        dim=384,
        dim_head=32,
        patch_size=[4, 4],
        freq_pool_factor=None,
        n_layers=12,
        n_pre_layers=1,
        n_post_layers=1,
        pre_time_block: bool = False,
        post_time_block: bool = False,
        use_checkpoint: bool = False,
        delta_attn_ratio=3,
        rope_len=8192,
        conv_kernel_size=7,
        frame_rate: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(
            audio_channels=audio_channels,
            sample_rate=sample_rate,
            n_bands=n_bands,
            dim_sp=dim_sp,
            dim=dim,
            dim_head=dim_head,
            patch_size=patch_size,
            freq_pool_factor=freq_pool_factor,
            n_layers=n_layers,
            n_pre_layers=n_pre_layers,
            n_post_layers=n_post_layers,
            pre_time_block=pre_time_block,
            post_time_block=post_time_block,
            use_checkpoint=use_checkpoint,
            delta_attn_ratio=delta_attn_ratio,
            rope_len=rope_len,
            conv_kernel_size=conv_kernel_size,
            frame_rate=frame_rate,
            **kwargs,
        )
        max_bandwidth = 800
        banks = erb_linear_banks(
            sr=sample_rate, n_bands=n_bands, max_bandwidth=max_bandwidth
        )
        n_bands_full = len(banks)
        self.fusion = CrossAttentionFusion(dim_sp, dim_head, n_bands_full)


if __name__ == "__main__":
    model = BSRoformer(use_checkpoint=True)
    dummy_audio = torch.randn(1, 2, 48000 * 2)
    output = model(dummy_audio)
    print(output.shape)
