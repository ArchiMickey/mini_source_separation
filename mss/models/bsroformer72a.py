from __future__ import annotations

from mss.models.bsroformer71a import BSRoformer as _BSRoformer


class BSRoformer(_BSRoformer):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_bands=128,
        dim_sp=96,
        dim=384,
        dim_head=32,
        patch_size=[4, 4],
        freq_pool_factor=2,
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
            freq_pool_factor=2,
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
        self.freq_pool_factor = 2


if __name__ == "__main__":
    model = BSRoformer(use_checkpoint=True)
    dummy_audio = __import__("torch").randn(1, 2, 48000 * 2)
    output = model(dummy_audio)
    print(output.shape)
