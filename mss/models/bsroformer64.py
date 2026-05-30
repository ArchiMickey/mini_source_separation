from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from torch.nn.utils.rnn import pad_sequence

from . import bsroformer63 as base


def _default_bank_path(bank_source: Literal["mixture", "vocals"]) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root
        / "outputs"
        / "bandsplit_analysis"
        / f"{bank_source}_entropy_banks.npy"
    )


def load_equal_energy_banks(
    *,
    n_fft: int,
    n_bands: int,
    bank_source: Literal["mixture", "vocals"] = "mixture",
    bank_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(bank_path) if bank_path is not None else _default_bank_path(bank_source)
    if not path.exists():
        raise FileNotFoundError(
            f"Equal-energy bank file not found: {path}. "
            "Generate it with scripts/analyze_bandsplit_energy.py or pass bandsplit_bank_path explicitly."
        )

    banks = np.load(path).astype(np.float32)
    expected_shape = (n_bands, n_fft // 2 + 1)
    if banks.shape != expected_shape:
        raise ValueError(
            f"Equal-energy bank shape mismatch: expected {expected_shape}, got {banks.shape} from {path}."
        )

    ola_window = banks.sum(axis=0)
    if np.any(ola_window <= 0):
        raise ValueError(
            f"Invalid equal-energy banks from {path}: overlap-add window must stay positive."
        )

    return banks, ola_window.astype(np.float32)


class BandSplit(base.BandSplit):
    """BandSplit initialized from precomputed max-entropy equal-energy banks."""

    def __init__(
        self,
        sr: float,
        n_fft: int,
        n_bands: int,
        in_channels: int,
        out_channels: int,
        bank_source: Literal["mixture", "vocals"] = "mixture",
        bank_path: str | Path | None = None,
    ) -> None:
        self.bank_source = bank_source
        self.bank_path = Path(bank_path) if bank_path is not None else None
        nn.Module.__init__(self)

        self.sr = sr
        self.n_fft = n_fft
        self.n_bands = n_bands
        self.in_channels = in_channels
        self.out_channels = out_channels

        n_freqs = n_fft // 2 + 1

        melbanks_init, ola_window_init = self.init_melbanks()
        melbanks_init = Tensor(melbanks_init)
        ola_window_init = Tensor(ola_window_init)

        self.register_buffer("melbanks", melbanks_init)
        self.register_buffer("ola_window", ola_window_init)

        self.pre_w = nn.Parameter(torch.zeros((in_channels, n_freqs, out_channels)))
        self.post_w = nn.Parameter(torch.zeros((out_channels, in_channels, n_freqs)))
        self.pre_b = nn.Parameter(torch.zeros((n_bands, out_channels)))
        self.post_b = nn.Parameter(torch.zeros((in_channels, n_freqs)))

        gram = in_channels * melbanks_init.cpu().numpy() ** 2
        gain_sq = np.linalg.pinv(gram) @ np.ones(n_bands, dtype=np.float32)
        gain_sq = np.clip(gain_sq, a_min=0.0, a_max=None)
        gain = torch.from_numpy(np.sqrt(gain_sq)[None, :, None]).float()

        nn.init.uniform_(self.pre_w, -1, 1)
        with torch.no_grad():
            self.pre_w *= gain

        bound = 1 / math.sqrt(out_channels)
        nn.init.uniform_(self.post_w, -bound, bound)

        nonzero_indexes = []
        nonzero_melbanks = []

        for f in range(n_bands):
            idxes = torch.nonzero(melbanks_init[f].abs() > 1e-6, as_tuple=True)[0]
            nonzero_indexes.append(idxes)
            nonzero_melbanks.append(melbanks_init[f, idxes])

        q_groups = 4
        self.Q = q_groups
        cumsum = np.cumsum([len(idxes) for idxes in nonzero_indexes])
        total = cumsum[-1]
        subbands = []

        for q in range(q_groups):
            subband = []
            for i in range(n_bands):
                if q / q_groups * total <= cumsum[i] < (q + 1) / q_groups * total:
                    subband.append(i)
            subbands.append(subband)
        subbands[-1].append(i)

        for q in range(q_groups):
            sb_idxes = []
            sb_melbanks = []
            for f in subbands[q]:
                sb_idxes.append(nonzero_indexes[f])
                sb_melbanks.append(nonzero_melbanks[f])

            sb_idxes = pad_sequence(
                sequences=sb_idxes, batch_first=True, padding_value=-1
            )
            sb_masks = (sb_idxes != -1).float()
            sb_idxes[sb_idxes == -1] = n_fft // 2
            sb_melbanks = pad_sequence(
                sequences=sb_melbanks, batch_first=True, padding_value=0
            )

            self.register_buffer(name=f"sb_idxes_{q}", tensor=sb_idxes)
            self.register_buffer(name=f"sb_melbanks_{q}", tensor=sb_melbanks)
            self.register_buffer(name=f"sb_masks_{q}", tensor=sb_masks)
            self.register_buffer(
                name=f"sb_subbands_{q}", tensor=LongTensor(subbands[q])
            )

    def init_melbanks(self) -> tuple[np.ndarray, np.ndarray]:
        return load_equal_energy_banks(
            n_fft=self.n_fft,
            n_bands=self.n_bands,
            bank_source=self.bank_source,
            bank_path=self.bank_path,
        )


class BSRoformer(base.BSRoformer):
    """bsroformer63 with a precomputed equal-energy BandSplit initializer."""

    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_fft=2048,
        hop_length=480,
        n_bands=256,
        dim_sp=96,
        dim=384,
        dim_head=32,
        patch_size=[4, 4],
        n_layers=12,
        n_pre_layers=1,
        n_post_layers=1,
        delta_attn_ratio=3,
        rope_len=8192,
        conv_kernel_size=7,
        bandsplit_bank: Literal["mixture", "vocals"] = "mixture",
        bandsplit_bank_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            audio_channels=audio_channels,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_bands=n_bands,
            dim_sp=dim_sp,
            dim=dim,
            dim_head=dim_head,
            patch_size=patch_size,
            n_layers=n_layers,
            n_pre_layers=n_pre_layers,
            n_post_layers=n_post_layers,
            delta_attn_ratio=delta_attn_ratio,
            rope_len=rope_len,
            conv_kernel_size=conv_kernel_size,
            **kwargs,
        )

        if dim_sp % audio_channels != 0:
            raise ValueError(
                f"dim_sp ({dim_sp}) must be divisible by audio_channels ({audio_channels})."
            )

        self.bandsplit = BandSplit(
            sr=sample_rate,
            n_fft=n_fft,
            n_bands=n_bands,
            in_channels=2,
            out_channels=dim_sp // audio_channels,
            bank_source=bandsplit_bank,
            bank_path=bandsplit_bank_path,
        )
