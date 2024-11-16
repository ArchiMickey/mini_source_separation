from typing import Optional

import os
import time
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from torch.utils.data import Dataset
from pathlib import Path

from hydra.utils import instantiate


class RandomCrop:
    r"""Prepare start time and duration of to crop from random time."""

    def __init__(
        self, clip_duration: float, end_pad: float = 0  # Pad silent at the end (s)
    ):
        self.clip_duration = clip_duration
        self.end_pad = end_pad

    def __call__(self, audio_duration: float) -> tuple[float, float]:

        padded_duration = audio_duration + self.end_pad

        if self.clip_duration <= padded_duration:
            start_time = random.uniform(0.0, padded_duration - self.clip_duration)

        else:
            start_time = 0

        return start_time, self.clip_duration

def load(
        path: str,
        sr: int,
        offset: float = 0.0,  # Load start time (s)
        duration: Optional[float] = None,  # Load duration (s)
        mono: bool = False,
    ) -> np.ndarray:
        r"""Load audio.

        Returns:
        audio: (channels, audio_samples)

        Examples:
            >>> audio = load_audio(path="xx/yy.wav", sr=16000)
        """

        # Load audio
        audio, in_sr = torchaudio.load(
            path,
        )
        # (channels, audio_samples)

        # Resample. Faster than librosa
        audio = torchaudio.functional.resample(
            waveform=audio, orig_freq=in_sr, new_freq=sr
        )
        # shape: (channels, audio_samples)

        # Calculate the end of audio
        start_sample = round(offset * sr)
        end_sample = (
            round((offset + duration) * sr) if duration else audio.shape[1]
        )
        audio = audio[:, start_sample:end_sample]

        if mono:
            audio = audio.mean(dim=0, keepdim=True)

        return audio

class MUSDB18HQ(Dataset):
    r"""MUSDB18HQ [1] is a dataset containing 100 training audios and 50
    testing audios with vocals, bass, drums, other stems. Audios are stereo and
    sampled at 48,000 Hz. Dataset size is 30 GB.

    [1] https://zenodo.org/records/3338373

    The dataset looks like:

        dataset_root (30 GB)
        ├── train (100 files)
        │   ├── A Classic Education - NightOwl
        │   │   ├── bass.wav
        │   │   ├── drums.wav
        │   │   ├── mixture.wav
        │   │   ├── other.wav
        │   │   └── vocals.wav
        │   ...
        │   └── ...
        └── test (50 files)
            ├── Al James - Schoolboy Facination
            │   ├── bass.wav
            │   ├── drums.wav
            │   ├── mixture.wav
            │   ├── other.wav
            │   └── vocals.wav
            ...
            └── ...
    """

    url = "https://zenodo.org/records/3338373"

    duration = 35359.56  # Dataset duration (s), 9.8 hours, including training,
    # valdiation, and testing

    source_types = ["bass", "drums", "other", "vocals"]
    acc_source_types = ["bass", "drums", "other"]

    def __init__(
        self,
        root: str = None,
        split: str = "train",
        sample_rate: int = 44100,
        crop: callable = RandomCrop(clip_duration=2.0),
        target_source_type: Optional[str] = "vocals",
        remix: dict = {"no_remix": 0.1, "half_remix": 0.4, "full_remix": 0.5},
        transform: Optional[callable] = None,
    ):
        self.root = root
        self.split = split
        self.sr = sample_rate
        self.crop = crop
        self.target_source_type = target_source_type
        self.remix_types = list(remix.keys())
        self.remix_weights = list(remix.values())
        self.transform = transform

        assert np.sum(self.remix_weights) == 1.0

        if not Path(self.root).exists():
            raise Exception(
                "Please download the MUSDB18HQ dataset from {}".format(MUSDB18HQ.url)
            )

        self.audios_dir = Path(self.root, self.split)
        self.audio_names = sorted(os.listdir(self.audios_dir))

    def get_start_times(
        self,
        audio_duration: float,
        source_types: str,
        target_source_type: str,
        remix_type: str,
    ) -> dict:

        start_time_dict = {}

        if remix_type == "no_remix":

            start_time1, _ = self.crop(audio_duration=audio_duration)

            for source_type in source_types:
                start_time_dict[source_type] = start_time1

        elif remix_type == "half_remix":

            start_time1, _ = self.crop(audio_duration=audio_duration)
            start_time2, _ = self.crop(audio_duration=audio_duration)

            for source_type in source_types:
                if source_type == target_source_type:
                    start_time_dict[source_type] = start_time1
                else:
                    start_time_dict[source_type] = start_time2

        elif remix_type == "full_remix":

            for source_type in source_types:
                start_time, _ = self.crop(audio_duration=audio_duration)
                start_time_dict[source_type] = start_time

        else:
            raise NotImplementedError

        return start_time_dict

    def __getitem__(self, index: int) -> dict:

        source_types = MUSDB18HQ.source_types
        acc_source_types = MUSDB18HQ.acc_source_types

        audio_name = self.audio_names[index]

        info = {
            "dataset_name": "MUSDB18HQ",
            "audio_path": str(Path(self.audios_dir, audio_name)),
        }

        remix_type = random.choices(
            population=self.remix_types, weights=self.remix_weights
        )[0]

        info["remix_type"] = remix_type

        audio_path = Path(self.audios_dir, audio_name, "mixture.wav")
        audio_info = torchaudio.info(str(audio_path))
        audio_duration = audio_info.num_frames / audio_info.sample_rate

        info["audio_duration"] = audio_duration

        start_time_dict = self.get_start_times(
            audio_duration=audio_duration,
            source_types=source_types,
            target_source_type=self.target_source_type,
            remix_type=remix_type,
        )
        info["start_time_dict"] = start_time_dict

        data = {}
        for source_type in source_types:

            audio_path = Path(self.audios_dir, audio_name, "{}.wav".format(source_type))
            start_time = start_time_dict[source_type]

            # Load audio
            audio = load(
                path=audio_path, sr=self.sr, offset=start_time, duration=self.crop.clip_duration
            )
            # shape: (channels, audio_samples)

            if self.transform is not None:
                audio = self.transform(audio)

            data[source_type] = audio
            data["{}_start_time".format(source_type)] = start_time

        data["accompaniment"] = torch.stack(
            [data[source_type] for source_type in acc_source_types]
        ).sum(dim=0)
        # shape: (channels, audio_samples)

        data["mixture"] = torch.stack(
            [data[source_type] for source_type in source_types]
        ).sum(dim=0)
        # shape: (channels, audio_samples)

        end_time = time.time()
        info["load_time"] = end_time - start_time

        return data, info

    def __len__(self) -> int:

        audios_num = len(self.audio_names)

        return audios_num

    def get_start_times(
        self,
        audio_duration: float,
        source_types: str,
        target_source_type: str,
        remix_type: str,
    ) -> dict:

        start_time_dict = {}

        if remix_type == "no_remix":

            start_time1, _ = self.crop(audio_duration=audio_duration)

            for source_type in source_types:
                start_time_dict[source_type] = start_time1

        elif remix_type == "half_remix":

            start_time1, _ = self.crop(audio_duration=audio_duration)
            start_time2, _ = self.crop(audio_duration=audio_duration)

            for source_type in source_types:
                if source_type == target_source_type:
                    start_time_dict[source_type] = start_time1
                else:
                    start_time_dict[source_type] = start_time2

        elif remix_type == "full_remix":

            for source_type in source_types:
                start_time, _ = self.crop(audio_duration=audio_duration)
                start_time_dict[source_type] = start_time

        else:
            raise NotImplementedError

        return start_time_dict


class InfiniteSampler:
    def __init__(self, dataset):

        self.indexes = list(range(len(dataset)))
        random.shuffle(self.indexes)
    
    def __iter__(self):

        pointer = 0

        while True:

            if pointer == len(self.indexes):
                random.shuffle(self.indexes)
                pointer = 0

            index = self.indexes[pointer]
            pointer += 1

            yield index


def create_dataloader_from_config(dataset_config, sample_rate, batch_size, num_workers):
    dataset = instantiate(dataset_config, sample_rate=sample_rate)

    sampler = InfiniteSampler(dataset)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
