from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import wandb
# import trackio as wandb
from mss.utils import (parse_yaml, requires_grad, update_ema, LinearWarmUp, 
    LinearWarmUpLinearDecay, LinearWarmUpConstantCosine, separate_overlap_add, calculate_sdr)


def count_params(model: nn.Module) -> str:
    r"""Count the number of parameters in the model."""

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return f"Total params: {total_params/1e6:.2f}M, Trainable params: {trainable_params/1e6:.2f}M"

def train(args) -> None:
    r"""Train a music source separation system."""

    # Arguments
    config_path = args.config
    wandb_log = not args.no_log
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]
    precision = configs["train"]["precision"]
    valid_num = configs["validate"]["audios_num"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Prepare for acceleration
    print(f"Mixed precision: {precision}")
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    # Gradient accumulation steps from config (default to 1 if not specified)
    gradient_accumulation_steps = configs["train"].get("gradient_accumulation_steps", 1)
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    accelerator = Accelerator(
        mixed_precision=precision, 
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[process_group_kwargs, ddp_kwargs]
    )

    # Datasets
    train_dataset = get_dataset(configs, split="train")

    # Sampler
    train_sampler = get_sampler(configs, train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=configs["train"]["batch_size_per_device"], 
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True
    )

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    )
    
    # Loss function
    loss_fn = get_loss_fn(configs)

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader)

    if accelerator.is_main_process:
        # EMA
        print("Preparing EMA model...")
        ema = deepcopy(model)
        requires_grad(ema, False)
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()  # EMA model should always be in eval mode
        ema.to(accelerator.device)

    # Print model parameters
    if accelerator.is_main_process:
        print(count_params(model))

    # Logger
    if wandb_log and accelerator.is_main_process:
        wandb.init(project="mss", name=f"{config_name}")

    # aa = AA(stems=train_dataset.stems)

    # Train
    # Use update_step to track optimizer steps (not batches)
    update_step = 0
    pbar = tqdm(train_dataloader, disable=not accelerator.is_main_process)
    for batch_step, data in enumerate(pbar):
        # ------ 1. Training ------
        # 1.1 Data
        target = data["target"]
        mixture = data["mixture"]

        # 1.1 Forward
        model.train()
        output = model(mixture)

        # 1.2 Loss - scale loss for gradient accumulation
        loss = loss_fn(output=output, target=target)
        loss = loss / gradient_accumulation_steps
        
        # 1.3 Optimize with gradient accumulation
        accelerator.backward(loss)
        
        # Only update weights when gradients are synchronized
        if accelerator.sync_gradients:
            log_info = {"loss": loss.item() * gradient_accumulation_steps, "lr": scheduler.get_last_lr()[0]}
            
            # Gradient clipping
            max_grad_norm = configs["train"].get("max_grad_norm", None)
            if max_grad_norm is not None:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                pbar.set_postfix(grad_norm=grad_norm.item())
                log_info["grad_norm"] = grad_norm.item()
            
            optimizer.step()  # Update all parameters based on all parameter.grad
            optimizer.zero_grad()  # Reset all parameter.grad to 0
            scheduler.step()
            if accelerator.is_main_process:
                update_ema(ema, model, decay=0.999)
            
            # Update tqdm only when optimizer is stepped
            pbar.set_description(f"Step {update_step}, Loss: {loss.item() * gradient_accumulation_steps:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            pbar.update(1)
            
            if wandb_log and accelerator.is_main_process:
                wandb.log(log_info, step=update_step)
            
            # ------ 2. Evaluation ------
            # 2.1 Evaluate
            if update_step % configs["train"]["test_every_n_steps"] == 0:
                if accelerator.is_main_process:
                    train_sdr = validate(
                        configs=configs,
                        model=accelerator.unwrap_model(ema),
                        split="train",
                        audios_num=valid_num,
                    )

                    test_sdr = validate(
                        configs=configs,
                        model=accelerator.unwrap_model(ema),
                        split="test",
                        audios_num=valid_num,
                    )

                    if wandb_log:
                        wandb.log(
                            data={
                                "train_sdr": train_sdr, 
                                "test_sdr": test_sdr,
                            },
                            step=update_step
                        )

                    print("====== Overall metrics ====== ")
                    print(f"Train SDR: {train_sdr:.2f} dB")
                    print(f"Test SDR: {test_sdr:.2f} dB")
                accelerator.wait_for_everyone()
            
            # 2.2 Save model
            if update_step % configs["train"]["save_every_n_steps"] == 0:
                if accelerator.is_main_process:
                    ckpt_path = Path(ckpts_dir, f"step={update_step}_ema.pth")
                    torch.save(accelerator.unwrap_model(ema).state_dict(), ckpt_path)
                    print("Save model to {}".format(ckpt_path))
                accelerator.wait_for_everyone()

            if update_step == configs["train"]["training_steps"]:
                break
            
            update_step += 1


# from mss.augmentations.torch.gain import RandomGain
# from mss.augmentations.torch.pitch import RandomPitch
# from einops import rearrange

class AA:
    def __init__(self, stems):
        self.stems = stems
        self.random_gain = RandomGain(min_db=-6, max_db=6)
        self.random_pitch = RandomPitch(sr=48000, min_semitone=-1., max_semitone=1.)

    def __call__(self, data):
        for stem in self.stems:
            x = rearrange(data[stem], 'b m c l -> (b m) c l')
            x = self.random_gain(x)
            x = self.random_pitch(x)
            from IPython import embed; embed(using=False); os._exit(0)
        # self.random_gain(data[])


def get_dataset(
    configs: dict, 
    split: str
) -> Dataset:
    r"""Get datasets."""

    from mss.io.crops import RandomCrop

    assert split == "train"

    sr = configs["sample_rate"]
    segment_duration = configs["segment_duration"]
    target_stem = configs["target_stem"]
    ds = f"{split}_datasets"

    for name in configs[ds].keys():
    
        if name == "MUSDB18HQ":
            from mss.datasets.musdb18hq import MUSDB18HQ
            return MUSDB18HQ(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=segment_duration, end_pad=0.),
                target_stems=[target_stem],
                time_align=configs[ds][name]["time_align"],
                mixture_transform=None,
                group_transform=None,
                stem_transform=None
            )

        elif name == "MUSDB18HQIntraMix":

            from mss.datasets.musdb18hq_mix import MUSDB18HQIntraMix

            return MUSDB18HQIntraMix(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=configs["load_duration"], end_pad=0.),
                segment_duration=configs["segment_duration"],
                target_stems=[target_stem],
                min_intra_sources=configs["augmentation"]["cpu"]["mixing"]["intra_source"]["min_sources"],
                max_intra_sources=configs["augmentation"]["cpu"]["mixing"]["intra_source"]["max_sources"],
                time_align=configs["augmentation"]["cpu"]["mixing"]["extra_source"]["time_align"],
                mixture_transform=None,
                group_transform=None,
                stem_transform=get_stem_transform(configs),

            )

        else:
            raise ValueError(name)
            

def get_sampler(configs: dict, dataset: Dataset) -> Iterable:
    r"""Get sampler."""

    name = configs["sampler"]

    if name == "RandomSongSampler":
        from mss.samplers.random_song_sampler import RandomSongSampler
        return RandomSongSampler(dataset)

    elif name == "RandomSongSampler_multi":
        from mss.samplers.random_song_sampler_multi import RandomSongSampler_multi
        return RandomSongSampler_multi(dataset)

    elif name == "RandomSongSamplerMix":
        from mss.samplers.random_song_sampler_mix import RandomSongSamplerMix
        return RandomSongSamplerMix(dataset, configs["augmentation"]["cpu"]["mixing"]["intra_source"]["max_sources"])

    else:
        raise ValueError(name)


def get_stem_transform(configs):

    stem_transform = []

    for name, value in configs["augmentation"]["cpu"].items():
        
        if name == "gain" and value["enabled"]:
            from mss.augmentations.numpy.gain import RandomGain
            stem_transform.append(RandomGain(
                min_db=value["min_db"], 
                max_db=value["max_db"]
            ))

        elif name == "eq" and value["enabled"]:
            from mss.augmentations.numpy.eq import RandomEQ
            stem_transform.append(RandomEQ(
                min_db=value["min_db"], 
                max_db=value["max_db"],
                n_bands=value["n_bands"]
            ))

        elif name == "resample" and value["enabled"]:
            from mss.augmentations.numpy.resample import RandomResample
            stem_transform.append(RandomResample(
                sr=configs["sample_rate"],
                min_ratio=value["min_ratio"],
                max_ratio=value["max_ratio"]
            ))

    return stem_transform


def get_model(
    configs: dict, 
    ckpt_path: str
) -> nn.Module:
    r"""Initialize model."""

    name = configs["model"]["name"]

    if name == "BSRoformer":
        from mss.models.bsroformer import BSRoformer
        model = BSRoformer(**configs["model"])
    if name == "BSRoformer53a":
        from mss.models.bsroformer53a import BSRoformer53a as BSRoformer
        model = BSRoformer(**configs["model"])
    elif name == "BSRoformer55e":
        from mss.models.bsroformer55e import BSRoformer53a as BSRoformer
        model = BSRoformer(**configs["model"])
    elif name == "BSRoformer55e1":
        from mss.models.bsroformer55e1 import BSRoformer53a as BSRoformer
        model = BSRoformer(**configs["model"])
    elif name == "BSRoformer56":
        from mss.models.bsroformer56 import BSRoformer
        model = BSRoformer(**configs["model"])
    elif name == "BSRoformer56a":
        from mss.models.bsroformer56a import BSRoformer
        model = BSRoformer(**configs["model"])
    elif name == "BSRoformer56b":
        from mss.models.bsroformer56b import BSRoformer
        model = BSRoformer(**configs["model"])
    elif name == "BSRoformer56c":
        from mss.models.bsroformer56c import BSRoformer
        model = BSRoformer(**configs["model"])
    elif name == "BSRoformer56d":
        from mss.models.bsroformer56d import BSRoformer
        model = BSRoformer(**configs["model"])

    else:
        raise ValueError(name)    

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

    return model


def get_loss_fn(configs: dict) -> callable:
    r"""Get loss function."""

    loss_type = configs["train"]["loss"]

    if loss_type == "l1":
        from mss.losses.l1 import l1
        return l1

    elif loss_type == "l1_wav_l1_multistft":
        from mss.losses.wav_stft import MultiResolutionSTFTLoss
        return MultiResolutionSTFTLoss()

    else:
        raise ValueError(loss_type)


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[torch.Tensor]
) -> tuple[optim.Optimizer, None | optim.lr_scheduler.LambdaLR]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    training_steps = configs["train"]["training_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params, lr=lr)

    if warm_up_steps:
        min_lr = configs["train"].get("min_lr", 1e-6)
        # Use LinearWarmUpConstantCosine scheduler with 1000 warmup steps,
        # 70% constant phase, and 30% cosine annealing
        lr_lambda = LinearWarmUpConstantCosine(
            warm_up_steps=warm_up_steps,
            total_steps=training_steps,
            constant_ratio=0.7,
            min_lr=min_lr
        )
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler
        

def validate(
    configs: dict,
    model: nn.Module,
    split: str,
    audios_num: None | int = None,
) -> float:
    r"""Validate the model on part of data.

    c: channels_num
    L: audio_samples
    """

    # root = configs[f"{split}_datasets"]["MUSDB18HQ"]["root"]
    root = f"/datasets/musdb18hq"
    sr = configs["sample_rate"]
    segment_duration = configs["segment_duration"]
    target_stem = configs["target_stem"]
    batch_size = configs["train"]["batch_size_per_device"]
    segment_samples = round(segment_duration * sr)

    # Paths
    audios_dir = Path(root, split)
    audio_names = sorted(os.listdir(audios_dir))

    if audios_num:
        # Evaluate only part of data
        skip_n = max(1, len(audio_names) // audios_num)
    else:
        skip_n = 1
    
    stems = ["vocals", "bass", "drums", "other"]
    sdrs = []

    for idx in range(0, len(audio_names), skip_n):

        # Get data
        audio_name = audio_names[idx]    
        data = {}

        for stem in stems:
            audio_path = Path(audios_dir, audio_name, f"{stem}.wav")
            audio, _ = librosa.load(audio_path, sr=sr, mono=False)  # (c, L)
            data[stem] = audio

        data["mixture"] = np.sum([data[stem] for stem in stems], axis=0)  # (c, L)

        # Foward
        output = separate_overlap_add(
            model=model, 
            audio=data["mixture"], 
            segment_samples=segment_samples,
            hop_length=segment_samples,
            batch_size=batch_size
        )  # (c, L)
        
        sdr, _ = calculate_sdr(
            output=output, 
            target=data[target_stem], 
            sr=sr, 
        )
        
        print("{}/{}, {}, SDR: {:.2f} dB".format(idx, len(audio_names), audio_name, sdr))

        sdrs.append(sdr)

    return np.nanmedian(sdrs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)
