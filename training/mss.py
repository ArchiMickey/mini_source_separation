import torch
import torch.nn as nn
import torchaudio
import pytorch_lightning as pl
from ema_pytorch import EMA
from safetensors.torch import save_file
from typing import Union, Optional
import wandb

from pytorch_lightning.utilities import rank_zero_only

import numpy as np
import museval
import librosa
from pathlib import Path
import os
from tqdm.auto import tqdm
from einops import rearrange

from models.fourier import Fourier
from data.dataset import load
from .losses.auraloss import MultiResolutionSTFTLoss, SumAndDifferenceSTFTLoss
from .losses.losses import L1Loss, AuralossLoss, MultiLoss
from .utils import create_optimizer_from_config, create_scheduler_from_config

from time import time


class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep


class MSSTrainingWrapper(pl.LightningModule):
    def __init__(
        self,
        model: Fourier,
        lr=None,
        sample_rate=44100,
        mono=False,
        use_ema=True,
        loss_configs=None,
        optimizer_configs=None,
    ):
        super().__init__()
        self.mss_model = model

        if use_ema:
            self.mss_model_ema = EMA(
                self.mss_model,
                beta=0.9999,
                power=3 / 4,
                update_every=1,
                update_after_step=1,
                include_online_model=False,
            )

        self.lr = lr
        self.use_ema = use_ema
        self.sample_rate = sample_rate
        self.mono = mono

        assert (
            lr is not None or optimizer_configs is not None
        ), "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "mss": {"optimizer": {"type": "Adam", "config": {"lr": lr}}}
            }
        else:
            if lr is not None:
                print(
                    f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs."
                )

        self.optimizer_configs = optimizer_configs

        if loss_configs is None:
            loss_configs = {
                "spectral": {
                    "type": "mrstft",
                    "config": {
                        "fft_sizes": [2048, 1024, 512, 256, 128, 64, 32],
                        "hop_sizes": [512, 256, 128, 64, 32, 16, 8],
                        "win_lengths": [2048, 1024, 512, 256, 128, 64, 32],
                        "perceptual_weighting": True,
                    },
                    "weights": {"mrstft": 1.0},
                },
                "time": {"type": "l1", "weights": {"l1": 0.0}},
            }

        self.loss_configs = loss_configs

        stft_loss_kwargs = loss_configs["spectral"]["config"]
        if mono:
            self.sdstft = MultiResolutionSTFTLoss(
                sample_rate=sample_rate, **stft_loss_kwargs
            )
        else:
            self.sdstft = SumAndDifferenceSTFTLoss(
                sample_rate=sample_rate, **stft_loss_kwargs
            )
            self.lrstft = MultiResolutionSTFTLoss(
                sample_rate=sample_rate, **stft_loss_kwargs
            )

        self.loss_modules = []
        
        self.loss_modules += [
            AuralossLoss(
                self.sdstft,
                "vocals_pred",
                "vocals",
                name="mrstft_loss",
                weight=self.loss_configs["spectral"]["weights"]["mrstft"],
            ),
        ]

        if not mono:
            self.loss_modules += [
                AuralossLoss(
                    self.lrstft,
                    "vocals_pred_left",
                    "vocals_left",
                    name="stft_loss_left",
                    weight=self.loss_configs["spectral"]["weights"]["mrstft"] / 2,
                ),
                AuralossLoss(
                    self.lrstft,
                    "vocals_pred_right",
                    "vocals_right",
                    name="stft_loss_right",
                    weight=self.loss_configs["spectral"]["weights"]["mrstft"] / 2,
                ),
            ]

        if self.loss_configs["time"]["weights"]["l1"] > 0.0:
            self.loss_modules.append(
                L1Loss(
                    "vocals_pred",
                    "vocals",
                    weight=self.loss_configs["time"]["weights"]["l1"],
                    name="l1_time_loss",
                )
            )

        self.losses = MultiLoss(self.loss_modules)

    def configure_optimizers(self):
        opt_config = self.optimizer_configs["mss"]
        opt = create_optimizer_from_config(
            opt_config["optimizer"], self.mss_model.parameters()
        )

        if "scheduler" in opt_config:
            sched_diff = create_scheduler_from_config(opt_config["scheduler"], opt)
            sched_diff_config = {"scheduler": sched_diff, "interval": "step"}
            return [opt], [sched_diff_config]

        return [opt]

    def training_step(self, batch, batch_idx):
        p = Profiler()

        audios, info = batch

        loss_info = {}

        loss_info.update({"mixture": audios["mixture"], "vocals": audios["vocals"]})

        model_output = self.mss_model(audios["mixture"])
        loss_info["vocals_pred"] = model_output

        if not self.mono:
            loss_info["vocals_left"] = audios["vocals"][:, 0:1, :]
            loss_info["vocals_right"] = audios["vocals"][:, 1:2, :]
            loss_info["vocals_pred_left"] = model_output[:, 0:1, :]
            loss_info["vocals_pred_right"] = model_output[:, 1:2, :]

        loss, losses = self.losses(loss_info)

        p.tick("loss")

        log_dict = {
            "train/loss": loss.detach(),
            "train_lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        p.tick("log")
        # print(f"Profiler: {p}")
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.mss_model_ema is not None:
            self.mss_model_ema.update()

    def export_model(self, path, use_safetensors=False):
        if self.mss_model_ema is not None:
            self.mss_model = self.mss_model_ema.ema_model

        if use_safetensors:
            save_file(self.mss_model.state_dict(), path)
        else:
            torch.save({"state_dict": self.mss_model.state_dict()}, path)

class MSSDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        sample_rate,
        demo_every=1000
    ):
        self.demo_every = demo_every
        self.last_demo_step = -1
        self.demo_dl = demo_dl
        self.sample_rate = sample_rate
        
        if not os.path.exists("demo"):
            os.makedirs("demo")
    
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step == 0 or trainer.global_step % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        
        self.last_demo_step = trainer.global_step
        
        module.eval()
        
        try:
            audios, info = next(iter(self.demo_dl))
            
            model_input = audios["mixture"].to(module.device)
            target = audios["vocals"].to(module.device)
            
            model = module.mss_model_ema if module.mss_model_ema is not None else module.mss_model
            
            model_output = model(model_input)
            
            reals_fakes = rearrange([model_input, target, model_output], "i b d n -> (b i) d n")
            reals_fakes = rearrange(reals_fakes, "b d n -> d (b n)")
            
            log_dict = {}
            
            filename = f"demo/mss_{trainer.global_step}.wav"
            reals_fakes = reals_fakes.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, reals_fakes, self.sample_rate)
            
            log_dict["mss"] = wandb.Audio(filename, sample_rate=self.sample_rate, caption="MSS")
            
            trainer.logger.experiment.log(log_dict)
        
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e
        finally:
            module.train()
            
            
    

class MSSValidateCallback(pl.Callback):
    def __init__(
        self,
        validate_every: int,
        root: str,
        splits: Union["train", "test"],
        sr: int,
        clip_duration: float,
        source_types: list,
        target_source_type: str,
        batch_size: int,
        evaluate_num: Optional[int],
        verbose: bool = False,
    ):
        self.validate_every = validate_every
        self.root = root
        self.splits = splits
        self.sr = sr
        self.clip_duration = clip_duration
        self.source_types = source_types
        self.target_source_type = target_source_type
        self.batch_size = batch_size
        self.evaluate_num = evaluate_num
        self.verbose = verbose
        self.last_val_step = -1
    
    def separate(self, model, audio, clip_samples, batch_size):
        r"""Separate a long audio.
        """

        device = next(model.parameters()).device

        audio_samples = audio.shape[1]
        padded_audio_samples = round(np.ceil(audio_samples / clip_samples) * clip_samples)
        audio = librosa.util.fix_length(data=audio, size=padded_audio_samples, axis=-1)

        clips = librosa.util.frame(
            audio, 
            frame_length=clip_samples, 
            hop_length=clip_samples
        )
        # shape: (channels_num, clip_samples, clips_num)
        
        clips = clips.transpose(2, 0, 1)
        # shape: (clips_num, channels_num, clip_samples)

        clips_num = clips.shape[0]

        pointer = 0
        outputs = []

        while pointer < clips_num:

            batch_clips = torch.Tensor(clips[pointer : pointer + batch_size].copy()).to(device)

            with torch.no_grad():
                model.eval()
                batch_output = model(mixture=batch_clips)
                batch_output = batch_output.cpu().numpy()

            outputs.append(batch_output)
            pointer += batch_size

        outputs = np.concatenate(outputs, axis=0)
        # shape: (clips_num, channels_num, clip_samples)

        channels_num = outputs.shape[1]
        outputs = outputs.transpose(1, 0, 2).reshape(channels_num, -1)
        # shape: (channels_num, clips_num * clip_samples)

        outputs = outputs[:, 0 : audio_samples]
        # shape: (channels_num, audio_samples)

        return outputs
    
    def validate(self, split, model):
        clip_samples = round(self.clip_duration * self.sr)

        audios_dir = Path(self.root, split)
        audio_names = sorted(os.listdir(audios_dir))

        all_sdrs = []

        if self.evaluate_num:
            audio_names = audio_names[0 : self.evaluate_num]

        for audio_name in tqdm(audio_names):

            data = {}

            for source_type in self.source_types:
                audio_path = Path(audios_dir, audio_name, "{}.wav".format(source_type))

                audio = load(
                    audio_path,
                    sr=self.sr,
                    mono=False
                )
                # shape: (channels, audio_samples)

                data[source_type] = audio

            data["mixture"] = np.sum([
                data[source_type] for source_type in self.source_types], axis=0)

            sep_wav = self.separate(
                model=model, 
                audio=data["mixture"], 
                clip_samples=clip_samples,
                batch_size=self.batch_size
            )

            target_wav = data[self.target_source_type]

            # Calculate SDR. Shape should be (sources_num, channels_num, audio_samples)
            (sdrs, _, _, _) = museval.evaluate([target_wav.T], [sep_wav.T])

            sdr = np.nanmedian(sdrs)
            all_sdrs.append(sdr)

            if self.verbose:
                print(audio_name, "{:.2f} dB".format(sdr))

        sdr = np.nanmedian(all_sdrs)

        return sdr
    
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step == 0 or trainer.global_step % self.validate_every != 0 or self.last_val_step == trainer.global_step:
            return

        self.last_val_step = trainer.global_step
        
        module.eval()
        
        try:
            sdrs = {}
            
            model = module.mss_model_ema if module.mss_model_ema is not None else module.mss_model
            
            for split in self.splits:
                
                sdr = self.validate(split, model)
                
                sdrs[split] = sdr
            
            print("--- step: {} ---".format(self.last_val_step))
            print("Evaluate on {} songs.".format(self.evaluate_num))
            print("Train SDR: {:.3f}".format(sdrs["train"]))
            print("Test SDR: {:.3f}".format(sdrs["test"]))
            
            log_dict = {
                "train_sdr": sdrs["train"],
                "test_sdr": sdrs["test"]
            }
            
            module.log_dict(log_dict)
            
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e
        finally:
            module.train()