import torch
import torch.nn as nn
import torchaudio
import pytorch_lightning as pl
from ema_pytorch import EMA
from safetensors.torch import save_file
from typing import Union, Optional
import wandb
import typing as tp

from pytorch_lightning.utilities import rank_zero_only

import numpy as np
import museval
import librosa
from pathlib import Path
import os
from tqdm.auto import tqdm
from einops import rearrange

from stable_audio_tools.inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper
from data.dataset import load
from .losses.losses import MSELoss, MultiLoss
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
    
class DiffusionMSSTrainingWrapper(pl.LightningModule):
    def __init__(
        self,
        model: ConditionedDiffusionModelWrapper,
        target_instrument: str,
        lr: float = None,
        use_ema: bool = True,
        optimizer_configs: dict = None,
        pre_encoded: bool = False,
        cfg_dropout_prob: float = 0.1,
        timestep_sampler: tp.Literal["uniform", "logit_normal"] = "uniform",
    ):
        super().__init__()
        
        self.target_instrument = target_instrument
        self.diffusion = model
        
        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None
        
        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler
        
        self.diffusion_objective = model.diffusion_objective

        self.loss_modules = [
            MSELoss("output", 
                   "targets", 
                   weight=1.0, 
                   name="mse_loss"
            )
        ]
        
        self.losses = MultiLoss(self.loss_modules)

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "Adam",
                        "config": {
                            "lr": lr
                        }
                    }
                }
            }
        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded
    
    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]
    
    def training_step(self, batch, batch_idx):
        p = Profiler()

        audios, info = batch
        
        loss_info = {}
        
        # diffusion_input = audios[self.target_instrument]
        diffusion_input = audios["mixture"] - audios[self.target_instrument]
        mixture_cond = audios["mixture"]
        
        p.tick("setup")
        
        conditioning = {}
        
        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)
            
            if not self.pre_encoded:
                with torch.amp.autocast(str(self.device)) and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    mixture_cond = self.diffusion.pretransform.encode(mixture_cond)
        else:            
            # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
            if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                diffusion_input = diffusion_input / self.diffusion.pretransform.scale
                mixture_cond = mixture_cond / self.diffusion.pretransform.scale
        
        conditioning["mixture"] = (mixture_cond, torch.ones((mixture_cond.shape[0], mixture_cond.shape[1]), device=self.device))
                
        if self.timestep_sampler == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(diffusion_input.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(diffusion_input.shape[0], device=self.device))
            
        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1-t, t
            
        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas
        
        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input
        
        p.tick("noise")

        extra_args = {}
        
        with torch.amp.autocast(str(self.device)):
            p.tick("amp")
            output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, **extra_args)
            p.tick("diffusion")

            loss_info.update({
                "output": output,
                "targets": targets,
            })

            loss, losses = self.losses(loss_info)

            p.tick("loss")
        
        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        p.tick("log")
        #print(f"Profiler: {p}")
        return loss
        
        
    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model
        
        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

class DiffusionMSSDemoCallback(pl.Callback):
    def __init__(self, 
                 demo_dl,
                 sample_rate,
                 demo_every=2000,
                 demo_steps=250,
                 demo_cfg_scales: tp.Optional[tp.List[float]] = [3, 5, 7],
    ):
        super().__init__()

        self.demo_every = demo_every
        self.last_demo_step = -1
        self.demo_dl = demo_dl
        self.sample_rate = sample_rate
        self.demo_steps = demo_steps
        self.demo_cfg_scales = demo_cfg_scales
        
        if not os.path.exists("demo"):
            os.makedirs("demo")
        
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        
        module.eval()
        
        print(f"Generating demo")
        self.last_demo_step = trainer.global_step
        
        try:
            audios, info = next(iter(self.demo_dl))
            
            mixture = audios["mixture"].to(module.device)
            mixture_cond = mixture
            
            demo_samples = mixture_cond.shape[-1]
            
            if module.diffusion.pretransform is not None:
                demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio
            
            noise = torch.randn([mixture_cond.shape[0], module.diffusion.io_channels, demo_samples]).to(module.device)
            if module.diffusion.pretransform is not None:
                mixture_cond = module.diffusion.pretransform.encode(mixture_cond)
            
            conditioning = {"mixture": (mixture_cond, torch.ones((mixture_cond.shape[0], mixture_cond.shape[1]), device=module.device))}
            
            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)
            
            for cfg_scale in self.demo_cfg_scales:
                print(f"Generating demo for cfg scale {cfg_scale}")
                with torch.autocast(str(module.device)):
                    model = module.diffusion_ema.model if module.diffusion_ema is not None else module.diffusion.model
                    
                    if module.diffusion_objective == "v":
                            fakes = sample(model, noise, self.demo_steps, 0, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)
                    elif module.diffusion_objective == "rectified_flow":
                        fakes = sample_discrete_euler(model, noise, self.demo_steps, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)
                    
                    if module.diffusion.pretransform is not None:
                        fakes = module.diffusion.pretransform.decode(fakes)
                
                reverse_fakes = audios["mixture"] - fakes
                
                reals_fakes = rearrange([audios["mixture"], fakes, audios[module.target_instrument], reverse_fakes], "i b d n -> (b i) d n")
                reals_fakes = rearrange(reals_fakes, "b d n -> d (b n)")
                
                log_dict = {}
                
                filename = f"demo/mss_cfg_{cfg_scale}_{trainer.global_step}.wav"
                reals_fakes = reals_fakes.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, reals_fakes, self.sample_rate)
                
                log_dict[f"mss_cfg_{cfg_scale}"] = wandb.Audio(filename, sample_rate=self.sample_rate, caption=f"MSS_cfg_{cfg_scale}")
                
                trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e
        finally:
            module.train()