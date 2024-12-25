import torch
import torchaudio
import wandb
from einops import rearrange
from safetensors.torch import save_file, save_model
from ema_pytorch import EMA
import numpy as np
from ..models.autoencoder.stable_audio_tools.training.losses.auraloss import SumAndDifferenceSTFTLoss, MultiResolutionSTFTLoss
import pytorch_lightning as pl
from ..models.autoencoder.stable_audio_tools.models.autoencoders import AudioAutoencoder
from ..models.autoencoder.stable_audio_tools.models.discriminators import EncodecDiscriminator, OobleckDiscriminator, DACGANLoss
from ..models.autoencoder.discriminators import MultiPeriodDiscriminator, MultiScaleSubbandCQTDiscriminator, CombinedDiscriminator
from ..models.autoencoder.stable_audio_tools.models.bottleneck import VAEBottleneck, RVQBottleneck, DACRVQBottleneck, DACRVQVAEBottleneck, RVQVAEBottleneck, WassersteinBottleneck
from ..models.autoencoder.stable_audio_tools.training.losses import MultiLoss, AuralossLoss, ValueLoss, L1Loss
from .losses.losses import MultiScaleMelSpectrogramLoss
from ..models.autoencoder.stable_audio_tools.training.utils import create_optimizer_from_config, create_scheduler_from_config

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import grad_norm
from aeiou.viz import pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image

class AutoencoderTrainingWrapper(pl.LightningModule):
    def __init__(
            self, 
            autoencoder: AudioAutoencoder,
            lr: float = 1e-4,
            clip_grad_norm: float = 0.0,
            warmup_steps: int = 0,
            encoder_freeze_on_warmup: bool = False,
            sample_rate=48000,
            loss_config: dict = None,
            optimizer_configs: dict = None,
            use_ema: bool = True,
            ema_copy = None,
            force_input_mono = False,
            latent_mask_ratio = 0.0,
            teacher_model: AudioAutoencoder = None
    ):
        super().__init__()

        self.automatic_optimization = False

        self.autoencoder = autoencoder

        self.warmed_up = False
        self.warmup_steps = warmup_steps
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup
        self.lr = lr
        self.grad_clip_norm = clip_grad_norm
        if clip_grad_norm > 0.0:
            print(f'Using gradient clipping with norm {clip_grad_norm}')

        self.force_input_mono = force_input_mono

        self.teacher_model = teacher_model

        if optimizer_configs is None:
            optimizer_configs ={
                "autoencoder": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": lr,
                            "betas": (.8, .99)
                        }
                    }
                },
                "discriminator": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": lr,
                            "betas": (.8, .99)
                        }
                    }
                }

            } 
            
        self.optimizer_configs = optimizer_configs

        if loss_config is None:
            scales = [2048, 1024, 512, 256, 128, 64, 32]
            hop_sizes = []
            win_lengths = []
            overlap = 0.75
            for s in scales:
                hop_sizes.append(int(s * (1 - overlap)))
                win_lengths.append(s)
        
            loss_config = {
                "discriminator": {
                    "type": "encodec",
                    "config": {
                        "n_ffts": scales,
                        "hop_lengths": hop_sizes,
                        "win_lengths": win_lengths,
                        "filters": 32
                    },
                    "weights": {
                        "adversarial": 0.1,
                        "feature_matching": 5.0,
                    }
                },
                "spectral": {
                    "type": "mrstft",
                    "config": {
                        "fft_sizes": scales,
                        "hop_sizes": hop_sizes,
                        "win_lengths": win_lengths,
                        "perceptual_weighting": True
                    },
                    "weights": {
                        "mrstft": 1.0,
                    }
                },
                "time": {
                    "type": "l1",
                    "config": {},
                    "weights": {
                        "l1": 0.0,
                    }
                }
            }
        
        self.loss_config = loss_config

        # Discriminator

        if loss_config['discriminator']['type'] == 'oobleck':
            self.discriminator = OobleckDiscriminator(**loss_config['discriminator']['config'])
        elif loss_config['discriminator']['type'] == 'encodec':
            self.discriminator = EncodecDiscriminator(in_channels=self.autoencoder.out_channels, **loss_config['discriminator']['config'])
        elif loss_config['discriminator']['type'] == 'dac':
            self.discriminator = DACGANLoss(channels=self.autoencoder.out_channels, sample_rate=sample_rate, **loss_config['discriminator']['config'])
        elif loss_config['discriminator']['type'] == 'mscqt':
            self.discriminator = MultiScaleSubbandCQTDiscriminator(in_channels=self.autoencoder.in_channels, **loss_config['discriminator']['config'])
        elif loss_config['discriminator']['type'] == 'mpd':
            self.discriminator = MultiPeriodDiscriminator(in_channels=self.autoencoder.in_channels, **loss_config['discriminator']['config'])
        elif loss_config['discriminator']['type'] == 'combined':
            assert len(loss_config['discriminator']['config']) > 0, "Combined discriminator requires a list of discriminators config dicts"
            self.discriminator = CombinedDiscriminator(loss_config['discriminator']['config'])

        self.gen_loss_modules = []

        # Adversarial and feature matching losses
        self.gen_loss_modules += [
            ValueLoss(key='loss_adv', weight=self.loss_config['discriminator']['weights']['adversarial'], name='loss_adv'),
            ValueLoss(key='feature_matching_distance', weight=self.loss_config['discriminator']['weights']['feature_matching'], name='feature_matching'),
        ]

        # Spectral reconstruction loss
        stft_loss_args = loss_config['spectral'].get('config', {})
        stft_loss_type = loss_config['spectral']['type']
        if stft_loss_type == 'mrstft':
            if self.autoencoder.out_channels == 2:
                self.sdstft = SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
                self.lrstft = MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
            else:
                self.sdstft = MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
            
            if self.teacher_model is not None:
                # Distillation losses

                stft_loss_weight = self.loss_config['spectral']['weights']['mrstft'] * 0.25
                self.gen_loss_modules += [
                    AuralossLoss(self.sdstft, 'reals', 'decoded', name='mrstft_loss', weight=stft_loss_weight), # Reconstruction loss
                    AuralossLoss(self.sdstft, 'decoded', 'teacher_decoded', name='mrstft_loss_distill', weight=stft_loss_weight), # Distilled model's decoder is compatible with teacher's decoder
                    AuralossLoss(self.sdstft, 'reals', 'own_latents_teacher_decoded', name='mrstft_loss_own_latents_teacher', weight=stft_loss_weight), # Distilled model's encoder is compatible with teacher's decoder
                    AuralossLoss(self.sdstft, 'reals', 'teacher_latents_own_decoded', name='mrstft_loss_teacher_latents_own', weight=stft_loss_weight) # Teacher's encoder is compatible with distilled model's decoder
                ]

            else:

                # Reconstruction loss
                self.gen_loss_modules += [
                    AuralossLoss(self.sdstft, 'reals', 'decoded', name='mrstft_loss', weight=self.loss_config['spectral']['weights']['mrstft']),
                ]

                if self.autoencoder.out_channels == 2:

                    # Add left and right channel reconstruction losses in addition to the sum and difference
                    self.gen_loss_modules += [
                        AuralossLoss(self.lrstft, 'reals_left', 'decoded_left', name='stft_loss_left', weight=self.loss_config['spectral']['weights']['mrstft']/2),
                        AuralossLoss(self.lrstft, 'reals_right', 'decoded_right', name='stft_loss_right', weight=self.loss_config['spectral']['weights']['mrstft']/2),
                    ]

                self.gen_loss_modules += [
                    AuralossLoss(self.sdstft, 'reals', 'decoded', name='mrstft_loss', weight=self.loss_config['spectral']['weights']['mrstft']),
                ]
        elif stft_loss_type == 'ms_melspec':
            self.gen_loss_modules += [
                MultiScaleMelSpectrogramLoss('decoded', 'reals', sampling_rate=sample_rate, **stft_loss_args, name='ms_melspec_loss', weight=self.loss_config['spectral']['weights']['ms_melspec']),
            ]
        else:
            raise ValueError(f"Unknown spectral loss type: {stft_loss_type}")

        if self.loss_config['time']['weights']['l1'] > 0.0:
            self.gen_loss_modules.append(L1Loss(key_a='reals', key_b='decoded', weight=self.loss_config['time']['weights']['l1'], name='l1_time_loss'))

        if self.autoencoder.bottleneck is not None:
            self.gen_loss_modules += create_loss_modules_from_bottleneck(self.autoencoder.bottleneck, self.loss_config)

        self.losses_gen = MultiLoss(self.gen_loss_modules)

        self.disc_loss_modules = [
            ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss'),
        ]

        self.losses_disc = MultiLoss(self.disc_loss_modules)

        # Set up EMA for model weights
        self.autoencoder_ema = None
        
        self.use_ema = use_ema

        if self.use_ema:
            self.autoencoder_ema = EMA(
                self.autoencoder,
                ema_model=ema_copy,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1
            )

        self.latent_mask_ratio = latent_mask_ratio

    def configure_optimizers(self):

        opt_gen = create_optimizer_from_config(self.optimizer_configs['autoencoder']['optimizer'], self.autoencoder.parameters())
        opt_disc = create_optimizer_from_config(self.optimizer_configs['discriminator']['optimizer'], self.discriminator.parameters())

        if "scheduler" in self.optimizer_configs['autoencoder'] and "scheduler" in self.optimizer_configs['discriminator']:
            sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
            sched_disc = create_scheduler_from_config(self.optimizer_configs['discriminator']['scheduler'], opt_disc)
            return [opt_gen, opt_disc], [sched_gen, sched_disc]

        return [opt_gen, opt_disc]
    
    def get_optimizers(self):
        opt_gen, opt_disc = self.optimizers()
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        return opt_gen, opt_disc
    
    def general_step(self, reals):
        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True

        loss_info = {}

        loss_info["reals"] = reals

        encoder_input = reals

        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        loss_info["encoder_input"] = encoder_input

        if self.warmed_up and self.encoder_freeze_on_warmup:
            with torch.no_grad():
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
        else:
            latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)

        loss_info["latents"] = latents

        loss_info.update(encoder_info)

        # Encode with teacher model for distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_latents = self.teacher_model.encode(encoder_input, return_info=False)
                loss_info['teacher_latents'] = teacher_latents

        # Optionally mask out some latents for noise resistance
        if self.latent_mask_ratio > 0.0:
            mask = torch.rand_like(latents) < self.latent_mask_ratio
            latents = torch.where(mask, torch.zeros_like(latents), latents)

        decoded = self.autoencoder.decode(latents)

        loss_info["decoded"] = decoded

        if self.autoencoder.out_channels == 2:
            loss_info["decoded_left"] = decoded[:, 0:1, :]
            loss_info["decoded_right"] = decoded[:, 1:2, :]
            loss_info["reals_left"] = reals[:, 0:1, :]
            loss_info["reals_right"] = reals[:, 1:2, :]

        # Distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_decoded = self.teacher_model.decode(teacher_latents)
                own_latents_teacher_decoded = self.teacher_model.decode(latents) #Distilled model's latents decoded by teacher
                teacher_latents_own_decoded = self.autoencoder.decode(teacher_latents) #Teacher's latents decoded by distilled model

                loss_info['teacher_decoded'] = teacher_decoded
                loss_info['own_latents_teacher_decoded'] = own_latents_teacher_decoded
                loss_info['teacher_latents_own_decoded'] = teacher_latents_own_decoded

       
        if self.warmed_up:
            loss_dis, loss_adv, feature_matching_distance = self.discriminator.loss(reals, decoded)
        else:
            loss_dis = torch.tensor(0.).to(reals)
            loss_adv = torch.tensor(0.).to(reals)
            feature_matching_distance = torch.tensor(0.).to(reals)

        loss_info["loss_dis"] = loss_dis
        loss_info["loss_adv"] = loss_adv
        loss_info["feature_matching_distance"] = feature_matching_distance
        
        return loss_info
    
    def training_step(self, batch, batch_idx):
        reals, _ = batch
        
        data_std = reals.std()
        
        opt_gen, opt_disc = self.get_optimizers()

        lr_schedulers = self.lr_schedulers()

        sched_gen = None
        sched_disc = None

        if lr_schedulers is not None:
            sched_gen, sched_disc = lr_schedulers

        log_dict = {}
        
        # Train the discriminator
        if self.warmed_up:
            self.toggle_optimizer(opt_disc)
            loss_info = self.general_step(reals)
            loss, losses = self.losses_disc(loss_info)

            log_dict.update({
                'train/disc_lr': opt_disc.param_groups[0]['lr']
            })
            
            for loss_name, loss_value in losses.items():
                log_dict[f'train/{loss_name}'] = loss_value.detach()

            opt_disc.zero_grad()
            self.manual_backward(loss)
            if self.global_step % 25 == 0:
                disc_grad_norms = grad_norm(self.discriminator, norm_type=2)
                disc_grad_norms = {f'disc_{k}': v for k, v in disc_grad_norms.items()}
                self.log_dict(disc_grad_norms, on_step=True)
            opt_disc.step()

            if sched_disc is not None:
                # sched step every step
                sched_disc.step()
            self.untoggle_optimizer(opt_disc)

        # Train the generator 
        self.toggle_optimizer(opt_gen)
        loss_info = self.general_step(reals)
        latents = loss_info['latents']
        loss, losses = self.losses_gen(loss_info)

        if self.use_ema:
            self.autoencoder_ema.update()

        opt_gen.zero_grad()
        self.manual_backward(loss)
        if self.global_step % 25 == 0:
            gen_grad_norms = grad_norm(self.autoencoder, norm_type=2)
            gen_grad_norms = {f'gen_{k}': v for k, v in gen_grad_norms.items()}
            self.log_dict(gen_grad_norms, on_step=True)
        if self.grad_clip_norm > 0.0:
            self.clip_gradients(opt_gen, self.grad_clip_norm, gradient_clip_algorithm='norm')
        opt_gen.step()

        if sched_gen is not None:
            # scheduler step every step
            sched_gen.step()
        self.untoggle_optimizer(opt_gen)
        
        log_dict.update({
            'train/loss': loss.detach(),
            'train/latent_std': latents.std().detach(),
            'train/data_std': data_std.detach(),
            'train/gen_lr': opt_gen.param_groups[0]['lr']
        })

        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss
    
    def export_model(self, path, use_safetensors=False):
        if self.autoencoder_ema is not None:
            model = self.autoencoder_ema.ema_model
        else:
            model = self.autoencoder
            
        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)
        

class AutoencoderDemoCallback(pl.Callback):
    def __init__(
        self, 
        demo_dl, 
        demo_every=2000,
        sample_size=65536,
        sample_rate=48000
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_samples = sample_size
        self.demo_dl = iter(demo_dl)
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        
        from pathlib import Path
        if not Path('demos').exists():
            Path('demos').mkdir()

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx): 
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        
        self.last_demo_step = trainer.global_step

        module.eval()

        try:
            demo_reals, _ = next(self.demo_dl)

            # Remove extra dimension added by WebDataset
            if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
                demo_reals = demo_reals[0]

            encoder_input = demo_reals
            
            encoder_input = encoder_input.to(module.device)

            if module.force_input_mono:
                encoder_input = encoder_input.mean(dim=1, keepdim=True)

            demo_reals = demo_reals.to(module.device)

            with torch.no_grad():
                if module.use_ema:

                    latents = module.autoencoder_ema.ema_model.encode(encoder_input)

                    fakes = module.autoencoder_ema.ema_model.decode(latents)
                else:
                    latents = module.autoencoder.encode(encoder_input)

                    fakes = module.autoencoder.decode(latents)

            #Interleave reals and fakes
            reals_fakes = rearrange([demo_reals, fakes], 'i b d n -> (b i) d n')

            # Put the demos together
            reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')

            log_dict = {}
            
            filename = f'demos/recon_{trainer.global_step:08}.wav'
            reals_fakes = reals_fakes.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, reals_fakes, self.sample_rate)

            log_dict[f'recon'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Reconstructed')
            
            log_dict[f'embeddings_3dpca'] = pca_point_cloud(latents)
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(latents))

            log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(reals_fakes))

            trainer.logger.experiment.log(log_dict)
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e
        finally:
            module.train()

def calculate_sdr(ref, est):
	s_true = ref
	s_artif = est - ref
	sdr = 10. * (
		np.log10(np.clip(np.mean(s_true ** 2), 1e-8, np.inf)) \
		- np.log10(np.clip(np.mean(s_artif ** 2), 1e-8, np.inf)))
	return sdr

def framewise_sdr(ref, est, frame_length=2*44100):
	sdrs = []
	for i in range(0, ref.shape[-1], frame_length):
		sdrs.append(calculate_sdr(ref[..., i:i+frame_length], est[..., i:i+frame_length]))
	return sdrs

def create_loss_modules_from_bottleneck(bottleneck, loss_config):
    losses = []
    
    if isinstance(bottleneck, VAEBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        try:
            kl_weight = loss_config['bottleneck']['weights']['kl']
        except:
            kl_weight = 1e-6

        kl_loss = ValueLoss(key='kl', weight=kl_weight, name='kl_loss')
        losses.append(kl_loss)

    if isinstance(bottleneck, RVQBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        quantizer_loss = ValueLoss(key='quantizer_loss', weight=1.0, name='quantizer_loss')
        losses.append(quantizer_loss)

    if isinstance(bottleneck, DACRVQBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck):
        codebook_loss = ValueLoss(key='vq/codebook_loss', weight=1.0, name='codebook_loss')
        commitment_loss = ValueLoss(key='vq/commitment_loss', weight=0.25, name='commitment_loss')
        losses.append(codebook_loss)
        losses.append(commitment_loss)

    if isinstance(bottleneck, WassersteinBottleneck):
        try:
            mmd_weight = loss_config['bottleneck']['weights']['mmd']
        except:
            mmd_weight = 100

        mmd_loss = ValueLoss(key='mmd', weight=mmd_weight, name='mmd_loss')
        losses.append(mmd_loss)
    
    return losses