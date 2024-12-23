from prefigure.prefigure import push_wandb_config
from omegaconf import OmegaConf
import os
import torch
import pytorch_lightning as pl
import random
import hydra

from icecream import install
install()

from mss.data.dataset import create_dataloader_from_config
from mss.models.autoencoder.stable_audio_tools.data.dataset import create_dataloader_from_config as create_ae_dataloader_from_config
from mss.models import create_model_from_config
from mss.models.utils import load_ckpt_state_dict
from mss.training import create_training_wrapper_from_config, create_validation_callback_from_config, create_demo_callback_from_config
from mss.training.utils import copy_state_dict

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

@hydra.main(version_base=None, config_path='.', config_name="default")
def main(cfg):
    seed = cfg.seed
    
    random.seed(seed)
    torch.manual_seed(seed)
    
    model_config = OmegaConf.load(cfg.model_config)
    dataset_config = OmegaConf.load(cfg.dataset_config)
    
    if model_config["model_type"] in ["autoencoder", "autoencoder_v2"]:
        train_dl = create_ae_dataloader_from_config(dataset_config, batch_size=cfg.batch_size, sample_size=model_config.sample_size, sample_rate=model_config.sample_rate, num_workers=cfg.num_workers)
    else:
        train_dl = create_dataloader_from_config(dataset_config["train"], batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    
    model = create_model_from_config(model_config)
    
    model_summary_callback = pl.callbacks.ModelSummary()
    
    training_wrapper = create_training_wrapper_from_config(model_config, model)
    
    wandb_logger = pl.loggers.WandbLogger(project=cfg.name, name=cfg.get("run_name", None))
    
    exc_callback = ExceptionCallback()
    
    if cfg.save_dir and isinstance(wandb_logger.experiment.id, str):
        checkpoint_dir = os.path.join(cfg.save_dir, wandb_logger.experiment.project, wandb_logger.experiment.id, "checkpoints") 
    else:
        checkpoint_dir = None
    
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=cfg.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
    
    args_dict = OmegaConf.to_container(cfg, resolve=True)
    args_dict.update({"model_config": OmegaConf.to_container(model_config, resolve=True)})
    args_dict.update({"dataset_config": OmegaConf.to_container(dataset_config, resolve=True)})
    push_wandb_config(wandb_logger, args_dict)
    
    if cfg.strategy:
        strategy = cfg.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if cfg.num_gpus > 1 else "auto"
    
    if model_config["model_type"] in ["autoencoder", "autoencoder_v2"]:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)
    else:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=create_dataloader_from_config(dataset_config["test"], batch_size=8, num_workers=cfg.num_workers))
    
    callbacks = [ckpt_callback, demo_callback, exc_callback, model_summary_callback, save_model_config_callback]
    
    if "validation" in model_config:
        callbacks += [create_validation_callback_from_config(model_config)]
    
    trainer = pl.Trainer(
        devices=cfg.num_gpus,
        accelerator="gpu",
        num_nodes = cfg.num_nodes,
        strategy=strategy,
        precision=cfg.precision,
        accumulate_grad_batches=cfg.accum_batches, 
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=cfg.get("max_epochs", None),
        max_steps=cfg.get("max_steps", None),
        default_root_dir=cfg.save_dir,
        gradient_clip_val=cfg.gradient_clip_val if 'autoencoder' not in model_config["model_type"] else None,
        use_distributed_sampler=False
    )
    
    trainer.fit(training_wrapper, train_dl, ckpt_path=cfg.ckpt_path if cfg.ckpt_path else None)

if __name__ == "__main__":
    main()