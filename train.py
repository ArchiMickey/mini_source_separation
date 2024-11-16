from prefigure.prefigure import push_wandb_config
from omegaconf import OmegaConf
import os
import torch
import pytorch_lightning as pl
import random
import hydra

from icecream import install
install()

from data.dataset import create_dataloader_from_config
from models import create_model_from_config
from models.utils import load_ckpt_state_dict
from training import create_training_wrapper_from_config, create_validation_callback_from_config, create_demo_callback_from_config
from training.utils import copy_state_dict

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
    
    train_dl = create_dataloader_from_config(dataset_config["train"], sample_rate=model_config["sample_rate"], batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    
    model = create_model_from_config(model_config)
    
    training_wrapper = create_training_wrapper_from_config(model_config, model)
    
    wandb_logger = pl.loggers.WandbLogger(project=cfg.name, name=cfg.get("run_name", None))
    wandb_logger.watch(training_wrapper)
    
    exc_callback = ExceptionCallback()
    
    if cfg.save_dir and isinstance(wandb_logger.experiment.id, str):
        checkpoint_dir = os.path.join(cfg.save_dir, wandb_logger.experiment.project, wandb_logger.experiment.id, "checkpoints") 
    else:
        checkpoint_dir = None
    
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=cfg.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
    
    args_dict = OmegaConf.to_container(cfg, resolve=True)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    push_wandb_config(wandb_logger, args_dict)
    
    if cfg.strategy:
        strategy = cfg.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if cfg.num_gpus > 1 else "auto"
        
    demo_callback = create_demo_callback_from_config(model_config, demo_dl=create_dataloader_from_config(dataset_config["test"], sample_rate=model_config["sample_rate"], batch_size=8, num_workers=cfg.num_workers))
    
    callbacks = [ckpt_callback, demo_callback, exc_callback, save_model_config_callback]
    
    if cfg.get("validation", None):
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
        gradient_clip_val=cfg.gradient_clip_val,
        use_distributed_sampler=False
    )
    
    trainer.fit(training_wrapper, train_dl, ckpt_path=cfg.ckpt_path if cfg.ckpt_path else None)

if __name__ == "__main__":
    main()