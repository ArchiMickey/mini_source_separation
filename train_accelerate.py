from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm


import wandb
# import trackio as wandb
from mss.utils import parse_yaml, requires_grad, update_ema
from train import get_dataset, get_model, get_optimizer_and_scheduler, get_sampler, validate
from mss.losses import get_loss_fn


def train(args) -> None:
    r"""Train a music source separation system."""

    # Arguments
    config_path = args.config
    wandb_log = not args.no_log
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    precision = configs["train"]["precision"]
    valid_num = configs["validate"]["audios_num"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

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
    max_grad_norm = configs["train"].get("max_grad_norm", 1e10)

    # Prepare for acceleration
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    
    accelerator = Accelerator(
        log_with="wandb" if wandb_log else None,
        mixed_precision=precision, 
        kwargs_handlers=[process_group_kwargs],
        step_scheduler_with_optimizer=False
    )
    
    if accelerator.is_main_process:
        print(f"Mixed precision: {precision}")
    
    # EMA
    if accelerator.is_main_process:
        ema = deepcopy(model)
        requires_grad(ema, False)
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()  # EMA model should always be in eval mode
        ema.to(accelerator.device)
        
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader)

    # Logger
    if wandb_log:
        accelerator.init_trackers(
            project_name="mss",
            init_kwargs={
                "wandb": {"name": configs.get("name", filename)}
            },
            config=configs,
        )

    # Train
    pbar = tqdm(train_dataloader, disable=not accelerator.is_main_process)    
    for step, data in enumerate(pbar):
        # ------ 1. Training ------
        # 1.1 Data
        target = data["target"]
        mixture = data["mixture"]

        # 1.1 Forward
        model.train()
        output = model(mixture)

        # 1.2 Loss
        loss, loss_info = loss_fn(output=output, target=target)
        
        # 1.3 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        accelerator.backward(loss)  # Update all parameter.grad
        
        if accelerator.sync_gradients:
            if max_grad_norm is not None:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), max_grad_norm)
                if hasattr(grad_norm, "item"):
                    grad_norm = grad_norm.item()
        
        optimizer.step()  # Update all parameters based on all parameter.grad
        scheduler.step()
        if accelerator.sync_gradients and accelerator.is_main_process:
            update_ema(ema, model, decay=0.999)

        pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm, **loss_info})
        accelerator.log(dict(
            loss=loss.item(),
            lr=scheduler.get_last_lr()[0],
            grad_norm=grad_norm,
            **loss_info
        ), step=step)
        
        # ------ 2. Evaluation ------
        # 2.1 Evaluate
        if (step + 1) % configs["train"]["test_every_n_steps"] == 0 and accelerator.is_main_process:

            train_sdr = validate(
                configs=configs,
                model=ema,
                split="train",
                audios_num=valid_num,
            )

            test_sdr = validate(
                configs=configs,
                model=ema,
                split="test",
                audios_num=valid_num,
            )

            accelerator.log(
                {
                    "train_sdr": train_sdr, 
                    "test_sdr": test_sdr,
                },
                step=step
            )

            print("====== Overall metrics ====== ")
            print(f"Train SDR fast: {train_sdr:.2f}")
            print(f"Test SDR fast: {test_sdr:.2f}")
        
        
        # 2.2 Save model
        if (step + 1) % configs["train"]["save_every_n_steps"] == 0 and accelerator.is_main_process:
            ckpt_path = Path(ckpts_dir, f"step={step}_ema.pth")
            torch.save(ema.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if (step + 1) == configs["train"]["training_steps"]:
            break
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)