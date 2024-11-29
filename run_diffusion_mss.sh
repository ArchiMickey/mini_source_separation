export WANDB_API_KEY=22e64bbabad135cc86cecaca0e4eef0df0dcd775

export HYDRA_FULL_ERROR=1
# export WANDB_MODE=offline

RUN_NAME="diffusion_mss"
python train.py \
    run_name=$RUN_NAME \
    batch_size=32 \
    num_workers=64 \
    num_gpus=8 \
    model_config=configs/model_configs/diffusion/diffusion_mss.yaml \
    dataset_config=configs/dataset_configs/musdb18hq.yaml