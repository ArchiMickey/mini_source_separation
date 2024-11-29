export WANDB_API_KEY=22e64bbabad135cc86cecaca0e4eef0df0dcd775

export HYDRA_FULL_ERROR=1
# export WANDB_MODE=offline

export CUDA_VISIBLE_DEVICES=2,3,4,5

BATCH_SIZE=8
ACCUM_BATCHES=1
NUM_WORKERS=32
NUM_GPUS=4

MODEL_CONFIG=configs/model_configs/bsroformer/bsroformer_dev.yaml
DATASET_CONFIG=configs/dataset_configs/musdb18hq.yaml

RUN_NAME=bsroformer_dev

python train.py batch_size=$BATCH_SIZE accum_batches=$ACCUM_BATCHES num_workers=$NUM_WORKERS num_gpus=$NUM_GPUS \
    model_config=$MODEL_CONFIG \
    dataset_config=$DATASET_CONFIG \
    run_name=$RUN_NAME