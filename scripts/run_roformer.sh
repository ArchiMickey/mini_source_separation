export WANDB_API_KEY=22e64bbabad135cc86cecaca0e4eef0df0dcd775

export HYDRA_FULL_ERROR=1
# export WANDB_MODE=offline

BATCH_SIZE=4
ACCUM_BATCHES=1
NUM_WORKERS=64
DEVICES=4

MODEL_CONFIG=configs/model_configs/bsroformer/bsroformer_bsunet2_res.yaml
DATASET_CONFIG=configs/dataset_configs/musdb18hq.yaml

RUN_NAME=bsroformer_bsuent2_res

python train.py batch_size=$BATCH_SIZE accum_batches=$ACCUM_BATCHES num_workers=$NUM_WORKERS devices=$DEVICES strategy=ddp \
    model_config=$MODEL_CONFIG \
    dataset_config=$DATASET_CONFIG \
    run_name=$RUN_NAME