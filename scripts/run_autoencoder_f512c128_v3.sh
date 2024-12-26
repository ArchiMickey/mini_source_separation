export WANDB_API_KEY=22e64bbabad135cc86cecaca0e4eef0df0dcd775

export HYDRA_FULL_ERROR=1
# export WANDB_MODE=offline

BATCH_SIZE=8
ACCUM_BATCHES=1
NUM_WORKERS=16
NUM_GPUS=4

MODEL_CONFIG=configs/model_configs/autoencoder/stable_audio_2_0_ae_f512c128_v3.yaml
DATASET_CONFIG=configs/dataset_configs/autoencoder/musdb18hq_mixture-only.yaml

NAME=stable_audio_ae
RUN_NAME=stable_audio_ae_f512c128_v3

python train.py batch_size=$BATCH_SIZE accum_batches=$ACCUM_BATCHES num_workers=$NUM_WORKERS num_gpus=$NUM_GPUS precision=bf16-mixed \
    model_config=$MODEL_CONFIG \
    dataset_config=$DATASET_CONFIG \
    name=$NAME \
    run_name=$RUN_NAME \