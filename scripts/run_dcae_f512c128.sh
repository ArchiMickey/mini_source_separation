cd mss/third_party/stable_audio_tools

export WANDB_API_KEY=22e64bbabad135cc86cecaca0e4eef0df0dcd775
# export WANDB_MODE=offline

export HYDRA_FULL_ERROR=1

BATCH_SIZE=8
ACCUM_BATCHES=1
NUM_WORKERS=32
NUM_GPUS=4

MODEL_CONFIG=/root/autodl-tmp/mini_source_separation/mss/third_party/stable_audio_tools/stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_dcae_f512c128.json
DATASET_CONFIG=/root/autodl-tmp/mini_source_separation/mss/third_party/stable_audio_tools/stable_audio_tools/configs/dataset_configs/musdb18hq_mixture-only.json

python train.py \
    --precision bf16-mixed \
    --dataset-config $DATASET_CONFIG \
    --model-config $MODEL_CONFIG \
    --name musdb18hq_stable_audio_2_0_dcae \
    --batch-size $BATCH_SIZE \
    --num-gpus $NUM_GPUS