#!/usr/bin/env bash
# Front-only DrivoR pretraining.
#   GPUS_PER_NODE=8 bash scripts/training/train_drivor_front.sh
# Extra Hydra args are appended as-is.

set -euo pipefail
source "$(dirname "$0")/../env.sh"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
WORLD_GPUS=$((NNODES * GPUS_PER_NODE))

EXPERIMENT="${EXPERIMENT:-training_drivoR_Nav1_front_trainval_25epochs}"
RECIPE="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/config/training/train_drivoR_front.yaml"
GLOBAL_BATCH=$(python -c "from omegaconf import OmegaConf; print(int(OmegaConf.load('${RECIPE}').global_batch))")
PER_GPU_BATCH=$((GLOBAL_BATCH / WORLD_GPUS))

exec torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_full.py" \
    --config-name=train_drivoR_front \
    "experiment_name=${EXPERIMENT}" \
    "++trainer.params.devices=${GPUS_PER_NODE}" \
    "trainer.params.num_nodes=${NNODES}" \
    "dataloader.params.batch_size=${PER_GPU_BATCH}" \
    "agent.num_gpus=${WORLD_GPUS}" \
    "$@"
