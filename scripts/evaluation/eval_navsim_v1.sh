#!/usr/bin/env bash
# NavSim v1 PDMS. Needs ${NAVSIM_EXP_ROOT}/metric_cache.
#   bash scripts/evaluation/eval_navsim_v1.sh /path/to/dino_dit.ckpt

set -euo pipefail
source "$(dirname "$0")/../env.sh"

CKPT="$1"
shift
EXPERIMENT="${EXPERIMENT:-dino_dit_nav1_eval}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export SUBSCORE_PATH="${NAVSIM_EXP_ROOT}"

exec python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_multi_gpu.py" \
    agent=dino_dit \
    "agent.checkpoint_path=${CKPT}" \
    agent.drivor_backbone_ckpt=null \
    "experiment_name=${EXPERIMENT}" \
    "++trainer.params.devices=${GPUS_PER_NODE}" \
    "$@"
