#!/usr/bin/env bash
# NavSim v1 PDMS. Needs ${NAVSIM_EXP_ROOT}/metric_cache.
# One trajectory:
#   GPUS_PER_NODE=8 bash scripts/evaluation/eval_navsim_v1.sh /path/to/sparsewam.ckpt
# 8 trajectories x (1+7) noise, ranked by a frozen front QueryEncoder Scorer:
#   GPUS_PER_NODE=8 bash scripts/evaluation/eval_navsim_v1.sh /path/to/sparsewam.ckpt /path/to/query_encoder.ckpt

set -euo pipefail
source "$(dirname "$0")/../env.sh"

CKPT="$1"
shift
SCORER_CKPT=""
if [[ $# -gt 0 && "$1" != *=* ]]; then
    SCORER_CKPT="$1"
    shift
fi

EXPERIMENT="${EXPERIMENT:-sparsewam_nav1_eval}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export SUBSCORE_PATH="${NAVSIM_EXP_ROOT}"

# Single-sample protocol. The Scorer branch below is the 64-candidate protocol.
SCORER_ARGS=(
    agent.config.use_drivor_scorer=false
    agent.config.num_proposals=1
    agent.config.proposal_mul_noise_copies=0
    agent.drivor_scorer_ckpt=null
)
if [[ -n "${SCORER_CKPT}" ]]; then
    SCORER_ARGS=(
        agent.config.use_drivor_scorer=true
        agent.config.num_proposals=8
        agent.config.proposal_mul_noise_copies=7
        agent.config.proposal_mul_noise_std=0.1
        agent.config.proposal_chunk_size=32
        "agent.config.scorer_camera_order=[cam_f0]"
        "agent.drivor_scorer_ckpt=\"${SCORER_CKPT}\""
        dataloader.params.batch_size=4
    )
fi

exec python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_multi_gpu.py" \
    agent=sparsewam \
    "agent.checkpoint_path=\"${CKPT}\"" \
    agent.drivor_backbone_ckpt=null \
    "experiment_name=${EXPERIMENT}" \
    "++trainer.params.devices=${GPUS_PER_NODE}" \
    "${SCORER_ARGS[@]}" \
    "$@"
