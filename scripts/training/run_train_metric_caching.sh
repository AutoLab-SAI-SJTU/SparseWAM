#!/usr/bin/env bash
# Training-time PDMS cache -> ${NAVSIM_EXP_ROOT}/train_metric_cache
#   bash scripts/training/run_train_metric_caching.sh

set -euo pipefail
source "$(dirname "$0")/../env.sh"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_train_metric_caching.py" \
    "cache.cache_path=${NAVSIM_EXP_ROOT}/train_metric_cache"
