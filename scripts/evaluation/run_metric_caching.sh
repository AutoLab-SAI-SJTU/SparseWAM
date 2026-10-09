#!/usr/bin/env bash
# NavSim v1 navtest cache -> ${NAVSIM_EXP_ROOT}/metric_cache
#   bash scripts/evaluation/run_metric_caching.sh

set -euo pipefail
source "$(dirname "$0")/../env.sh"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
    train_test_split=navtest \
    "cache.cache_path=${NAVSIM_EXP_ROOT}/metric_cache"
