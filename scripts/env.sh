#!/usr/bin/env bash
# source scripts/env.sh

_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${_ENV_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DEVKIT_ROOT}/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${NAVSIM_DEVKIT_ROOT}/exp}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${NAVSIM_DEVKIT_ROOT}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1

mkdir -p "${NAVSIM_EXP_ROOT}"
