# SparseWAM

<!-- TODO: paper title, authors, and affiliations. -->

[**Paper**](TODO) | [**Project**](TODO) | [**Models**](TODO)

## News

- **`October 9th, 2026`:** NavSim v1 training and evaluation code released.

## Table of Contents

- [News](#news)
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Preparation](#model-preparation)
- [Training](#training)
- [NavSim v1 Evaluation](#navsim-v1-evaluation)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## Introduction

SparseWAM is a world action model that plans in a sparse register space. A frozen DINOv2 encoder is compressed by 16 learnable register tokens per frame. A lightweight Mixture-of-Transformers denoiser then models the joint distribution of future registers and the ego trajectory with rectified flow. Modality-specific parameters keep the two streams from overwriting each other, while a causal attention mask lets action tokens read the denoised future registers. Future registers do not read action tokens, so inference follows an imagine-then-act order.

This repository releases that NavSim v1 path:

1. Pretrain a front-camera DrivoR backbone.
2. Initialize DinoDiT from that checkpoint and pretrain the world-action model.
3. Evaluate the DinoDiT checkpoint with NavSim v1 PDMS.

<!-- TODO: teaser figure and the paper's reported PDMS. -->

## Installation

Recommended environment, matching the stack used to develop this code:

- Linux
- Python 3.9
- PyTorch 2.1.0 with CUDA 12.1
- PyTorch Lightning 2.2.1
- A CUDA GPU

```bash
git clone https://github.com/AutoLab-SAI-SJTU/SparseWAM.git
cd SparseWAM

conda create -n navsim python=3.9 -y
conda activate navsim

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "pytorch-lightning==2.2.1"
pip install -e ./nuplan-devkit
pip install -e .
```

`requirements.txt` lists PyTorch Lightning 2.4.0. Install 2.2.1 afterwards so the environment matches the EMA callback used in training.

`scripts/env.sh` exports the paths the launchers read. The defaults place data and experiment outputs inside this repository:

```text
SparseWAM/
├── dataset/
│   ├── maps/nuplan-maps-v1.0
│   ├── navsim_logs/{trainval,test}
│   └── sensor_blobs/{trainval,test}
├── weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors
└── exp/
```

Override any of these before sourcing the script:

```bash
export OPENSCENE_DATA_ROOT=/path/to/dataset
export NUPLAN_MAPS_ROOT=/path/to/dataset/maps/nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=/path/to/exp
source scripts/env.sh
```

## Data Preparation

Download the nuPlan maps and the OpenScene logs and sensor blobs used by NavSim. See [docs/install.md](docs/install.md) and [docs/splits.md](docs/splits.md). The launchers train on the `navtrain` filter of `trainval` and evaluate on the `navtest` filter of `test`.

```bash
cd download
bash download_maps.sh
bash download_trainval.sh
bash download_test.sh
```

Move the extracted files into `dataset/` so the layout above is satisfied, then return to the repository root.

DinoDiT reads the front camera at four timestamps: 1 s of history, the current frame, and futures at +2 s and +4 s. `download_navtrain.sh` only fetches current and history sensors. Use the full `trainval` sensor blobs for DinoDiT so those future front-camera frames are present. The front-camera DrivoR stage reads the current front image and can train from the smaller navtrain sensor package.

## Model Preparation

Download the DINOv2 ViT-S/14 register weights and keep the filename expected by the configs:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download timm/vit_small_patch14_reg4_dinov2.lvd142m model.safetensors \
    --local-dir weights/vit_small_patch14_reg4_dinov2.lvd142m
```

The file must be `weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors`.

<!-- TODO: public DrivoR and DinoDiT checkpoints, and the command to download them. -->

## Training

Training is two stages. Checkpoints are written to:

```text
$NAVSIM_EXP_ROOT/ke/<experiment_name>/<timestamp>/lightning_logs/version_*/checkpoints/
```

`global_batch` lives in the recipe yaml. Each launcher sets the per-GPU batch to `global_batch / (NNODES * GPUS_PER_NODE)`. Use a GPU count that divides the global batch: 64 for DrivoR, 32 for DinoDiT.

`GPUS_PER_NODE` defaults to 1. For several machines, set `NNODES`, `NODE_RANK`, and `MASTER_ADDR` on every node. `MASTER_PORT` defaults to `29500`.

### 1. Front-camera DrivoR

The recipe is `navsim/planning/script/config/training/train_drivoR_front.yaml`: 25 epochs, global batch 64, seed 2, online `navtrain` frames, one front camera. Model settings, including the learning rate, are in `navsim/planning/script/config/common/agent/drivoR_front.yaml`.

```bash
source scripts/env.sh
GPUS_PER_NODE=8 bash scripts/training/train_drivor_front.sh
```

Eight GPUs give a per-GPU batch of 8. The default experiment name is `training_drivoR_Nav1_front_trainval_25epochs`. Override it with `EXPERIMENT=my_run`.

### 2. Training metric cache

DinoDiT scores validation PDMS every 5 epochs. Build this cache before that run. The cache does not change the training loss. A missing cache only skips those scores.

```bash
source scripts/env.sh
bash scripts/training/run_train_metric_caching.sh
```

The cache is stored at `$NAVSIM_EXP_ROOT/train_metric_cache`.

### 3. DinoDiT

Pass the DrivoR checkpoint from stage 1. The recipe is `navsim/planning/script/config/training/train_dino_dit.yaml`: 50 epochs, global batch 32, seed 2, online frames, validation PDMS every 5 epochs. Architecture settings are in `navsim/planning/script/config/common/agent/dino_dit.yaml`.

```bash
source scripts/env.sh
GPUS_PER_NODE=8 bash scripts/training/train_dinodit.sh /path/to/drivor.ckpt
```

Eight GPUs give a per-GPU batch of 4. The default experiment name is `training_dino_dit_mot_v3_a1_50epochs`.

The DrivoR backbone is loaded from `agent.drivor_backbone_ckpt` and kept frozen. DinoDiT trains the register compression and the rectified-flow denoiser on top of it.

### Multi-node

Run the same command on every node. Only `NODE_RANK` changes:

```bash
source scripts/env.sh
NNODES=2 NODE_RANK=0 MASTER_ADDR=<rank0-host> GPUS_PER_NODE=8 \
    bash scripts/training/train_dinodit.sh /path/to/drivor.ckpt
```

Extra Hydra arguments can be appended to either training command.

## NavSim v1 Evaluation

Evaluation reads the navtest metric cache.

```bash
source scripts/env.sh
bash scripts/evaluation/run_metric_caching.sh
```

The cache is stored at `$NAVSIM_EXP_ROOT/metric_cache`.

Score a DinoDiT checkpoint with:

```bash
source scripts/env.sh
GPUS_PER_NODE=8 bash scripts/evaluation/eval_navsim_v1.sh /path/to/dino_dit.ckpt
```

The script sets `SUBSCORE_PATH` to `$NAVSIM_EXP_ROOT`. The default experiment name is `dino_dit_nav1_eval`. PDMS is written under `$NAVSIM_EXP_ROOT/ke/dino_dit_nav1_eval/<timestamp>/`.

This evaluation uses one trajectory sample. The scorer checkpoint stays unset.

## Configuration

| File | What it controls |
|---|---|
| `scripts/env.sh` | `NAVSIM_DEVKIT_ROOT`, `OPENSCENE_DATA_ROOT`, `NUPLAN_MAPS_ROOT`, `NAVSIM_EXP_ROOT` |
| `navsim/planning/script/config/training/train_drivoR_front.yaml` | DrivoR epochs, global batch, workers, seed |
| `navsim/planning/script/config/common/agent/drivoR_front.yaml` | Front-camera DrivoR model and optimizer |
| `navsim/planning/script/config/training/train_dino_dit.yaml` | DinoDiT epochs, global batch, validation PDMS |
| `navsim/planning/script/config/common/agent/dino_dit.yaml` | DinoDiT architecture, flow shifts, cameras |

Launchers only pass the experiment name, GPU count, per-GPU batch, and, for DinoDiT, the checkpoint path. Change hyperparameters in the yaml files.

## Acknowledgments

SparseWAM builds on [NAVSIM](https://github.com/autonomousvision/navsim), [nuPlan](https://github.com/motional/nuplan-devkit), [DrivoR](https://github.com/valeoai/DrivoR), and [DINOv2](https://github.com/facebookresearch/dinov2).

## Citation

<!-- TODO: replace with the paper bibtex. -->

```bibtex
@article{sparsewam2026,
  title={SparseWAM},
  author={TODO},
  year={2026}
}
```
