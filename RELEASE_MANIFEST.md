# 拷贝清单

源：`dino` @ `61d5254`。目标：`SparseWAM-code`。
下表按源目录顺序排列。整目录处理一致时只写目录一行，不再展开里面的文件。

| 路径 | 处理 | 说明 |
|---|---|---|
| `environment.yml` | 原样保留 | |
| `env_setup.sh` | 拷贝后删除 | 会激活 conda、加载 module |
| `env_setup_tidal.sh` | 拷贝后删除 | 潮汐盘、conda-pack、`/workspace/output` |
| `eval.sh` | 未移动 | |
| `.gitignore` | 原样保留 | |
| `LICENSE` | 原样保留 | |
| `metric_caching.sh` | 未移动 | |
| `pyrightconfig.json` | 原样保留 | |
| `README.md` | 新写 | NavSim v1 两阶段训练与评测说明 |
| `requirements.txt` | 原样保留 | |
| `setup.py` | 原样保留 | |
| `SparseWAM_yixiang_ICLR2027 (9).pdf` | 未移动 | |
| `weights` | 未移动 | 指向数据盘的软链接 |
| `assets/` | 未移动 | |
| `dataset/` | 未移动 | |
| `docs/agents.md` | 未移动 | |
| `docs/cache.md` | 原样保留 | |
| `docs/dino_dit.md` | 拷贝后修改 | 见下方改动 7 |
| `docs/dino_dit_mot.md` | 拷贝后修改 | 见下方改动 8 |
| `docs/install.md` | 原样保留 | |
| `docs/metrics.md` | 未移动 | |
| `docs/splits.md` | 原样保留 | |
| `docs/submission.md` | 未移动 | |
| `download/` | 未移动 | |
| `exp/` | 未移动 | |
| `figures/` | 未移动 | |
| `lightning_logs/` | 未移动 | |
| `navsim/__init__.py` | 原样保留 | |
| `navsim/agents/__init__.py` | 原样保留 | |
| `navsim/agents/abstract_agent.py` | 原样保留 | |
| `navsim/agents/constant_velocity_agent.py` | 未移动 | |
| `navsim/agents/dino_dit/` | 原样保留 | 不含下面的 `grpo/` |
| `navsim/agents/dino_dit/grpo/` | 未移动 | |
| `navsim/agents/dino_dit/layers/ema.py` | 拷贝后修改 | 见下方改动 9 |
| `navsim/agents/dino_dit_action/` | 未移动 | |
| `navsim/agents/dino_dit_pool/` | 未移动 | |
| `navsim/agents/dino_dit_scorer/` | 未移动 | |
| `navsim/agents/drivoR/` | 原样保留 | |
| `navsim/agents/ego_status_mlp_agent.py` | 未移动 | |
| `navsim/agents/human_agent.py` | 未移动 | |
| `navsim/common/` | 原样保留 | |
| `navsim/evaluate/` | 原样保留 | |
| `navsim/planning/` | 原样保留 | 含训练、打分、metric cache 的 Python |
| `navsim/planning/script/config/common/agent/drivoR_front.yaml` | 拷贝后修改 | 见下方改动 1 |
| `navsim/planning/script/config/common/agent/dino_dit.yaml` | 拷贝后修改 | 见下方改动 2 |
| `navsim/planning/script/config/training/train_drivoR_front.yaml` | 新写 | 见下方改动 3 |
| `navsim/planning/script/config/training/train_dino_dit.yaml` | 新写 | 见下方改动 4 |
| `navsim/planning/script/run_training_full.py` | 拷贝后修改 | 见下方改动 10 |
| `navsim/visualization/` | 未移动 | |
| `nuplan-devkit/` | 原样保留 | 整目录，未再拆文件 |
| `register_probe/` | 未移动 | |
| `scripts/evaluation/run_metric_caching.sh` | 保留 | 改成走 `scripts/env.sh`，输出 `${NAVSIM_EXP_ROOT}/metric_cache` |
| `scripts/evaluation/` 其余脚本 | 未移动 | Nav2、可视化、baseline 评测 |
| `scripts/helper/` | 未移动 | |
| `scripts/submission/` | 未移动 | |
| `scripts/training/run_train_metric_caching.sh` | 保留 | 改成走 `scripts/env.sh`，输出 `${NAVSIM_EXP_ROOT}/train_metric_cache` |
| `scripts/training/` 其余脚本 | 未移动 | 潮汐 / SimScale / 其他 agent 的训练入口 |
| `scripts/viz/` | 未移动 | |
| `scripts/env.sh` | 新写 | 见下方改动 6 |
| `scripts/training/train_drivor_front.sh` | 新写 | 见下方改动 6 |
| `scripts/training/train_dinodit.sh` | 新写 | 见下方改动 6 |
| `scripts/evaluation/eval_navsim_v1.sh` | 新写 | 见下方改动 6 |
| `tutorial/` | 未移动 | |
| `.git/` | 未移动 | |
| `.tidal/` | 未移动 | |

临时拷进副本、核对完 shell 覆盖后又删掉的文件：

| 路径 | 处理 |
|---|---|
| `env_setup.sh` | 拷贝后删除 |
| `env_setup_tidal.sh` | 拷贝后删除 |
| `scripts/training/run_drivoR_front_tidal_multinode_dlc.sh` | 拷贝后删除 |
| `scripts/training/run_dino_dit_tidal_multinode_dlc.sh` | 拷贝后删除 |
| `scripts/evaluation/run_dino_dit_nav1_eval.sh` | 拷贝后删除 |
| `scripts/evaluation/run_drivoR_nav1_eval.sh` | 拷贝后删除 |

## 改动

1. `navsim/planning/script/config/common/agent/drivoR_front.yaml`
   - `long_trajectory_additional_poses`：`-1` 改为 `2`
   - `lr_args.base_lr`：`5e-4` 改为 `0.0002`
   - `scheduler_args.dataset_size`：`85000` 改为 `103000`
   - `progress_bar`：`true` 改为 `false`
2. `navsim/planning/script/config/common/agent/dino_dit.yaml`
   - 只加了一段注释，说明脚本不再重复这些字段。数值没改，仍是 v3_a1。
3. 新写 `navsim/planning/script/config/training/train_drivoR_front.yaml`
   - 独立的 Hydra 主配置，选用 `agent=drivoR_front`
   - `seed=2`，`cache_path=null`，`use_cache_without_dataset=false`
   - `global_batch=64`，`max_epochs=25`，`num_workers=8`，`prefetch_factor=4`
4. 新写 `navsim/planning/script/config/training/train_dino_dit.yaml`
   - 独立的 Hydra 主配置，选用 `agent=dino_dit`
   - `seed=2`，`cache_path=null`，`use_cache_without_dataset=false`
   - `global_batch=32`，`max_epochs=50`，`num_workers=8`，`prefetch_factor=2`
   - `agent.eval_pdm=true`，`eval_pdm_every_n_epochs=5`
5. 删掉潮汐和旧入口
   - `env_setup.sh`、`env_setup_tidal.sh`
   - 上面表格里 6 个旧 shell
6. 新写四个脚本，不再写 `cam_f0`、head 数、flow shift
   - `scripts/env.sh`：导出四个路径，并带上 `PYTHONPATH`
   - `scripts/training/train_drivor_front.sh`：`torchrun`，只传实验名、GPU 数、按 `global_batch` 算出的 per-GPU batch
   - `scripts/training/train_dinodit.sh`：同上，另外传入 DrivoR backbone checkpoint
   - `scripts/evaluation/eval_navsim_v1.sh`：只传 checkpoint、实验名、GPU 数
   - 后来又收掉了目录检查、checkpoint 存在性检查和 batch 整除报错
11. 保留两个 metric cache 脚本，只把环境从 `env_setup.sh` 换成 `scripts/env.sh`
   - `scripts/training/run_train_metric_caching.sh` → `${NAVSIM_EXP_ROOT}/train_metric_cache`
   - `scripts/evaluation/run_metric_caching.sh` → `${NAVSIM_EXP_ROOT}/metric_cache`
7. `docs/dino_dit.md`
   - 删掉潮汐 / DLC 启动说明和 `env_setup_tidal.sh`
   - 改成 `scripts/training/train_dinodit.sh`
8. `docs/dino_dit_mot.md`
   - 表里 `history_frame_offsets` 从 `[-3]` 改为 `[-2]`
   - 表里 head 从 `8×64=512` 改为 `12×64=768`
   - 启动命令改成 `scripts/training/train_dinodit.sh` 和 `scripts/evaluation/eval_navsim_v1.sh`
   - 文件表里的潮汐 shell、smoke、可视化脚本改成上面两个入口
9. `navsim/agents/dino_dit/layers/ema.py`
   - 注释里的 “tidal env is 2.2.1” 改成 “this stack uses 2.2.1”。逻辑没改。
10. `navsim/planning/script/run_training_full.py`
    - 注释里的评测脚本路径改成 `scripts/evaluation/eval_navsim_v1.sh`
    - 注释里的 “Tidal / custom mounts” 改成 “custom mounts”。逻辑没改。
