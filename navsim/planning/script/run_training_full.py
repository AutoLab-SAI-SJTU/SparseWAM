import os
import random
from typing import Tuple
from pathlib import Path
import logging
import pickle
from datetime import datetime

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.distributed as dist
import pytorch_lightning as pl

# DataLoader workers serialize tensors through /dev/shm by default. On 8-GPU
# DDP with many workers per rank, this can exhaust shared memory and crash
# workers with SIGBUS ("Bus error"). Switch to file_system strategy which
# routes tensor sharing through file descriptors instead.
try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

# Official navtest scene filter (same as run_pdm_score / eval_navsim_v1).
# File lives next to this script under config/common/...
_NAVTEST_SCENE_FILTER_YAML = (
    Path(__file__).resolve().parent
    / "config"
    / "common"
    / "train_test_split"
    / "scene_filter"
    / "navtest.yaml"
)


def dist_ready():
    return dist.is_available() and dist.is_initialized()


def _resolve_openscene_split_paths(data_split: str) -> Tuple[Path, Path]:
    """Resolve navsim log / sensor blob roots for a data split.

    Args:
        data_split: OpenScene split name, e.g. ``trainval`` or ``test``.

    Returns:
        ``(navsim_log_path, sensor_blobs_path)``.
    """
    root = os.environ.get("OPENSCENE_DATA_ROOT", "")
    if not root:
        raise ValueError(
            "OPENSCENE_DATA_ROOT must be set when building online datasets."
        )
    return (
        Path(root) / "navsim_logs" / data_split,
        Path(root) / "sensor_blobs" / data_split,
    )


def _build_navtest_scene_filter(cfg: DictConfig) -> SceneFilter:
    """Build the official navtest ``SceneFilter`` for validation / PDMS.

    Keeps the same token whitelist as ``train_test_split=navtest`` eval, and
    copies ``num_history_frames`` from the training override when present so
    SparseWAM history windows stay aligned (default training uses 4).

    Args:
        cfg: Hydra training config.

    Returns:
        Instantiated navtest ``SceneFilter``.
    """
    if not _NAVTEST_SCENE_FILTER_YAML.is_file():
        raise FileNotFoundError(
            f"navtest scene filter not found: {_NAVTEST_SCENE_FILTER_YAML}"
        )
    val_filter_cfg = OmegaConf.load(_NAVTEST_SCENE_FILTER_YAML)
    # Align history window with training (e.g. SparseWAM needs >=5 for -2s).
    hist = OmegaConf.select(
        cfg, "train_test_split.scene_filter.num_history_frames"
    )
    if hist is not None:
        val_filter_cfg.num_history_frames = int(hist)
    explicit_hist = OmegaConf.select(cfg, "val_num_history_frames")
    if explicit_hist is not None:
        val_filter_cfg.num_history_frames = int(explicit_hist)
    return instantiate(val_filter_cfg)


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config.

    When ``cfg.val_on_test`` is true, validation uses the official **navtest**
    split (``data_split=test`` + navtest scene filter), matching closed-loop
    PDMS evaluation. Training remains on ``cfg.train_test_split`` (navtrain).

    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    
    print("Train without caching....")
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs or log_name in cfg.val_logs 
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs + cfg.val_logs
    

    print("len(train_scene_filter.log_names) ", len(train_scene_filter.log_names))

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_on_test = bool(cfg.get("val_on_test", False))
    if val_on_test:
        # Match scripts/evaluation/eval_navsim_v1.sh (navtest / test).
        val_scene_filter = _build_navtest_scene_filter(cfg)
        val_data_path, val_sensor_blobs_path = _resolve_openscene_split_paths("test")
        # Allow explicit path overrides for custom mounts.
        if cfg.get("val_navsim_log_path"):
            val_data_path = Path(cfg.val_navsim_log_path)
        if cfg.get("val_sensor_blobs_path"):
            val_sensor_blobs_path = Path(cfg.val_sensor_blobs_path)
        logger.info(
            "Validation uses navtest (test split): logs=%s sensors=%s "
            "num_history_frames=%s log_names=%s",
            val_data_path,
            val_sensor_blobs_path,
            val_scene_filter.num_history_frames,
            (
                len(val_scene_filter.log_names)
                if val_scene_filter.log_names is not None
                else "all"
            ),
        )
        print(
            f"[val_on_test] navtest logs={val_data_path} "
            f"sensors={val_sensor_blobs_path} "
            f"history={val_scene_filter.num_history_frames} "
            f"n_logs={len(val_scene_filter.log_names) if val_scene_filter.log_names else 'all'}"
        )
    else:
        val_scene_filter = instantiate(cfg.train_test_split.scene_filter)
        if val_scene_filter.log_names is not None:
            val_scene_filter.log_names = [
                log_name
                for log_name in val_scene_filter.log_names
                if log_name in cfg.val_logs
            ]
        else:
            val_scene_filter.log_names = cfg.val_logs
        val_data_path = data_path
        val_sensor_blobs_path = sensor_blobs_path

    val_scene_loader = SceneLoader(
        sensor_blobs_path=val_sensor_blobs_path,
        data_path=val_data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        if bool(cfg.get("val_on_test", False)):
            raise ValueError(
                "val_on_test=true requires online SceneLoader (navtest / test "
                "split). Set use_cache_without_dataset=false and cache_path=null."
            )
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True,drop_last=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False,drop_last=True)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")

    # automatically resume training
    # find latest ckpt
    import glob
    def find_latest_checkpoint(search_pattern):
        # List all files matching the pattern
        list_of_files = glob.glob(search_pattern, recursive=True)
        # Find the file with the latest modification time
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file


    if cfg.train_ckpt_path is None and not bool(cfg.get("skip_ckpt_auto_resume", False)):
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = "/".join(str(cfg.output_dir).split("/")[:-1]) + "/*/lightning_logs/version_*/checkpoints/" + '*.ckpt'
        print("/".join(str(cfg.output_dir).split("/")[:-1]))
        print("search_pattern ", search_pattern)
        cfg.train_ckpt_path = find_latest_checkpoint(search_pattern)
        print("cfg.train_ckpt_path ", cfg.train_ckpt_path)
    elif bool(cfg.get("skip_ckpt_auto_resume", False)):
        print("[train] skip_ckpt_auto_resume=true; Lightning will not resume optimizer/epoch")
        cfg.train_ckpt_path = None

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    if cfg.validation_run:
        logger.info("Starting Validation")
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        dump_root = os.path.join(os.getenv('SUBSCORE_PATH'), "navsim1_pdm_scores", cfg.experiment_name)
        os.makedirs(dump_root, exist_ok=True)
        dump_path = os.path.join(dump_root, f"{timestamp}.pkl")
        trainer.validate(
            model=lightning_module,
            dataloaders=[val_dataloader],
            ckpt_path=cfg.train_ckpt_path,
            verbose=True
        )
        logger.info("Running predictions to collect trajectories")
        predictions = trainer.predict(
            AgentLightningModule(agent=agent, for_viz=True),
            val_dataloader,
            return_predictions=True
        )

        if dist_ready():
            dist.barrier()
        
        world_size = dist.get_world_size() if dist_ready() else 1
        all_predictions = [None for _ in range(world_size)]

        if dist_ready():
            dist.all_gather_object(all_predictions, predictions)
        else:
            all_predictions = [predictions]

        rank = dist.get_rank() if dist_ready() else 0
        if rank != 0:
            return None

        merged_predictions = {}
        for proc_prediction in all_predictions:
            for d in proc_prediction:
                merged_predictions.update(d)

        pickle.dump(predictions, open(dump_path, 'wb'))
    else:
        logger.info("Starting Training")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.train_ckpt_path
        )


if __name__ == "__main__":
    main()