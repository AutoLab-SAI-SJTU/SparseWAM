"""NAVSIM agent wrapper for DinoDiT."""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from .dino_dit_features import DinoDiTFeatureBuilder, DinoDiTTargetBuilder
from .dino_dit_model import DinoDiTModel
from .layers.ema import AdaptiveEMAWeightAveraging
from .layers.loss import DinoDiTLoss


class DinoDiTAgent(AbstractAgent):
    """Joint future DINO scene-query + action Rectified-Flow agent."""

    def __init__(
        self,
        config,
        lr: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        drivor_backbone_ckpt: Optional[str] = None,
        scene_weight: float = 1.0,
        action_weight: float = 1.0,
        action_weight_schedule: str = "half_then_linear",
        action_weight_end: float = 2.0,
        action_weight_ramp_start_ratio: float = 0.5,
        eval_pdm: bool = False,
        eval_pdm_every_n_epochs: int = 5,
        metric_cache_path: Optional[str] = None,
        drivor_scorer_ckpt: Optional[str] = None,
        # --- AdamW / cosine-warmup (Diffusion Policy defaults) ---
        weight_decay: float = 1.0e-6,
        adam_betas: Optional[List[float]] = None,
        warmup_steps: int = 500,
        lr_min_ratio: float = 0.0,
        # Optional override; Lightning prefers estimated_stepping_batches.
        scheduler_total_steps: int = 0,
        # --- Adaptive / fixed EMA ---
        use_ema: bool = True,
        ema_schedule: str = "fixed",
        ema_decay: float = 0.9999,
        ema_inv_gamma: float = 1.0,
        ema_power: float = 0.75,
        ema_max_value: float = 0.9999,
        ema_update_after_step: int = 0,
    ) -> None:
        """
        Args:
            config: Model / sensor config (Hydra DictConfig or dict).
            lr: AdamW peak learning rate (after warmup).
            checkpoint_path: Optional full DinoDiT Lightning ckpt for resume/eval.
            drivor_backbone_ckpt: DrivoR pretrained ckpt used to initialize and
                freeze the DINO register / backbone.
            scene_weight: Loss weight for scene-token RF velocity.
            action_weight: Base / first-half λ_action for action RF velocity.
            action_weight_schedule: ``constant`` or ``half_then_linear``
                (first half of training keeps ``action_weight``, then linear
                ramp to ``action_weight_end``).
            action_weight_end: λ_action at the end of training.
            action_weight_ramp_start_ratio: Progress ∈ [0,1] where ramp starts.
            eval_pdm: If True, run closed-loop PDMS during validation.
            eval_pdm_every_n_epochs: Run PDMS when ``epoch % N == 0``.
            metric_cache_path: Root of NAVSIM metric cache (token → .xz).
                Defaults to ``$NAVSIM_EXP_ROOT/train_metric_cache``.
            drivor_scorer_ckpt: Frozen DrivoR Lightning ckpt used at eval to
                rank ``num_proposals`` RF trajectories (neural Scorer).
            weight_decay: AdamW weight decay (DP default ``1e-6``).
            adam_betas: AdamW betas (DP default ``[0.95, 0.999]``).
            warmup_steps: Linear LR warmup steps before cosine decay.
            lr_min_ratio: Final LR as a fraction of peak ``lr`` (0 → anneal to 0).
            scheduler_total_steps: Cosine horizon in optimizer steps. ``0`` →
                filled from ``trainer.estimated_stepping_batches`` via
                ``get_optimizers_with_total_steps``.
            use_ema: Enable EMA callback (val / ckpt use EMA weights).
            ema_schedule: ``"fixed"`` (constant decay, default) or
                ``"adaptive"`` (DP power-law).
            ema_decay: Fixed EMA decay when ``ema_schedule="fixed"``.
            ema_inv_gamma: Power-law EMA ``inv_gamma`` (adaptive only).
            ema_power: Power-law EMA exponent (adaptive only).
            ema_max_value: Adaptive decay upper clamp.
            ema_update_after_step: Skip EMA updates before this step.
        """
        super().__init__()
        if OmegaConf.is_config(config):
            self._config = OmegaConf.to_container(config, resolve=True)
        else:
            self._config = dict(config)
        self._lr = float(lr)
        self._weight_decay = float(weight_decay)
        betas = list(adam_betas) if adam_betas is not None else [0.95, 0.999]
        if len(betas) != 2:
            raise ValueError(f"adam_betas must have length 2, got {betas}")
        self._adam_betas = (float(betas[0]), float(betas[1]))
        self._warmup_steps = max(int(warmup_steps), 0)
        self._lr_min_ratio = float(lr_min_ratio)
        self._scheduler_total_steps = int(scheduler_total_steps)

        self._use_ema = bool(use_ema)
        self._ema_schedule = str(ema_schedule)
        self._ema_decay = float(ema_decay)
        self._ema_inv_gamma = float(ema_inv_gamma)
        self._ema_power = float(ema_power)
        self._ema_max_value = float(ema_max_value)
        self._ema_update_after_step = int(ema_update_after_step)

        self._checkpoint_path = checkpoint_path or ""
        self._drivor_backbone_ckpt = (
            drivor_backbone_ckpt
            or self._config.get("drivor_backbone_ckpt")
            or ""
        )
        self._drivor_scorer_ckpt = (
            drivor_scorer_ckpt
            or self._config.get("drivor_scorer_ckpt")
            or ""
        )
        self._eval_pdm = bool(eval_pdm)
        self._eval_pdm_every_n_epochs = max(int(eval_pdm_every_n_epochs), 1)
        self._metric_cache_path = metric_cache_path or self._config.get(
            "metric_cache_path"
        )
        self._metric_cache_paths: Optional[Dict[str, str]] = None
        self._get_scores = None

        self._model = DinoDiTModel(self._config)
        self._loss_fn = DinoDiTLoss(
            scene_weight=scene_weight,
            action_weight=action_weight,
            action_weight_schedule=action_weight_schedule,
            action_weight_end=action_weight_end,
            action_weight_ramp_start_ratio=action_weight_ramp_start_ratio,
        )

        if self._drivor_backbone_ckpt:
            stats = self._model.load_drivor_backbone(self._drivor_backbone_ckpt)
            print(
                f"[DinoDiT] loaded DrivoR backbone from {self._drivor_backbone_ckpt}: "
                f"{stats}"
            )

    def name(self) -> str:
        return self.__class__.__name__

    def update_loss_schedule(
        self, global_step: int, total_steps: int
    ) -> float:
        """Update λ_action from training progress (Lightning hook)."""
        return self._loss_fn.set_train_progress(global_step, total_steps)

    def initialize(self) -> None:
        """Load DinoDiT (eval/resume) then the frozen DrivoR Scorer if set.

        Scorer is loaded after the DinoDiT ckpt so random scorer weights are
        not overwritten by a missing-key ``strict=False`` load, and the
        DinoDiT ``state_dict`` cannot clobber Scorer tensors.
        """
        if self._checkpoint_path:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            state = torch.load(self._checkpoint_path, map_location=map_location)
            state_dict = state["state_dict"] if "state_dict" in state else state
            remapped = {k.replace("agent.", ""): v for k, v in state_dict.items()}
            self.load_state_dict(remapped, strict=False)
            # Ensure backbone stays frozen after resume.
            self._model.dino_register.freeze()
            if self._model.scorer_dino_register is not None:
                self._model.scorer_dino_register.freeze()

        scorer_ckpt = str(self._drivor_scorer_ckpt or "").strip()
        if scorer_ckpt:
            stats = self._model.load_drivor_scorer(scorer_ckpt)
            print(
                f"[DinoDiT] loaded frozen DrivoR Scorer from {scorer_ckpt}: "
                f"{stats} scorer_cams={self._model.scorer_camera_order}"
            )
        elif self._model.use_drivor_scorer:
            raise RuntimeError(
                "agent.config.use_drivor_scorer=true but drivor_scorer_ckpt "
                "is empty. Pass agent.drivor_scorer_ckpt=..."
            )

    def should_eval_pdm(self, epoch: int) -> bool:
        """Whether to run PDMS on this validation epoch.

        Args:
            epoch: Current Lightning epoch (0-indexed).

        Returns:
            True iff PDMS is enabled and ``epoch % N == 0``.
        """
        if not self._eval_pdm:
            return False
        return int(epoch) % self._eval_pdm_every_n_epochs == 0

    def _ensure_pdm_ready(self) -> bool:
        """Lazy-load metric cache + scorer used by DrivoR PDMS.

        Returns:
            True if PDMS dependencies are available.
        """
        if self._metric_cache_paths is not None and self._get_scores is not None:
            return True

        cache_root = self._metric_cache_path
        if not cache_root:
            exp_root = os.getenv("NAVSIM_EXP_ROOT", "")
            cache_root = (
                os.path.join(exp_root, "train_metric_cache") if exp_root else ""
            )
        if not cache_root or not Path(cache_root).exists():
            warnings.warn(
                f"[DinoDiT] PDMS enabled but metric cache not found at "
                f"'{cache_root}'. Skipping PDMS this epoch.",
                stacklevel=2,
            )
            return False

        try:
            from navsim.agents.drivoR.score_module.compute_navsim_score import (
                get_scores,
            )
            from navsim.common.dataloader import MetricCacheLoader
        except ImportError as exc:
            warnings.warn(
                f"[DinoDiT] Failed to import PDMS helpers: {exc}",
                stacklevel=2,
            )
            return False

        loader = MetricCacheLoader(Path(cache_root))
        self._metric_cache_paths = dict(loader.metric_cache_paths)
        self._get_scores = get_scores
        print(
            f"[DinoDiT] PDMS metric cache loaded from {cache_root} "
            f"({len(self._metric_cache_paths)} tokens)"
        )
        return True

    def compute_pdm_score(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Score inferred trajectories with closed-loop PDMS (DrivoR scorer).

        Args:
            features: Batch features; may contain ``scenario_token``.
            targets: Must contain GT ``trajectory`` ``[B, T, 3]`` and ideally
                ``token`` (scene initial tokens).
            predictions: Infer outputs with absolute ``trajectory`` ``[B, T, 3]``.

        Returns:
            Dict of scalar metrics (``pdm_score``, ``l2_2s``, collision, …).
            Empty dict if metric cache / tokens are unavailable.
        """
        if not self._ensure_pdm_ready():
            return {}

        traj = predictions["trajectory"].detach()  # [B, T, 3]
        gt_traj = targets["trajectory"].detach()
        tokens = targets.get("token", features.get("scenario_token"))
        if tokens is None:
            warnings.warn(
                "[DinoDiT] PDMS needs targets['token'] or features['scenario_token'].",
                stacklevel=2,
            )
            return {}

        # Collate may yield list[str] or a single str for B=1.
        if isinstance(tokens, str):
            token_list = [tokens]
        else:
            token_list = list(tokens)

        # DrivoR scorer expects per-sample poses ``[num_proposals, T, 3]``.
        proposals = traj[:, None].cpu().numpy()  # [B, 1, T, 3]
        data_points = []
        valid_indices = []
        for i, (token, poses) in enumerate(zip(token_list, proposals)):
            cache_path = self._metric_cache_paths.get(token)
            if cache_path is None:
                continue
            data_points.append({"token": cache_path, "poses": poses, "test": True})
            valid_indices.append(i)

        if not data_points:
            warnings.warn(
                "[DinoDiT] No val tokens found in metric cache; skip PDMS batch.",
                stacklevel=2,
            )
            return {}

        all_res = self._get_scores(data_points)
        # scores: [1, 7] metrics per proposal → stack to [B_valid, 1, 7]
        target_scores = torch.as_tensor(
            np.stack([res[0] for res in all_res]),
            device=traj.device,
            dtype=torch.float32,
        )
        final_scores = target_scores[:, 0, -1]  # [B_valid]
        sub = target_scores[:, 0]  # [B_valid, 7]

        gt_sel = gt_traj[valid_indices]
        pred_sel = traj[valid_indices]
        # 2s ADE on first 4 poses @ 2Hz (same as DrivoR val).
        l2_2s = torch.linalg.norm(pred_sel - gt_sel, dim=-1)[:, :4].mean()

        return {
            "pdm_score": final_scores.mean(),
            "l2_2s": l2_2s,
            "collision": sub[:, 0].mean(),
            "dac": sub[:, 1].mean(),
            "progress": sub[:, 2].mean(),
            "ttc": sub[:, 3].mean(),
            "comfort": sub[:, 4].mean(),
            "pdm_coverage": torch.tensor(
                len(valid_indices) / max(len(token_list), 1),
                device=traj.device,
                dtype=torch.float32,
            ),
        }

    def get_sensor_config(self) -> SensorConfig:
        """Build sensor load indices for history + current + future GT frames.

        ``SensorConfig`` lists are absolute indices into
        ``[history_frames | future_frames]``. With
        ``num_history_frames=4``, current is index 3; history offset ``-2``
        (-1.0s @ 2Hz) → absolute 1; future offsets ``[4,8]`` → 7 and 11.
        When ``use_drivor_scorer``, also load current-frame ``cam_b0/l0/r0``.
        """
        cfg = self._config
        num_history = int(cfg.get("num_history_frames", 4))
        current_idx = num_history - 1
        history_offsets = [
            int(x) for x in list(cfg.get("history_frame_offsets", [-2]))
        ]
        future_offsets = [
            int(x) for x in list(cfg.get("future_frame_offsets", [4, 8]))
        ]
        default_cam_f0 = (
            [current_idx + off for off in history_offsets]
            + [current_idx]
            + [current_idx + off for off in future_offsets]
        )
        for abs_idx in default_cam_f0:
            if abs_idx < 0:
                raise ValueError(
                    f"Sensor index {abs_idx} < 0; increase num_history_frames "
                    f"(got {num_history}) or reduce |history_frame_offsets|."
                )

        def _as_sensor_spec(value: Any, default: Any) -> Any:
            if value is None:
                return default
            # Keep bool ``True/False`` (load all / none) if user sets it.
            if isinstance(value, bool):
                return value
            return list(value)

        cam_f0 = _as_sensor_spec(cfg.get("cam_f0"), default_cam_f0)
        cam_l0 = _as_sensor_spec(cfg.get("cam_l0"), [])
        cam_l1 = _as_sensor_spec(cfg.get("cam_l1"), [])
        cam_l2 = _as_sensor_spec(cfg.get("cam_l2"), [])
        cam_r0 = _as_sensor_spec(cfg.get("cam_r0"), [])
        cam_r1 = _as_sensor_spec(cfg.get("cam_r1"), [])
        cam_r2 = _as_sensor_spec(cfg.get("cam_r2"), [])
        cam_b0 = _as_sensor_spec(cfg.get("cam_b0"), [])

        # Scorer KV needs current-frame 4-cam blobs; planner cam_f0 already
        # includes current_idx so only side/rear lists are extended.
        if bool(cfg.get("use_drivor_scorer", False)):
            scorer_order = list(
                cfg.get(
                    "scorer_camera_order",
                    ["cam_f0", "cam_b0", "cam_l0", "cam_r0"],
                )
            )
            cam_map = {
                "cam_f0": cam_f0,
                "cam_l0": cam_l0,
                "cam_l1": cam_l1,
                "cam_l2": cam_l2,
                "cam_r0": cam_r0,
                "cam_r1": cam_r1,
                "cam_r2": cam_r2,
                "cam_b0": cam_b0,
            }
            for name in scorer_order:
                spec = cam_map[name]
                if isinstance(spec, bool):
                    continue
                if current_idx not in spec:
                    spec.append(current_idx)

        return SensorConfig(
            cam_f0=cam_f0,
            cam_l0=cam_l0,
            cam_l1=cam_l1,
            cam_l2=cam_l2,
            cam_r0=cam_r0,
            cam_r1=cam_r1,
            cam_r2=cam_r2,
            cam_b0=cam_b0,
            lidar_pc=_as_sensor_spec(cfg.get("lidar_pc"), []),
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [DinoDiTFeatureBuilder(self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [DinoDiTTargetBuilder(self._config)]

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        # Val also passes targets to score RF loss; only pure infer omits them.
        return self._model(features, targets)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self._loss_fn(predictions, targets)

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Build AdamW + cosine-warmup; uses ``scheduler_total_steps`` if set.

        Prefer ``get_optimizers_with_total_steps`` via the Lightning module so
        ``trainer.estimated_stepping_batches`` fills the cosine horizon.
        """
        total = self._scheduler_total_steps
        if total <= 0:
            # Fallback when Lightning cannot estimate (e.g. IterableDataset).
            # Warmup still applies; cosine uses a large placeholder horizon.
            total = max(self._warmup_steps + 1, 100_000)
            warnings.warn(
                f"[DinoDiT] scheduler_total_steps unset; using placeholder "
                f"total_steps={total}. Prefer Lightning estimated_stepping_batches.",
                stacklevel=2,
            )
        return self.get_optimizers_with_total_steps(total)

    def get_optimizers_with_total_steps(
        self, total_steps: int
    ) -> Dict[str, Any]:
        """AdamW + linear warmup + cosine decay (Diffusion Policy style).

        Args:
            total_steps: Total optimizer steps over the full training run
                (typically ``trainer.estimated_stepping_batches``).

        Returns:
            Lightning-compatible dict with ``optimizer`` and ``lr_scheduler``.
        """
        total_steps = max(int(total_steps), 1)
        warmup = min(self._warmup_steps, max(total_steps - 1, 0))
        min_ratio = self._lr_min_ratio

        # Only optimize trainable params (DiT / traj / ego); DINO is frozen.
        params = [p for p in self._model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self._lr,
            betas=self._adam_betas,
            weight_decay=self._weight_decay,
        )

        def lr_lambda(current_step: int) -> float:
            # Linear warmup from ~0 → 1.0 over ``warmup`` steps.
            if warmup > 0 and current_step < warmup:
                return float(current_step + 1) / float(warmup)
            # Cosine anneal from 1.0 → min_ratio over the remaining steps.
            progress = float(current_step - warmup) / float(
                max(total_steps - warmup, 1)
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        print(
            f"[DinoDiT] AdamW lr={self._lr} wd={self._weight_decay} "
            f"betas={self._adam_betas}; cosine warmup={warmup}/{total_steps} "
            f"min_ratio={min_ratio}"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def get_training_callbacks(self) -> List[Any]:
        """Lightning callbacks: every-epoch ckpt, LR monitor, optional EMA.

        EMA (when enabled) swaps averaged weights for validation / PDMS and
        writes them into checkpoint ``state_dict``, so eval without the
        callback still loads EMA weights.
        """
        # save_top_k=-1 → never prune; one file per epoch + last.ckpt.
        checkpoint_cb = ModelCheckpoint(
            filename="epoch={epoch}-step={step}",
            auto_insert_metric_name=False,
            monitor=None,
            save_top_k=-1,
            every_n_epochs=1,
            save_last=True,
        )
        callbacks: List[Any] = [
            checkpoint_cb,
            LearningRateMonitor(logging_interval="step"),
        ]
        if self._use_ema:
            ema_cb = AdaptiveEMAWeightAveraging(
                schedule=self._ema_schedule,
                decay=self._ema_decay,
                inv_gamma=self._ema_inv_gamma,
                power=self._ema_power,
                min_value=0.0,
                max_value=self._ema_max_value,
                update_every_n_steps=1,
                update_starting_at_step=self._ema_update_after_step,
            )
            callbacks.append(ema_cb)
            if self._ema_schedule == "fixed":
                print(
                    f"[DinoDiT] EMA enabled: schedule=fixed decay={self._ema_decay} "
                    f"update_after_step={self._ema_update_after_step}"
                )
            else:
                print(
                    f"[DinoDiT] EMA enabled: schedule=adaptive "
                    f"inv_gamma={self._ema_inv_gamma} power={self._ema_power} "
                    f"max_decay={self._ema_max_value} "
                    f"update_after_step={self._ema_update_after_step}"
                )
        return callbacks
