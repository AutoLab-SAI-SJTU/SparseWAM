"""Rectified-Flow velocity losses for scene queries and trajectory."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def action_weight_at_progress(
    progress: float,
    schedule: str = "constant",
    weight_start: float = 1.0,
    weight_end: float = 2.0,
    ramp_start_ratio: float = 0.5,
) -> float:
    """Scalar λ_action as a function of training progress in ``[0, 1]``.

    Schedules:
      - ``constant``: always ``weight_start``.
      - ``half_then_linear``: ``weight_start`` for progress < ``ramp_start_ratio``,
        then linear ramp to ``weight_end`` over the remaining progress.

    Args:
        progress: ``global_step / total_steps``, clipped to ``[0, 1]``.
        schedule: Schedule name.
        weight_start: Weight in the first phase (and constant mode).
        weight_end: Weight at progress=1 for ``half_then_linear``.
        ramp_start_ratio: Progress where the linear ramp begins.

    Returns:
        Scalar action loss weight.
    """
    p = float(min(max(progress, 0.0), 1.0))
    name = str(schedule).lower()
    if name in ("constant", "none", "off"):
        return float(weight_start)
    if name in ("half_then_linear", "half_linear", "piecewise_linear"):
        start = float(ramp_start_ratio)
        if p <= start:
            return float(weight_start)
        denom = max(1.0 - start, 1.0e-8)
        t = (p - start) / denom
        return float(weight_start + t * (weight_end - weight_start))
    raise ValueError(
        f"Unknown action_weight_schedule={schedule!r}; "
        "expected 'constant' or 'half_then_linear'."
    )


class DinoDiTLoss(nn.Module):
    """Weighted MSE on RF velocities (scene token space + action poses).

    Flow-matching theory requires L2 so the optimum is the conditional mean
    velocity field (Smooth-L1 would target the median). ``loss_weight``
    (per-timestep, e.g. Beta) is applied to **action only**, so early /
    high-noise steps emphasize trajectory matching without skewing scene RF.
    """

    def __init__(
        self,
        scene_weight: float = 1.0,
        action_weight: float = 1.0,
        action_weight_schedule: str = "constant",
        action_weight_end: float = 2.0,
        action_weight_ramp_start_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.scene_weight = float(scene_weight)
        self.action_weight_start = float(action_weight)
        self.action_weight_schedule = str(action_weight_schedule)
        self.action_weight_end = float(action_weight_end)
        self.action_weight_ramp_start_ratio = float(action_weight_ramp_start_ratio)
        # Mutable scalar used in forward; updated via ``set_train_progress``.
        self.action_weight = float(action_weight)

    def set_train_progress(
        self, global_step: int, total_steps: int
    ) -> float:
        """Update ``action_weight`` from training progress.

        Args:
            global_step: Current optimizer step.
            total_steps: Estimated total optimizer steps.

        Returns:
            The updated scalar ``action_weight``.
        """
        total = max(int(total_steps), 1)
        progress = float(global_step) / float(total)
        self.action_weight = action_weight_at_progress(
            progress,
            schedule=self.action_weight_schedule,
            weight_start=self.action_weight_start,
            weight_end=self.action_weight_end,
            ramp_start_ratio=self.action_weight_ramp_start_ratio,
        )
        return self.action_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute RF velocity matching losses.

        Args:
            predictions: From ``forward_train``, must contain
                ``pred_scene_vel``, ``pred_action_vel``,
                ``target_scene_vel``, ``target_action_vel``,
                and optional ``loss_weight`` ``[B]`` (action-only).
            targets: Unused (kept for agent API compatibility).

        Returns:
            Dict with ``loss`` and detached logging terms.
        """
        del targets
        pred_scene = predictions["pred_scene_vel"]
        tgt_scene = predictions["target_scene_vel"]
        pred_action = predictions["pred_action_vel"]
        tgt_action = predictions["target_action_vel"]

        # Per-element MSE [B, F, K, dim] / [B, T, 3], then mean over trailing dims.
        scene_elem = F.mse_loss(pred_scene, tgt_scene, reduction="none")
        action_elem = F.mse_loss(pred_action, tgt_action, reduction="none")

        while scene_elem.ndim > 1:
            scene_elem = scene_elem.mean(dim=-1)
        while action_elem.ndim > 1:
            action_elem = action_elem.mean(dim=-1)

        # Beta / timestep weight only on action RF.
        weight = predictions.get("loss_weight")
        if weight is None:
            weight = torch.ones_like(action_elem)
        else:
            weight = weight.to(device=action_elem.device, dtype=action_elem.dtype)

        scene_loss = scene_elem.mean()
        action_loss = (action_elem * weight).mean()
        total = self.scene_weight * scene_loss + self.action_weight * action_loss
        return {
            "loss": total,
            "loss_scene": scene_loss.detach(),
            "loss_action": action_loss.detach(),
            "action_weight": torch.tensor(
                self.action_weight, device=total.device, dtype=total.dtype
            ),
        }
