"""EMA callback for DinoDiT (compatible with Lightning 2.2+).

Default: **fixed** decay::

    ema = decay * ema + (1 - decay) * current     # decay=0.9999

Optional adaptive (Diffusion Policy power-law)::

    decay(n) = clip(1 - (1 + n / inv_gamma)^(-power), min_value, max_value)

Implemented with ``torch.optim.swa_utils.AveragedModel`` + a plain Lightning
``Callback``. Does **not** import ``pytorch_lightning.callbacks.weight_averaging``
(that module only exists in newer PL, e.g. 2.5+/2.6; this stack uses 2.4.0).

Behavior:
  * update shadow weights after each optimizer step;
  * swap EMA ↔ raw for validation / PDMS;
  * checkpoint ``state_dict`` stores EMA weights; raw under ``current_model_state``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn


def get_adaptive_ema_avg_fn(
    inv_gamma: float = 1.0,
    power: float = 0.75,
    min_value: float = 0.0,
    max_value: float = 0.9999,
):
    """Build an AveragedModel avg_fn with DP-style adaptive decay.

    Args:
        inv_gamma: Inverse gamma of the power-law schedule.
        power: Exponent; 0.75 is the Diffusion Policy default.
        min_value: Lower clamp on decay.
        max_value: Upper clamp on decay (asymptotic EMA rate).

    Returns:
        ``avg_fn(ema_param, current_param, num_averaged) -> Tensor``.
    """

    @torch.no_grad()
    def avg_fn(
        ema_param: torch.Tensor,
        current_param: torch.Tensor,
        num_averaged: torch.Tensor,
    ) -> torch.Tensor:
        # num_averaged counts how many times update_parameters ran (1-based).
        n = num_averaged.to(dtype=ema_param.dtype)
        decay = 1.0 - (1.0 + n / inv_gamma).pow(-power)
        decay = torch.clamp(decay, min=min_value, max=max_value)
        return decay * ema_param + (1.0 - decay) * current_param

    return avg_fn


class AdaptiveEMAWeightAveraging(Callback):
    """EMA callback with fixed (default) or adaptive decay.

    Args:
        device: Device for the AveragedModel shadow copy. ``None`` → same as
            the training model. Use ``"cpu"`` if GPU memory is tight.
        use_buffers: Also average buffers (BatchNorm stats, etc.).
        schedule: ``"fixed"`` (default) or ``"adaptive"`` (DP power-law).
        decay: Fixed EMA decay when ``schedule="fixed"`` (default 0.9999).
        inv_gamma: Power-law inverse gamma (only if adaptive).
        power: Power-law exponent (only if adaptive).
        min_value: Adaptive decay lower clamp.
        max_value: Adaptive decay upper clamp.
        update_every_n_steps: Update EMA every N optimizer steps.
        update_starting_at_step: Skip EMA updates before this global step.
    """

    def __init__(
        self,
        device: Optional[Union[torch.device, str, int]] = None,
        use_buffers: bool = True,
        schedule: str = "fixed",
        decay: float = 0.9999,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
        update_every_n_steps: int = 1,
        update_starting_at_step: Optional[int] = 0,
    ) -> None:
        super().__init__()
        schedule = str(schedule).lower().strip()
        if schedule == "fixed":
            self._avg_fn = get_ema_avg_fn(decay=float(decay))
        elif schedule == "adaptive":
            self._avg_fn = get_adaptive_ema_avg_fn(
                inv_gamma=inv_gamma,
                power=power,
                min_value=min_value,
                max_value=max_value,
            )
        else:
            raise ValueError(
                f"Unknown ema schedule '{schedule}'. Use 'fixed' or 'adaptive'."
            )

        if isinstance(device, str):
            self._device: Optional[Union[torch.device, int]] = torch.device(device)
        else:
            self._device = device
        self._use_buffers = bool(use_buffers)
        self.schedule = schedule
        self.decay = float(decay)
        self.update_every_n_steps = int(update_every_n_steps)
        self.update_starting_at_step = (
            None
            if update_starting_at_step is None
            else int(update_starting_at_step)
        )

        self._average_model: Optional[AveragedModel] = None
        # Align with PL WeightAveraging: first real update check after step 1.
        self._latest_update_step = 0
        self._swapped = False

    def should_update(self, step_idx: Optional[int] = None) -> bool:
        """Whether to refresh EMA after this optimizer step.

        Args:
            step_idx: Zero-based optimizer step index.

        Returns:
            True iff EMA should be updated.
        """
        if step_idx is None:
            return False
        if (
            self.update_starting_at_step is not None
            and step_idx < self.update_starting_at_step
        ):
            return False
        return (
            self.update_every_n_steps > 0
            and step_idx % self.update_every_n_steps == 0
        )

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Create AveragedModel when fit begins."""
        if stage != "fit":
            return
        device = self._device or pl_module.device
        self._average_model = AveragedModel(
            model=pl_module,
            device=device,
            avg_fn=self._avg_fn,
            use_buffers=self._use_buffers,
        )
        rank_zero_info(
            f"[DinoDiT-EMA] AveragedModel ready on {device} "
            f"(schedule={self.schedule}, decay={self.decay})"
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA after each new optimizer step."""
        del outputs, batch, batch_idx
        # trainer.global_step is #optimizer steps taken so far (1 after first).
        step_idx = trainer.global_step - 1
        if (trainer.global_step > self._latest_update_step) and self.should_update(
            step_idx=step_idx
        ):
            assert self._average_model is not None
            self._average_model.update_parameters(pl_module)
            self._latest_update_step = trainer.global_step

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Swap EMA weights into the live model for val / PDMS."""
        del trainer
        if self._average_model is not None and not self._swapped:
            self._swap_models(pl_module)
            self._swapped = True

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Restore raw training weights after validation."""
        del trainer
        if self._average_model is not None and self._swapped:
            self._swap_models(pl_module)
            self._swapped = False

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Leave EMA weights in the live model after training."""
        del trainer
        if self._average_model is not None:
            if self._swapped:
                # Currently holding EMA in pl_module already (rare).
                self._swapped = False
            else:
                self._copy_average_to_current(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        """Persist callback bookkeeping."""
        return {"latest_update_step": self._latest_update_step}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore callback bookkeeping."""
        self._latest_update_step = int(state_dict.get("latest_update_step", 0))

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Put EMA weights into ``state_dict``; keep raw under ``current_model_state``.

        Eval scripts that load ``state_dict`` therefore get EMA weights without
        needing this callback.
        """
        del trainer, pl_module
        if self._average_model is None:
            rank_zero_info(
                "[DinoDiT-EMA] Saving outside fit; EMA shadow not available. "
                "state_dict will keep the current model weights."
            )
            return
        average_model_state = self._average_model.state_dict()
        checkpoint["current_model_state"] = checkpoint["state_dict"]
        # AveragedModel prefixes params with ``module.``.
        checkpoint["state_dict"] = {
            name[7:]: value
            for name, value in average_model_state.items()
            if name.startswith("module.")
        }
        checkpoint["averaging_state"] = {
            name: value
            for name, value in average_model_state.items()
            if not name.startswith("module.")
        }

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Restore EMA shadow + raw weights when resuming fit."""
        del trainer
        if self._average_model is None:
            rank_zero_warn(
                "[DinoDiT-EMA] Loading outside fit; cannot restore AveragedModel. "
                "If this is predict/test, state_dict already has EMA weights."
            )
            return
        if ("current_model_state" in checkpoint) and ("averaging_state" in checkpoint):
            rank_zero_info(
                "[DinoDiT-EMA] Restoring AveragedModel + current_model_state."
            )
            average_model_state = {
                "module." + name: value
                for name, value in checkpoint["state_dict"].items()
            }
            average_model_state.update(checkpoint["averaging_state"])
            self._average_model.load_state_dict(average_model_state)
            pl_module.load_state_dict(checkpoint["current_model_state"])
        else:
            rank_zero_warn(
                "[DinoDiT-EMA] Checkpoint has no EMA extras; "
                "initializing shadow from state_dict."
            )
            self._average_model.module.load_state_dict(
                deepcopy(checkpoint["state_dict"]), strict=False
            )

    def _swap_models(self, pl_module: LightningModule) -> None:
        """In-place swap of parameters/buffers between live model and EMA."""
        assert self._average_model is not None
        avg_params = list(self._average_model.module.parameters()) + list(
            self._average_model.module.buffers()
        )
        cur_params = list(pl_module.parameters()) + list(pl_module.buffers())
        if len(avg_params) != len(cur_params):
            raise RuntimeError(
                f"EMA swap size mismatch: avg={len(avg_params)} cur={len(cur_params)}"
            )
        for average_param, current_param in zip(avg_params, cur_params):
            tmp = average_param.data.clone()
            average_param.data.copy_(current_param.data)
            current_param.data.copy_(tmp)

    def _copy_average_to_current(self, pl_module: LightningModule) -> None:
        """Copy EMA weights into the live model (no swap back)."""
        assert self._average_model is not None
        avg_params = list(self._average_model.module.parameters()) + list(
            self._average_model.module.buffers()
        )
        cur_params = list(pl_module.parameters()) + list(pl_module.buffers())
        for average_param, current_param in zip(avg_params, cur_params):
            current_param.data.copy_(average_param.data)
