"""Rectified-Flow / Flow-Matching scheduler (DriveVA-compatible).

Ported from DriveVA ``diffsynth.schedulers.flow_match.FlowMatchScheduler``
and extended with optional β-distribution timestep loss weights (π0-style).
"""

from __future__ import annotations

from typing import Optional

import torch


def _beta_pdf(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Beta(α, β) PDF on ``x ∈ (0, 1]`` (no scipy dependency).

    Args:
        x: Values in ``(0, 1]``.
        alpha: Beta shape α (>0).
        beta: Beta shape β (>0).

    Returns:
        PDF values, same shape as ``x``.
    """
    # B(α,β) = Γ(α)Γ(β)/Γ(α+β). For common (1.5, 1): B = Γ(1.5)Γ(1)/Γ(2.5).
    # Use torch.lgamma for general (α, β).
    log_b = (
        torch.lgamma(torch.tensor(alpha))
        + torch.lgamma(torch.tensor(beta))
        - torch.lgamma(torch.tensor(alpha + beta))
    )
    # pdf = x^{α-1} (1-x)^{β-1} / B(α,β)
    log_pdf = (
        (alpha - 1.0) * torch.log(x.clamp(min=1.0e-8))
        + (beta - 1.0) * torch.log((1.0 - x).clamp(min=1.0e-8))
        - log_b
    )
    return torch.exp(log_pdf)


class FlowMatchScheduler:
    """Linear Rectified-Flow scheduler with optional shift.

    Training:
      x_t = (1 - sigma) * x_0 + sigma * noise
      target velocity v = noise - x_0

    Inference step:
      x_{t'} = x_t + v_pred * (sigma' - sigma)

    Timestep loss weights (``loss_weight_mode``):
      - ``uniform``: w(σ)=1
      - ``beta``: w(σ) ∝ Beta(α,β).pdf(σ), mean-normalized to 1.
        Default α=1.5, β=1 → larger weight at high σ (early / noisier steps).
    """

    def __init__(
        self,
        num_inference_steps: int = 10,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
        extra_one_step: bool = True,
        loss_weight_mode: str = "uniform",
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.loss_weight_mode = str(loss_weight_mode)
        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)
        self.training = False
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 10,
        training: bool = False,
        shift: Optional[float] = None,
    ) -> None:
        """Build sigma / timestep grids.

        Args:
            num_inference_steps: Number of denoising steps at inference,
                or the training discretization size when ``training=True``.
            training: If True, also build per-timestep loss weights.
            shift: Optional override for the flow shift.
        """
        if shift is not None:
            self.shift = shift
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                self.sigma_max, self.sigma_min, num_inference_steps + 1
            )[:-1]
        else:
            self.sigmas = torch.linspace(
                self.sigma_max, self.sigma_min, num_inference_steps
            )
        # DriveVA-style shift: sigma' = shift*sigma / (1 + (shift-1)*sigma)
        self.sigmas = self.shift * self.sigmas / (
            1.0 + (self.shift - 1.0) * self.sigmas
        )
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            self.linear_timesteps_weights = self._build_timestep_weights(
                self.sigmas
            )
            self.training = True
        else:
            self.training = False

    def _build_timestep_weights(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Build per-grid loss weights from ``loss_weight_mode``.

        Args:
            sigmas: RF noise levels ``[S]``, typically descending 1→0.

        Returns:
            Weights ``[S]`` with mean ≈ 1 (scale-neutral vs uniform).
        """
        mode = self.loss_weight_mode.lower()
        n = len(sigmas)
        if mode in ("uniform", "none", "off"):
            return torch.ones(n, dtype=torch.float32)

        if mode in ("beta", "beta_pdf"):
            # Map σ into (0,1] for the Beta support; high σ → larger PDF for
            # Beta(1.5, 1) ⇒ early / noisier steps get larger action loss.
            x = sigmas.detach().float().clamp(1.0e-4, 1.0)
            w = _beta_pdf(x, self.beta_alpha, self.beta_beta)
            w = w / w.mean().clamp(min=1.0e-8)
            return w.to(dtype=torch.float32)

        raise ValueError(
            f"Unknown loss_weight_mode={self.loss_weight_mode!r}; "
            "expected 'uniform' or 'beta'."
        )

    def _sigma_from_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Map a continuous/discrete timestep to sigma on the current grid."""
        # timestep: scalar or [B]
        t = timestep.detach().float().reshape(-1)
        # Nearest grid index per batch element.
        # self.timesteps: [S]
        grid = self.timesteps.to(device=t.device, dtype=t.dtype)
        idx = torch.argmin((grid[None, :] - t[:, None]).abs(), dim=-1)
        return self.sigmas.to(device=t.device, dtype=t.dtype)[idx]

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Form x_t = (1 - sigma) * x_0 + sigma * noise.

        Args:
            original_samples: Clean sample, any shape ``[B, ...]``.
            noise: Same shape as ``original_samples``.
            timestep: ``[B]`` in scheduler timestep units.

        Returns:
            Noisy sample ``[B, ...]``.
        """
        sigma = self._sigma_from_timestep(timestep)
        while sigma.ndim < original_samples.ndim:
            sigma = sigma.unsqueeze(-1)
        return (1.0 - sigma) * original_samples + sigma * noise

    def training_target(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor = None,
    ) -> torch.Tensor:
        """Rectified-Flow velocity target: ``v = noise - x_0``."""
        del timestep  # unused; kept for DriveVA API compatibility
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """Per-sample loss weight for the sampled timesteps.

        Args:
            timestep: Diffusion timesteps ``[B]``.

        Returns:
            Weights ``[B]`` (mean≈1 under ``beta`` mode).
        """
        t = timestep.detach().float().reshape(-1)
        if not getattr(self, "training", False):
            return torch.ones_like(t, dtype=torch.float32)
        weights = self.linear_timesteps_weights.to(device=t.device, dtype=t.dtype)
        grid = self.timesteps.to(device=t.device, dtype=t.dtype)
        idx = torch.argmin((grid[None, :] - t[:, None]).abs(), dim=-1)
        return weights[idx]

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        to_final: bool = False,
    ) -> torch.Tensor:
        """One Euler step along the predicted velocity field.

        Args:
            model_output: Predicted velocity ``v``, same shape as ``sample``.
            timestep: Current timestep (scalar or ``[B]``; uses batch[0]).
            sample: Current noisy sample.
            to_final: Force sigma_next = 0.

        Returns:
            Updated sample.
        """
        t0 = timestep.reshape(-1)[0].detach().float().cpu()
        timestep_id = int(torch.argmin((self.timesteps - t0).abs()).item())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = torch.tensor(0.0, dtype=sigma.dtype)
        else:
            sigma_next = self.sigmas[timestep_id + 1]
        delta = (sigma_next - sigma).to(
            device=sample.device, dtype=sample.dtype
        )
        return sample + model_output * delta
