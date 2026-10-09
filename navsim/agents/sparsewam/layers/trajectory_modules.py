"""Trajectory encode / decode helpers for SparseWAM.

Pipeline (one geometric transform + one scale align):
  absolute (x,y,ψ) → relative deltas → per-dim z-score → encoder / RF
  decode: inv z-score → cumsum → absolute poses

Relative delta is a *representation* change (not a second scale norm).
Z-score is the only scale alignment, so action RF space has std≈1 like
frozen DINO tokens. Stats from full navtrain (103288 scenes).

TrajectoryEncoder follows DiffusionDrive (hustvl/DiffusionDrive):
DAB-DETR ``gen_sineembed_for_position`` on metric (x, y) [, heading],
then an MLP (Linear-ReLU-LN + Linear), matching
``plan_anchor_encoder`` in ``transfuser_model_v2.py``.

Google-style comments annotate abstract and concrete tensor shapes, e.g.
``X [B, T, 3] [2, 8, 3]``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# Per-dim mean/std of relative-delta (x,y,ψ) on navtrain.
# Source: navtrain relative-delta statistics.
_TRAJ_MEAN = (2.225117, 0.109378, 0.010122)
_TRAJ_STD = (1.473598, 0.619594, 0.058290)
_TRAJ_STD_EPS = 1.0e-6


def to_relative_trajectory(traj: torch.Tensor) -> torch.Tensor:
    """Convert absolute poses to relative deltas (DriveVA).

    The first pose stays absolute; later poses become consecutive deltas.

    Args:
        traj: Absolute poses ``[B, T, 3]`` or ``[T, 3]`` e.g. ``[2, 8, 3]``.

    Returns:
        Relative poses with the same shape as ``traj``.
    """
    squeeze = False
    if traj.ndim == 2:
        traj = traj.unsqueeze(0)
        squeeze = True
    traj_xy = traj[..., :2]
    delta_xy = traj_xy[:, 1:] - traj_xy[:, :-1]
    rel_xy = torch.cat([traj_xy[:, 0:1], delta_xy], dim=1)
    if traj.shape[-1] > 2:
        heading = traj[..., 2:3]
        # Wrap to (-π, π] so ±π crossings stay small (e.g. 3.14 → -3.14
        # is ~0, not ~-6.28).
        delta_h = heading[:, 1:] - heading[:, :-1]
        delta_h = torch.atan2(torch.sin(delta_h), torch.cos(delta_h))
        rel_h = torch.cat([heading[:, 0:1], delta_h], dim=1)
        out = torch.cat([rel_xy, rel_h], dim=-1)
    else:
        out = rel_xy
    return out.squeeze(0) if squeeze else out


def from_relative_trajectory(traj_rel: torch.Tensor) -> torch.Tensor:
    """Accumulate relative deltas back to absolute poses (DriveVA).

    Args:
        traj_rel: Relative poses ``[B, T, C]`` or ``[T, C]``.

    Returns:
        Absolute poses with the same shape. Heading is wrapped to ``(-π, π]``.
    """
    squeeze = False
    if traj_rel.ndim == 2:
        traj_rel = traj_rel.unsqueeze(0)
        squeeze = True
    out = torch.cumsum(traj_rel, dim=1)
    if out.shape[-1] > 2:
        # Keep heading in NAVSIM's principal range after accumulation.
        heading = torch.atan2(torch.sin(out[..., 2:3]), torch.cos(out[..., 2:3]))
        out = torch.cat([out[..., :2], heading], dim=-1)
    return out.squeeze(0) if squeeze else out


def _traj_mean_std(
    traj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcastable mean/std tensors matching ``traj`` device/dtype."""
    n = traj.shape[-1]
    mean = torch.tensor(
        _TRAJ_MEAN[:n], device=traj.device, dtype=traj.dtype
    )
    std = torch.tensor(
        _TRAJ_STD[:n], device=traj.device, dtype=traj.dtype
    ).clamp_min(_TRAJ_STD_EPS)
    return mean, std


def norm_trajectory(traj: torch.Tensor) -> torch.Tensor:
    """Per-dim z-score of relative-delta (x, y, heading).

    ``(x - mean) / std`` with navtrain stats so RF action space ≈ unit std,
    matching frozen DINO token scale.

    Args:
        traj: Relative-delta poses ``[..., 2|3]`` in meters / radians.

    Returns:
        Standardized poses with the same shape.
    """
    mean, std = _traj_mean_std(traj)
    return (traj - mean) / std


def denorm_trajectory(traj: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`norm_trajectory` (z-score → meters / radians).

    Args:
        traj: Standardized poses ``[..., 2|3]``.

    Returns:
        Relative-delta poses with the same shape.
    """
    mean, std = _traj_mean_std(traj)
    return traj * std + mean


def encode_trajectory_for_flow(traj_abs: torch.Tensor) -> torch.Tensor:
    """Absolute poses → relative-delta → z-score space used by RF + MLP.

    Args:
        traj_abs: ``[B, T, 3]`` e.g. ``[2, 8, 3]`` (ego-local meters/rad).

    Returns:
        ``[B, T, 3]`` in navtrain z-scored relative-delta space (std≈1).
    """
    return norm_trajectory(to_relative_trajectory(traj_abs))


def decode_trajectory_from_flow(traj_norm_rel: torch.Tensor) -> torch.Tensor:
    """Z-scored relative-delta RF output → absolute poses.

    Args:
        traj_norm_rel: ``[B, T, 3]`` in navtrain z-scored relative-delta space.

    Returns:
        Absolute ego-local poses ``[B, T, 3]``.
    """
    return from_relative_trajectory(denorm_trajectory(traj_norm_rel))


def expand_trajectories_mul_noise(
    traj: torch.Tensor,
    num_copies: int,
    noise_std: float,
) -> torch.Tensor:
    """DiffusionDriveV2 multiplicative long/lat noise around RF trajectories.

    Paper form (selector augmentation)::

        τ' = (1 + ε_mul) τ,   ε_mul = (ε_long, ε_lat) ~ N(0, σ²)

    Each noisy copy draws **one** pair of scalars and scales the whole
    trajectory in ego-local coordinates::

        x' = (1 + ε_long) x,   y' = (1 + ε_lat) y,   ψ' = ψ

    Original trajectories are kept as the first ``K`` slots so Scorer
    can still pick an unperturbed RF sample.

    Args:
        traj: Absolute ego-local poses ``[B, K, T, 3]`` e.g. ``[4, 8, 8, 3]``.
        num_copies: Noisy copies per RF trajectory (7 → scored M = 8×8 = 64).
        noise_std: σ of ε_long / ε_lat (typically ``0.1``).

    Returns:
        ``[B, K*(1+num_copies), T, 3]`` e.g. ``[4, 64, 8, 3]``.
        Concat order: originals then noisy copies.
    """
    if num_copies <= 0:
        return traj
    batch, num_modes, num_poses, dim = traj.shape
    # [B, K, C, T, 3] e.g. [4, 8, 7, 8, 3] — C independent copies.
    base = traj.unsqueeze(2).expand(
        batch, num_modes, num_copies, num_poses, dim
    )
    # One (ε_long, ε_lat) per copy, broadcast over T.
    # [B, K, C, 1, 1] e.g. [4, 8, 7, 1, 1]
    eps_long = (
        torch.randn(
            batch,
            num_modes,
            num_copies,
            1,
            1,
            device=traj.device,
            dtype=traj.dtype,
        )
        * noise_std
    )
    eps_lat = (
        torch.randn(
            batch,
            num_modes,
            num_copies,
            1,
            1,
            device=traj.device,
            dtype=traj.dtype,
        )
        * noise_std
    )
    noisy = base.clone()
    noisy[..., 0:1] = (1.0 + eps_long) * base[..., 0:1]
    noisy[..., 1:2] = (1.0 + eps_lat) * base[..., 1:2]
    # [B, K*C, T, 3] e.g. [4, 56, 8, 3]
    noisy = noisy.reshape(batch, num_modes * num_copies, num_poses, dim)
    # [B, K + K*C, T, 3] e.g. [4, 64, 8, 3]
    return torch.cat([traj, noisy], dim=1)


def gen_sineembed_for_position(
    pos_tensor: torch.Tensor, hidden_dim: int = 256
) -> torch.Tensor:
    """DAB-DETR / DiffusionDrive 2D sine positional embedding.

    Copied from ``hustvl/DiffusionDrive``
    ``navsim/agents/diffusiondrive/modules/blocks.py``::

        gen_sineembed_for_position(pos_tensor, hidden_dim=64)

    Only ``(x, y)`` are used (heading is ignored here). Frequency basis::

        dim_t[i] = 10000 ** (2 * (i // 2) / (hidden_dim // 2))
        pos_x = sin/cos( (x * 2π) / dim_t )
        pos_y = sin/cos( (y * 2π) / dim_t )
        out   = cat(pos_y, pos_x)   # note: y then x

    Args:
        pos_tensor: Positions ``[..., C]`` with ``C >= 2`` (meters).
            e.g. ``[2, 8, 3]`` — only ``[..., 0]`` and ``[..., 1]`` used.
        hidden_dim: Output channels (even). DiffusionDrive uses ``64``.

    Returns:
        Sine embedding ``[..., hidden_dim]`` e.g. ``[2, 8, 64]``.
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    # Temperature bands [half_hidden_dim], same device as positions.
    dim_t = torch.arange(
        half_hidden_dim, dtype=torch.float32, device=pos_tensor.device
    )
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    # DiffusionDrive order: (y, x).
    return torch.cat((pos_y, pos_x), dim=-1)


def gen_sineembed_for_scalar(
    scalar: torch.Tensor, hidden_dim: int = 64
) -> torch.Tensor:
    """Same ``10000`` temperature sine embed for a single scalar (heading).

    DiffusionDrive's ``gen_sineembed_for_position`` only covers ``(x, y)``.
    Our RF still denoises heading, so each action token must see noisy ``ψ``.

    Uses the standard 1D form (like ``SinusoidalPosEmb``)::

        out = cat(sin(s * 2π / dim_t), cos(s * 2π / dim_t))

    so the output width is exactly ``hidden_dim`` (not half).

    Args:
        scalar: ``[...,]`` or ``[..., 1]`` e.g. heading ``[2, 8]``.
        hidden_dim: Output channels (even).

    Returns:
        ``[..., hidden_dim]`` e.g. ``[2, 8, 64]``.
    """
    if scalar.ndim >= 1 and scalar.shape[-1] == 1:
        scalar = scalar.squeeze(-1)
    half = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half, dtype=torch.float32, device=scalar.device)
    # Match DiffusionDrive / DAB-DETR temperature schedule.
    dim_t = 10000 ** (2 * (dim_t // 2) / half)
    embed = (scalar * scale)[..., None] / dim_t  # [..., half]
    return torch.cat((embed.sin(), embed.cos()), dim=-1)  # [..., hidden_dim]


class TrajectoryEncoder(nn.Module):
    """Map noisy trajectory points to DiT tokens (DiffusionDrive-style).

    **RF denoising space is unchanged**: flow variable = z-scored relative
    deltas. Sine embedding is applied **directly on that RF state** so the
    encoder input stays on the same ~unit-std scale as ``ε ~ N(0, I)``.

    Do **not** denorm before sine: that would map back to meters/rad with
    heterogeneous per-dim scales and break the standard-Gaussian RF prior
    that the noisy sample lives in.

    Pipeline::

        noisy RF state (z-scored relative, std≈1)
          → gen_sineembed_for_position(xy) + heading sine
          → Linear-ReLU-LN + Linear → action token

    Args:
        point_dim: Per-pose dim, usually 3 for (x, y, heading).
        hidden_dim: Unused alias kept for call-site compat; MLP width is
            ``output_dim``.
        output_dim: DiT token dim ``D``.
        sine_dim: Channels for ``gen_sineembed_for_position`` (DiffusionDrive=64).
        encode_heading: If True, append heading sine embed (recommended).
        use_sine_embed: If False, fall back to plain MLP on RF-space poses.
    """

    def __init__(
        self,
        point_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 256,
        sine_dim: int = 64,
        encode_heading: bool = True,
        use_sine_embed: bool = True,
    ) -> None:
        super().__init__()
        self.point_dim = int(point_dim)
        self.sine_dim = int(sine_dim)
        self.encode_heading = bool(encode_heading) and self.point_dim > 2
        self.use_sine_embed = bool(use_sine_embed)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        if self.use_sine_embed:
            feat_dim = self.sine_dim + (
                self.sine_dim if self.encode_heading else 0
            )
            # DiffusionDrive plan_anchor_encoder style:
            #   Linear → ReLU → LayerNorm → Linear
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, output_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(output_dim),
                nn.Linear(output_dim, output_dim),
            )
            self.norm = None
        else:
            # Legacy RF MLP used by early SparseWAM ckpts:
            #   Linear(3, H) → ReLU → Linear(H, D) + LayerNorm(D)
            # Matches keys ``proj.0 / proj.2 / norm`` with H=tf_d_ffn (1024).
            feat_dim = self.point_dim
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, output_dim),
            )
            self.norm = nn.LayerNorm(output_dim)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Encode trajectory points for DiT (RF state itself is not converted).

        Args:
            trajectory: Noisy **relative** RF poses
                ``[B, T, point_dim]`` e.g. ``[2, 8, 3]`` (z-scored, std≈1).

        Returns:
            Action tokens ``[B, T, D]`` e.g. ``[2, 8, 256]``.
        """
        if not self.use_sine_embed:
            # [B, T, 3] → [B, T, D]
            return self.norm(self.proj(trajectory))

        # Sine-embed the RF state as-is (z-scored relative). No denorm.
        xy_embed = gen_sineembed_for_position(
            trajectory, hidden_dim=self.sine_dim
        )  # [B, T, sine_dim] e.g. [2, 8, 64]
        if self.encode_heading:
            h_embed = gen_sineembed_for_scalar(
                trajectory[..., 2], hidden_dim=self.sine_dim
            )
            feats = torch.cat([xy_embed, h_embed], dim=-1)  # [B, T, 2*sine_dim]
        else:
            feats = xy_embed
        return self.proj(feats)


class TrajectoryDecoder(nn.Module):
    """Map DiT action tokens back to trajectory poses."""

    def __init__(self, dim: int, out_dim: int = 3) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode action tokens.

        Args:
            tokens: ``[B, T, D]`` e.g. ``[2, 8, 256]``.

        Returns:
            Poses ``[B, T, 3]`` e.g. ``[2, 8, 3]``.
        """
        return self.proj(self.norm(tokens))


class EgoStatusEncoder(nn.Module):
    """Encode ego velocity / acceleration / command into one token."""

    def __init__(self, in_dim: int = 8, output_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, ego_status: torch.Tensor) -> torch.Tensor:
        """Encode ego status.

        Args:
            ego_status: ``[B, in_dim]`` e.g. ``[2, 8]``
                (vx, vy, ax, ay, cmd0..cmd3).

        Returns:
            Ego token ``[B, 1, D]`` e.g. ``[2, 1, 256]``.
        """
        return self.proj(ego_status).unsqueeze(1)
