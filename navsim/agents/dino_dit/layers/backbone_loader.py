"""Load frozen DINO weights from a DrivoR Lightning checkpoint."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _strip_prefix(key: str, prefixes: Tuple[str, ...]) -> str:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _assign_scene_embeds(
    mapped: Dict[str, torch.Tensor],
    value: torch.Tensor,
    target: nn.Parameter,
    strict_scene_embeds: bool,
) -> Optional[str]:
    """Map a ckpt ``scene_embeds`` tensor into ``mapped``.

    Args:
        mapped: Destination state-dict fragment (mutated in place).
        value: Checkpoint tensor ``[1, num_cams_ckpt, N, C]``.
        target: Model parameter ``[1, num_cams, N, C]``.
        strict_scene_embeds: If True, raise on incompatible shapes.

    Returns:
        ``None`` on success, ``"skipped"`` if shape mismatched and not strict.
    """
    if value.shape == target.shape:
        mapped["scene_embeds"] = value
        return None
    if (
        value.ndim == target.ndim
        and value.shape[0] == target.shape[0]
        and value.shape[2:] == target.shape[2:]
        and value.shape[1] >= target.shape[1]
    ):
        # Multi-cam ckpt → fewer-cam register: take leading cams (cam_f0 first).
        mapped["scene_embeds"] = value[:, : target.shape[1]].clone()
        return None
    if strict_scene_embeds:
        raise RuntimeError(
            f"scene_embeds shape mismatch: ckpt {tuple(value.shape)} "
            f"vs model {tuple(target.shape)}"
        )
    return "skipped"


def load_drivor_image_backbone(
    dino_register: nn.Module,
    checkpoint_path: str,
    strict_scene_embeds: bool = False,
) -> Dict[str, int]:
    """Map DrivoR ``image_backbone.*`` (+ ``scene_embeds``) into ``DINORegister``.

    DrivoR Lightning keys look like::

        agent._drivor_model.scene_embeds                 # legacy location
        agent._drivor_model.image_backbone.scene_embeds  # post-refactor
        agent._drivor_model.image_backbone.model....
        agent._drivor_model.image_backbone.neck....

    DinoDiT keys look like::

        scene_embeds
        backbone.model....
        backbone.neck....

    If the checkpoint has more cameras than this register, only the leading
    camera slice of ``scene_embeds`` is kept (assumes ``cam_f0`` is first in
    DrivoR ``camera_order``).

    Args:
        dino_register: ``DINORegister`` module to receive weights.
        checkpoint_path: Path to a DrivoR ``.ckpt`` (Lightning) file.
        strict_scene_embeds: If True, require exact ``scene_embeds`` shape.

    Returns:
        Stats dict with ``loaded``, ``skipped``, ``missing_after`` counts.
    """
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    backbone_prefixes = (
        "agent._drivor_model.image_backbone.",
        "agent.image_backbone.",
        "_drivor_model.image_backbone.",
        "image_backbone.",
    )
    # Legacy DrivoR kept register tokens on the model, not under image_backbone.
    # DrivoRModel._load_from_state_dict remaps these; we must do the same.
    scene_embed_keys = (
        "agent._drivor_model.image_backbone.scene_embeds",
        "agent._drivor_model.scene_embeds",
        "agent.image_backbone.scene_embeds",
        "_drivor_model.image_backbone.scene_embeds",
        "_drivor_model.scene_embeds",
        "image_backbone.scene_embeds",
        "scene_embeds",
    )

    mapped: Dict[str, torch.Tensor] = {}
    skipped = 0

    # Prefer post-refactor path; fall back to legacy DrivoRModel.scene_embeds.
    for se_key in scene_embed_keys:
        if se_key not in state_dict:
            continue
        status = _assign_scene_embeds(
            mapped,
            state_dict[se_key],
            dino_register.scene_embeds,
            strict_scene_embeds,
        )
        if status == "skipped":
            skipped += 1
        break

    for key, value in state_dict.items():
        if not any(key.startswith(p) for p in backbone_prefixes):
            # Also accept already-stripped keys that still contain image_backbone.
            if "image_backbone." not in key:
                continue
            # Fallback: take substring after last image_backbone.
            key = key.split("image_backbone.", 1)[-1]
            suffix = key
        else:
            suffix = _strip_prefix(key, backbone_prefixes)

        # Already handled above (legacy or nested scene_embeds).
        if suffix == "scene_embeds" or suffix.startswith("scene_embeds"):
            if "scene_embeds" not in mapped:
                status = _assign_scene_embeds(
                    mapped,
                    value,
                    dino_register.scene_embeds,
                    strict_scene_embeds,
                )
                if status == "skipped":
                    skipped += 1
            continue

        mapped[f"backbone.{suffix}"] = value

    missing, unexpected = dino_register.load_state_dict(mapped, strict=False)
    return {
        "loaded": len(mapped),
        "skipped": skipped,
        "missing_after": len(missing),
        "unexpected_after": len(unexpected),
    }
