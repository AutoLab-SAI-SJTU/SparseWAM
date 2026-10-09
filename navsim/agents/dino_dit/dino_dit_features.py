"""Feature / target builders for DinoDiT.

Features: history + current camera image(s) + ego status.
Targets: future trajectory + future camera images for GT DINO supervision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

# DrivoR nav1 enabled cams in ``default_camera_order`` (drivor_features.py).
_DEFAULT_SCORER_CAMERA_ORDER: Tuple[str, ...] = (
    "cam_f0",
    "cam_b0",
    "cam_l0",
    "cam_r0",
)

import numpy as np
import torch
from PIL import Image

from navsim.common.dataclasses import AgentInput, Scene
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


def _normalize_image(image_hwc: np.ndarray, image_size_wh: Sequence[int]) -> torch.Tensor:
    """Resize and ImageNet-normalize an HWC uint8 image.

    Args:
        image_hwc: ``[H, W, 3]`` numpy image.
        image_size_wh: ``(W, H)`` target size.

    Returns:
        Float tensor ``[3, H, W]``.
    """
    im = Image.fromarray(image_hwc).resize(tuple(image_size_wh))
    arr = np.asarray(im).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1)


def _stack_cameras_at_frame(
    cameras: Any,
    camera_order: Sequence[str],
    config: Dict[str, Any],
) -> torch.Tensor:
    """Stack enabled cameras at one frame into ``[C, 3, H, W]``.

    Args:
        cameras: ``Cameras`` dataclass for a single frame.
        camera_order: Ordered camera names (e.g. ``[cam_f0]``).
        config: Agent config with per-cam enable lists + ``image_size``.

    Returns:
        Image tensor ``[num_cams, 3, H, W]``.
    """
    images = []
    for name in camera_order:
        if len(config.get(name, [])) == 0:
            continue
        cam = getattr(cameras, name)
        if cam is None or cam.image is None:
            continue
        images.append(_normalize_image(cam.image, config["image_size"]))
    if not images:
        raise ValueError("No camera images found for frame.")
    return torch.stack(images)


def _stack_named_cameras(
    cameras: Any,
    camera_names: Sequence[str],
    image_size_wh: Sequence[int],
) -> torch.Tensor:
    """Stack named cameras without consulting per-cam enable lists.

    Planner ``camera_order`` stays front-only. Scorer needs DrivoR's four
    current-frame views even when ``cam_l0`` / ``cam_r0`` / ``cam_b0`` are
    empty in the DinoDiT yaml.

    Args:
        cameras: ``Cameras`` dataclass for a single frame.
        camera_names: Ordered names, e.g. ``[cam_f0, cam_b0, cam_l0, cam_r0]``.
        image_size_wh: ``(W, H)`` target size.

    Returns:
        Image tensor ``[num_cams, 3, H, W]`` e.g. ``[4, 3, 672, 1148]``.
    """
    images = []
    width, height = int(image_size_wh[0]), int(image_size_wh[1])
    for name in camera_names:
        cam = getattr(cameras, name, None)
        if cam is None or getattr(cam, "image", None) is None:
            # Keep a fixed 4-cam layout so DDP ranks never desync on a missing
            # side/rear blob. Zero image → zero DINO tokens for that cam.
            images.append(torch.zeros(3, height, width, dtype=torch.float32))
            continue
        images.append(_normalize_image(cam.image, image_size_wh))
    if not images:
        raise ValueError("No scorer camera images found for current frame.")
    return torch.stack(images)


class DinoDiTFeatureBuilder(AbstractFeatureBuilder):
    """Build history + current frame images + ego status features."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        # Relative frame offsets for history DINO conditions (2Hz).
        # Default ``[-2]`` → -1.0s. Empty → current only.
        self._history_frame_offsets: List[int] = [
            int(x)
            for x in list(
                config.get("history_frame_offsets", [-2])
            )
        ]
        for off in self._history_frame_offsets:
            if off >= 0:
                raise ValueError(
                    f"history_frame_offsets must be negative, got {off}."
                )

    def get_unique_name(self) -> str:
        return "dino_dit_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Build condition images ``[T_cond, C, 3, H, W]`` + ego status.

        ``T_cond = len(history_frame_offsets) + 1`` (history…, then current).
        ``AgentInput.cameras`` is indexed in ``[0 .. num_history_frames)`` with
        current at ``[-1]``.

        Returns:
            Dict with ``image`` ``[T_cond, C, 3, H, W]``, ``ego_status`` ``[8]``.
            When ``use_drivor_scorer``, also ``scorer_image``
            ``[4, 3, H, W]`` (current-frame f0/b0/l0/r0).
        """
        camera_order = list(self._config.get("camera_order", ["cam_f0"]))
        current_idx = len(agent_input.cameras) - 1
        frame_indices = [current_idx + off for off in self._history_frame_offsets]
        frame_indices.append(current_idx)

        images_t = []
        for frame_idx in frame_indices:
            if frame_idx < 0 or frame_idx >= len(agent_input.cameras):
                raise IndexError(
                    f"History/current frame index {frame_idx} out of range "
                    f"(num_history={len(agent_input.cameras)}, "
                    f"history_frame_offsets={self._history_frame_offsets}). "
                    "Increase scene_filter.num_history_frames."
                )
            images_t.append(
                _stack_cameras_at_frame(
                    agent_input.cameras[frame_idx], camera_order, self._config
                )
            )
            # [T_cond, C, 3, H, W] e.g. [2, 1, 3, 672, 1148] for hist(-1.0s)+current.
        image = torch.stack(images_t)

        ego = agent_input.ego_statuses[-1]
        ego_status = torch.cat(
            [
                torch.tensor(ego.ego_velocity, dtype=torch.float32),
                torch.tensor(ego.ego_acceleration, dtype=torch.float32),
                torch.tensor(ego.driving_command, dtype=torch.float32),
            ],
            dim=-1,
        )
        out = {
            "image": image,  # [T_cond, C, 3, H, W]
            "ego_status": ego_status,  # [8]
        }
        if bool(self._config.get("use_drivor_scorer", False)):
            scorer_order = list(
                self._config.get(
                    "scorer_camera_order", list(_DEFAULT_SCORER_CAMERA_ORDER)
                )
            )
            # Current-frame 4-cam for Scorer KV; planner ``image`` stays 1-cam.
            # [4, 3, H, W] e.g. [4, 3, 672, 1148]
            out["scorer_image"] = _stack_named_cameras(
                agent_input.cameras[current_idx],
                scorer_order,
                self._config["image_size"],
            )
        return out


class DinoDiTTargetBuilder(AbstractTargetBuilder):
    """Build future trajectory + future images for DINO GT."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        # Frame offsets relative to the current frame, at dataset FPS (2Hz).
        # Default: +2s → +4 frames, +4s → +8 frames.
        self._future_frame_offsets: List[int] = list(
            config.get("future_frame_offsets", [4, 8])
        )
        self._num_poses = int(config.get("num_poses", 8))

    def get_unique_name(self) -> str:
        return "dino_dit_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_traj = scene.get_future_trajectory(
            num_trajectory_frames=self._num_poses
        )
        trajectory = torch.tensor(future_traj.poses, dtype=torch.float32)

        current_idx = scene.scene_metadata.num_history_frames - 1
        camera_order = list(self._config.get("camera_order", ["cam_f0"]))
        future_images = []
        for offset in self._future_frame_offsets:
            frame_idx = current_idx + int(offset)
            if frame_idx >= len(scene.frames):
                raise IndexError(
                    f"Future frame index {frame_idx} out of range "
                    f"(num_frames={len(scene.frames)}, offset={offset}). "
                    "Increase scene_filter.num_future_frames or lower offsets."
                )
            future_images.append(
                _stack_cameras_at_frame(
                    scene.frames[frame_idx].cameras, camera_order, self._config
                )
            )

        # [num_future, num_cams, 3, H, W]
        future_image = torch.stack(future_images)
        return {
            "trajectory": trajectory,  # [T, 3]
            "future_image": future_image,
            "token": scene.scene_metadata.initial_token,
        }
