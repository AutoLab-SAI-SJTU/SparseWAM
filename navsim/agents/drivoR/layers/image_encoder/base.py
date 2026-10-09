"""Pluggable image-feature-extractor interface for DrivoR.

The original ``DrivoRModel`` had its visual backbone hard-coded to
``ImgEncoder`` (timm DINOv2 + LoRA) and kept the per-camera ``scene_embeds``
learnable parameter at the model level.  This module abstracts that wiring
behind a single interface so alternative backbones can be plugged in without
touching ``DrivoRModel`` itself.

Contract for any extractor
--------------------------
``forward(features: Dict[str, Tensor]) -> Tensor``

The returned tensor has shape ``[B, S, D]`` where:

* ``B`` is the batch size.
* ``S`` is implementation-specific (typically ``num_cams * num_scene_tokens``)
  and is consumed as a flat sequence by the downstream trajectory decoder, so
  the exact value does not need to match across extractors.
* ``D`` must equal ``config.tf_d_model``.

Extractors must also expose a ``num_features`` attribute / property holding
the pre-projection embedding dimension of the underlying backbone (used by
the lidar branch to size its own learnable scene tokens).

Adding a new extractor
----------------------
1. Subclass :class:`BaseImgFeatureExtractor`.
2. Decorate it with ``@register_image_extractor("<name>")``.
3. Import the module from ``layers/image_encoder/__init__.py`` so the
   registry is populated at import time.
4. In the experiment YAML set ``image_backbone.kind: <name>``; omitting the
   field falls back to ``"dinov2_lora"`` and preserves legacy behaviour.
"""

from __future__ import annotations

from typing import Callable, Dict, Type

import torch
import torch.nn as nn


class BaseImgFeatureExtractor(nn.Module):
    """Base class for image feature extractors used by ``DrivoRModel``.

    Subclasses must implement :meth:`forward` and expose ``num_features`` so
    the lidar branch (which currently mirrors the image embed dim) keeps
    working unchanged.
    """

    num_features: int

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError


_REGISTRY: Dict[str, Type[BaseImgFeatureExtractor]] = {}


def register_image_extractor(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Class decorator that registers an extractor under ``name``."""

    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(
                f"Image extractor {name!r} is already registered "
                f"(existing={_REGISTRY[name].__name__}, new={cls.__name__})"
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_image_extractors() -> list[str]:
    """Return the sorted list of registered extractor names."""
    return sorted(_REGISTRY)


def build_image_backbone(config, num_cams: int) -> nn.Module:
    """Instantiate the extractor selected by ``config.image_backbone.kind``.

    A missing ``kind`` field defaults to ``"dinov2_lora"`` so legacy configs
    keep working without any change.
    """
    image_backbone_cfg = config["image_backbone"]
    # Mirror the field back-fill the original DrivoRModel was doing inline.
    image_backbone_cfg["image_size"] = config["image_size"]
    image_backbone_cfg["num_scene_tokens"] = config["num_scene_tokens"]
    image_backbone_cfg["tf_d_model"] = config["tf_d_model"]

    kind = image_backbone_cfg.get("kind", "dinov2_lora")
    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown image_backbone.kind={kind!r}. "
            f"Available: {available_image_extractors()}"
        )
    return _REGISTRY[kind](image_backbone_cfg, num_cams=num_cams)
