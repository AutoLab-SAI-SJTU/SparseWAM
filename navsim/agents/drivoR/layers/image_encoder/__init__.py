"""Pluggable image feature extractors for DrivoR.

See :mod:`base` for the contract every extractor must follow.  Importing
this package eagerly imports each backend module so its
``@register_image_extractor`` side-effects populate the global registry.
"""

from .base import (
    BaseImgFeatureExtractor,
    available_image_extractors,
    build_image_backbone,
    register_image_extractor,
)

# Side-effect imports: each module registers its extractor(s) on import.
from . import dinov2_lora  # noqa: F401

__all__ = [
    "BaseImgFeatureExtractor",
    "available_image_extractors",
    "build_image_backbone",
    "register_image_extractor",
]
