"""DINO register: image → fixed 16 scene tokens.

Reuses the DrivoR ``ImgEncoder`` (timm DINOv2 + optional LoRA) with learnable
register / scene tokens. The whole module is frozen when loading a DrivoR
pretrained checkpoint (see ``SparseWAMModel.load_drivor_backbone``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from navsim.agents.drivoR.layers.image_encoder.dinov2_lora import ImgEncoder


class DINORegister(nn.Module):
    """Extract a fixed number of DINO tokens from each camera image.

    Args:
        config: Backbone config (model_name, weights, lora, image_size, ...).
            Must contain ``num_scene_tokens`` (default 16) and ``tf_d_model``.
            Accepts dict or OmegaConf.
        num_cams: Number of cameras batched in the image tensor.
    """

    def __init__(self, config: Any, num_cams: int = 1) -> None:
        super().__init__()
        if not OmegaConf.is_config(config):
            config = OmegaConf.create(config)
        self.num_cams = num_cams
        self.num_scene_tokens = int(config.get("num_scene_tokens", 16))
        # Force frozen ImgEncoder path: no LoRA train, no finetune.
        # Weights come from DrivoR ckpt; LoRA adapters are still loaded if present.
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        config.finetune = False
        # Keep use_lora as configured so key names match DrivoR checkpoints.
        self.backbone = ImgEncoder(config)
        self.scene_embeds = nn.Parameter(
            torch.randn(
                1,
                num_cams,
                self.num_scene_tokens,
                self.backbone.num_features,
            )
            * 1e-6,
            requires_grad=True,
        )
        self._frozen = False

    def freeze(self) -> None:
        """Freeze all parameters (backbone + register tokens)."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        self._frozen = True

    def train(self, mode: bool = True):
        """Keep frozen modules in eval even if parent calls ``train()``."""
        super().train(mode)
        if self._frozen:
            self.backbone.eval()
            return self
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images into register tokens.

        Args:
            image: ``[B, num_cams, 3, H, W]`` e.g. ``[2, 1, 3, 672, 1148]``.

        Returns:
            DINO tokens ``[B, num_cams * 16, D]`` e.g. ``[2, 16, 256]``.
        """
        batch_size = image.shape[0]
        scene_tokens = self.scene_embeds.expand(batch_size, -1, -1, -1)
        if self._frozen:
            with torch.no_grad():
                return self.backbone(image, scene_tokens)
        return self.backbone(image, scene_tokens)
