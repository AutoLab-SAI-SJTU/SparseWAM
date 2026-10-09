"""Frozen DrivoR neural Scorer used to rank DinoDiT RF proposals.

DrivoR forward (``drivor_model.py``)::

    embedded = pos_embed(proposals.reshape(B, M, T*3))     # [B, M, 256]
    tok = scorer_attention(embedded, scene_features)       # [B, M, 256]
    tok = tok + ego_token                                  # hist_encoding(ego11)
    pred_logit = scorer(proposals, tok)                    # 6 PDMS sub-heads
    pdm_score = noc*logσ(NC) + dac*logσ(DAC) + ddc*logσ(DDC)
                + log(ttc*σ(TTC) + ep*σ(EP) + comfort*σ(C))
    best = argmax(pdm_score, dim=1)

DinoDiT encodes current-frame 4-cam images with a dedicated DrivoR DINO
register (``scene_embeds`` ``[1, 4, 16, 384]``) so the KV matches Scorer
training: ``[B, 64, 256]``. Planner RF still uses front-only H tokens.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from navsim.agents.drivoR.score_module.scorer import Scorer
from navsim.agents.drivoR.transformer_decoder import TransformerDecoderScorer


def build_scorer_hydra_cfg(config: Dict[str, Any]) -> Any:
    """Build the OmegaConf DrivoR Scorer expects (attribute access).

    Args:
        config: DinoDiT agent ``config`` dict.

    Returns:
        OmegaConf with Scorer / ``TransformerDecoderScorer`` fields.
    """
    return OmegaConf.create(
        {
            "b2d": bool(config.get("b2d", False)),
            "proposal_num": int(
                config.get("proposal_num", config.get("num_proposals", 64))
            ),
            "num_poses": int(config.get("num_poses", 8)),
            "tf_d_model": int(config.get("tf_d_model", 256)),
            "tf_d_ffn": int(config.get("tf_d_ffn", 1024)),
            "double_score": bool(config.get("double_score", False)),
            "agent_pred": bool(config.get("agent_pred", False)),
            "area_pred": bool(config.get("area_pred", False)),
            "bev_map": bool(config.get("bev_map", False)),
            "bev_agent": bool(config.get("bev_agent", False)),
            "one_token_per_traj": bool(config.get("one_token_per_traj", True)),
            "scorer_ref_num": int(config.get("scorer_ref_num", 4)),
            "refiner_num_heads": int(config.get("refiner_num_heads", 1)),
            "refiner_ls_values": float(config.get("refiner_ls_values", 0.0)),
        }
    )


class DrivoRFrozenScorer(nn.Module):
    """``pos_embed`` + ``scorer_attention`` + ``Scorer`` + ``hist_encoding``.

    All weights are loaded from a full DrivoR Lightning ckpt and frozen.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        cfg = build_scorer_hydra_cfg(config)
        d_model = int(cfg.tf_d_model)
        d_ffn = int(cfg.tf_d_ffn)
        num_poses = int(cfg.num_poses)
        self.num_poses = num_poses
        self.d_model = d_model
        self.noc = float(config.get("noc", 1.0))
        self.dac = float(config.get("dac", 1.0))
        self.ddc = float(config.get("ddc", 0.0))
        self.ttc = float(config.get("ttc", 5.0))
        self.ep = float(config.get("ep", 5.0))
        self.comfort = float(config.get("comfort", 2.0))

        self.pos_embed = nn.Sequential(
            nn.Linear(num_poses * 3, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.scorer_attention = TransformerDecoderScorer(
            num_layers=int(cfg.scorer_ref_num),
            d_model=d_model,
            proj_drop=0.1,
            drop_path=0.2,
            config=cfg,
        )
        self.scorer = Scorer(cfg)
        # DrivoR ego is 11-D (pose3 + vel2 + acc2 + cmd4).
        self.hist_encoding = nn.Linear(11, d_model)
        self._loaded = False
        self.freeze()

    def freeze(self) -> None:
        """Disable gradients on every scorer parameter."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_drivor_ckpt(self, checkpoint_path: str) -> Dict[str, int]:
        """Load ``pos_embed`` / ``scorer_attention`` / ``scorer`` / ``hist_encoding``.

        Args:
            checkpoint_path: DrivoR Lightning ``.ckpt`` / ``.pth``.

        Returns:
            Counts of loaded / skipped keys.

        Raises:
            FileNotFoundError: Missing checkpoint.
            RuntimeError: No matching scorer keys in the checkpoint.
        """
        raw = torch.load(checkpoint_path, map_location="cpu")
        state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        prefixes = (
            "agent._drivor_model.",
            "_drivor_model.",
            "agent.agent._drivor_model.",
        )
        mapped: Dict[str, torch.Tensor] = {}
        allowed_roots = {
            "pos_embed",
            "scorer_attention",
            "scorer",
            "hist_encoding",
        }
        for key, value in state.items():
            stripped = key
            for prefix in prefixes:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    break
            root = stripped.split(".", 1)[0]
            if root in allowed_roots:
                mapped[stripped] = value

        dest_keys = set(self.state_dict().keys())
        mapped = {k: v for k, v in mapped.items() if k in dest_keys}
        if not mapped:
            raise RuntimeError(
                f"No DrivoR scorer keys in {checkpoint_path}. "
                "Expected pos_embed / scorer_attention / scorer / hist_encoding."
            )
        incompatible = self.load_state_dict(mapped, strict=False)
        if incompatible.missing_keys:
            preview = ", ".join(incompatible.missing_keys[:8])
            raise RuntimeError(
                f"DrivoR scorer missing {len(incompatible.missing_keys)} keys "
                f"from {checkpoint_path}: {preview}"
            )
        self._loaded = True
        self.freeze()
        return {
            "loaded": len(mapped),
            "missing": 0,
            "unexpected": 0,
        }

    def _ego_token(self, ego_status: torch.Tensor) -> torch.Tensor:
        """Map DinoDiT 8-D ego to DrivoR ``hist_encoding`` token.

        Args:
            ego_status: ``[B, 8]`` ``[vx,vy,ax,ay,cmd0..3]``.

        Returns:
            ``[B, 1, 256]``.
        """
        if ego_status.dim() == 3:
            # Optional history stack ``[B, T, 8]`` → current frame.
            ego_status = ego_status[:, -1]
        batch = ego_status.shape[0]
        # Current-frame origin: pad pose (x, y, heading) with zeros.
        pose = torch.zeros(
            batch, 3, device=ego_status.device, dtype=ego_status.dtype
        )
        ego11 = torch.cat([pose, ego_status], dim=-1)  # [B, 11]
        return self.hist_encoding(ego11)[:, None]  # [B, 1, 256]

    @torch.no_grad()
    def score(
        self,
        proposals: torch.Tensor,
        scene_features: torch.Tensor,
        ego_status: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rank ``M`` trajectories with the frozen DrivoR Scorer.

        Args:
            proposals: Absolute poses ``[B, M, T, 3]`` e.g. ``[1, 64, 8, 3]``.
            scene_features: Current-frame 4-cam DINO tokens ``[B, 64, 256]``
                e.g. ``[1, 64, 256]`` (f0/b0/l0/r0 × 16 slots).
            ego_status: ``[B, 8]``.

        Returns:
            ``pdm_score`` ``[B, M]``, ``best_idx`` ``[B]``.
        """
        batch, num_modes, num_poses, _ = proposals.shape
        if num_poses != self.num_poses:
            raise ValueError(
                f"proposal T={num_poses} != scorer num_poses={self.num_poses}"
            )
        embedded = self.pos_embed(
            proposals.reshape(batch, num_modes, -1).detach()
        )  # [B, M, 256]
        tok = self.scorer_attention(embedded, scene_features)  # [B, M, 256]
        tok = tok + self._ego_token(ego_status)
        pred_logit, *_ = self.scorer(proposals, tok)
        # Same log-sigmoid mix as DrivoRModel.forward.
        pdm_score = (
            self.noc * pred_logit["no_at_fault_collisions"].sigmoid().log()
            + self.dac * pred_logit["drivable_area_compliance"].sigmoid().log()
            + self.ddc
            * pred_logit["driving_direction_compliance"].sigmoid().log()
            + (
                self.ttc
                * pred_logit["time_to_collision_within_bound"].sigmoid()
                + self.ep * pred_logit["ego_progress"].sigmoid()
                + self.comfort * pred_logit["comfort"].sigmoid()
            ).log()
        )
        best_idx = torch.argmax(pdm_score, dim=1)
        return pdm_score, best_idx
