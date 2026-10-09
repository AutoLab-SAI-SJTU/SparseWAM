"""DinoDiT core model: MoT video + action Rectified Flow.

Default layout (hist ``-2`` + current, 2 future frames, 8 poses)::

    Joint sequence ``[H | F | A]`` (True=attend)::

        H  hist_-2 + current     32 tokens, clean, t=0
        F  future scene RF       32 tokens, noisy
        A  action poses          8 tokens, noisy

    Video expert (dim=512) owns ``[H | F]``.
    Action expert (dim=256) owns ``A``.
    Joint attention width = n_heads * head_dim = 12 * 64 = 768.

    Mask:
        H → H only;  F → H+F;  A → H+F+A.

AdaLN condition is diffusion t (+ ego). Time uses a **shared 1D RoPE**
in 2Hz frame indices (video ``{-2,0,4,8}``, action pose ``k`` → ``k+1``).
Register slot uses a **fixed sinusoidal** identity (not RoPE, not learned).
DINO is frozen; RF loss is only on F and A.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .layers.backbone_loader import load_drivor_image_backbone
from .layers.dino_register import DINORegister
from .layers.dit import Expert, build_rope_freqs_1d, sinusoidal_embedding_1d
from .layers.drivor_scorer import DrivoRFrozenScorer
from .layers.flow_match import FlowMatchScheduler
from .layers.mot import MoT
from .layers.trajectory_modules import (
    EgoStatusEncoder,
    TrajectoryDecoder,
    TrajectoryEncoder,
    decode_trajectory_from_flow,
    encode_trajectory_for_flow,
    expand_trajectories_mul_noise,
)


class DinoDiTModel(nn.Module):
    """Joint scene-query / action Rectified-Flow denoiser with MoT experts."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        # DrivoR / DINO token space (frozen neck output dim).
        self.dim = int(config["tf_d_model"])
        # Video expert hidden size; also the default joint-attn width.
        self.dit_dim = int(config.get("dit_dim", self.dim))
        self.video_dim = self.dit_dim
        self.action_dim = int(config.get("action_dit_dim", self.dim))
        if self.video_dim < self.dim:
            raise ValueError(
                f"dit_dim / video_dim ({self.video_dim}) must be >= tf_d_model "
                f"({self.dim})."
            )
        self.num_scene_tokens = int(config.get("num_scene_tokens", 16))
        self.num_poses = int(config.get("num_poses", 8))
        # Default: a single history frame at -1.0s @ 2Hz.
        self.history_frame_offsets: List[int] = [
            int(x) for x in list(config.get("history_frame_offsets", [-2]))
        ]
        for off in self.history_frame_offsets:
            if off >= 0:
                raise ValueError(
                    f"history_frame_offsets must be negative, got {off}."
                )
        self.future_frame_offsets: List[int] = [
            int(x) for x in list(config.get("future_frame_offsets", [4, 8]))
        ]
        self.num_history = len(self.history_frame_offsets)
        self.num_future = len(self.future_frame_offsets)
        self.num_cond_frames = self.num_history + 1
        self.fuse_history = bool(config.get("fuse_history", False))
        if self.fuse_history:
            raise ValueError(
                "fuse_history=True is not supported. History+current tokens "
                "are packed into the MoT video sequence with shared 1D "
                "frame-time RoPE and fixed sinusoidal slot PE."
            )
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        self.num_inference_steps = int(config.get("num_inference_steps", 10))
        self.flow_shift = float(config.get("flow_shift", 1.0))
        # v3_a1 default: F shift=3, A shift=1 (navtest PDMS 0.891).
        # ``null`` used to inherit ``flow_shift``; this branch never does.
        video_shift_cfg = config.get("video_flow_shift", 3)
        action_shift_cfg = config.get("action_flow_shift", 1)
        self.video_flow_shift = (
            3.0 if video_shift_cfg is None else float(video_shift_cfg)
        )
        self.action_flow_shift = (
            1.0 if action_shift_cfg is None else float(action_shift_cfg)
        )
        # Eval-only neural Scorer: sample ``num_proposals`` RF trajectories then
        # rank with a frozen DrivoR Scorer. Train keeps 1 sample / scorer off.
        # Scorer KV = current-frame 4-cam DINO from the DrivoR ckpt backbone.
        # Optional: DiffusionDriveV2 mul-noise copies around each RF sample.
        self.num_proposals = int(config.get("num_proposals", 1))
        self.use_drivor_scorer = bool(config.get("use_drivor_scorer", False))
        # Max expanded RF batch (original B × chunk). Eval B=32 → chunk=1
        # unless the eval script lowers dataloader batch_size.
        self.proposal_chunk_size = int(config.get("proposal_chunk_size", 32))
        self.proposal_mul_noise_copies = int(
            config.get("proposal_mul_noise_copies", 0)
        )
        self.proposal_mul_noise_std = float(
            config.get("proposal_mul_noise_std", 0.1)
        )
        self.scorer_camera_order = list(
            config.get(
                "scorer_camera_order",
                ["cam_f0", "cam_b0", "cam_l0", "cam_r0"],
            )
        )
        self.scorer_num_cams = len(self.scorer_camera_order)
        self.drivor_scorer: Optional[DrivoRFrozenScorer] = None
        self.scorer_dino_register: Optional[DINORegister] = None
        if self.use_drivor_scorer:
            self.drivor_scorer = DrivoRFrozenScorer(config)

        camera_order = list(config.get("camera_order", ["cam_f0"]))
        self.num_cams = sum(
            1 for name in camera_order if len(config.get(name, [])) > 0
        )
        if self.num_cams < 1:
            raise ValueError("DinoDiTModel requires at least one enabled camera.")

        backbone_cfg = OmegaConf.create(dict(config["image_backbone"]))
        backbone_cfg.image_size = config["image_size"]
        backbone_cfg.num_scene_tokens = self.num_scene_tokens
        backbone_cfg.tf_d_model = self.dim
        backbone_cfg.setdefault("use_feature_pooling", False)
        backbone_cfg.setdefault("focus_front_cam", False)
        backbone_cfg.setdefault("compress_fc", False)
        backbone_cfg.setdefault("use_lora", True)
        backbone_cfg.setdefault("finetune", False)
        backbone_cfg.setdefault("lora_rank", 32)

        self.dino_register = DINORegister(backbone_cfg, num_cams=self.num_cams)
        self.dino_register.freeze()
        if self.use_drivor_scorer:
            # Separate 4-cam register so planner stays 1-cam and Scorer KV
            # uses the full DrivoR ``scene_embeds`` [1, 4, 16, 384].
            self.scorer_dino_register = DINORegister(
                backbone_cfg, num_cams=self.scorer_num_cams
            )
            self.scorer_dino_register.freeze()

        self.ego_encoder = EgoStatusEncoder(
            in_dim=int(config.get("ego_status_dim", 8)),
            output_dim=self.video_dim,
        )
        self.ego_to_action = (
            nn.Identity()
            if self.action_dim == self.video_dim
            else nn.Linear(self.video_dim, self.action_dim)
        )
        self.traj_encoder = TrajectoryEncoder(
            point_dim=3,
            hidden_dim=int(config.get("tf_d_ffn", 1024)),
            output_dim=self.action_dim,
            sine_dim=int(config.get("traj_sine_dim", 64)),
            encode_heading=bool(config.get("traj_sine_encode_heading", True)),
            use_sine_embed=bool(config.get("traj_use_sine_embed", True)),
        )
        self.traj_decoder = TrajectoryDecoder(dim=self.action_dim, out_dim=3)

        self.len_obs_per_frame = self.num_cams * self.num_scene_tokens
        # Clean visual condition: history + current.
        self.len_h = self.num_cond_frames * self.len_obs_per_frame
        self.len_ctx = self.len_h
        self.len_obs = self.len_h
        self.len_scene = self.num_future * self.len_obs_per_frame
        self.len_action = self.num_poses
        self.len_video = self.len_h + self.len_scene
        self.seq_len = self.len_video + self.len_action

        self.token_up = (
            nn.Identity()
            if self.video_dim == self.dim
            else nn.Linear(self.dim, self.video_dim)
        )
        self.scene_down = (
            nn.Identity()
            if self.video_dim == self.dim
            else nn.Linear(self.video_dim, self.dim)
        )

        def _visual_slot_ids() -> List[int]:
            ids: List[int] = []
            for _ in range(self.num_cams):
                ids.extend(range(self.num_scene_tokens))
            return ids

        # Shared 1D RoPE time axis: 2Hz frame offsets (not pose indices).
        # Video default {-2, 0, 4, 8}; action pose k → frame k+1
        # (pose0 = +0.5s = +1 frame), so F@+2s (t=4) aligns with pose3.
        video_time: List[float] = []
        for offset in list(self.history_frame_offsets) + [0.0]:
            video_time.extend([float(offset)] * self.len_obs_per_frame)
        for offset in self.future_frame_offsets:
            video_time.extend([float(offset)] * self.len_obs_per_frame)
        assert len(video_time) == self.len_video
        video_time_ids = torch.tensor(video_time, dtype=torch.float32)
        self.register_buffer("video_time_ids", video_time_ids, persistent=False)

        # Slot ids 0..K-1, repeating per frame / camera. Fixed sinusoidal PE
        # [Sv, Dv] e.g. [64, 512]; not RoPE and not a learned table.
        video_slot: List[int] = []
        for _ in range(self.num_cond_frames + self.num_future):
            video_slot.extend(_visual_slot_ids())
        assert len(video_slot) == self.len_video
        video_slot_ids = torch.tensor(video_slot, dtype=torch.float32)
        self.register_buffer("video_slot_ids", video_slot_ids, persistent=False)
        self.register_buffer(
            "video_slot_pe",
            sinusoidal_embedding_1d(self.video_dim, video_slot_ids),
            persistent=False,
        )

        # Action time on the same 2Hz axis: t = k+1, e.g. [8] → 1..8.
        action_time_ids = torch.arange(
            self.num_poses, dtype=torch.float32
        ) + 1.0
        self.register_buffer("action_time_ids", action_time_ids, persistent=False)

        dit_num_heads = int(config.get("dit_num_heads", 12))
        attn_head_dim = int(
            config.get("dit_attn_head_dim", self.video_dim // dit_num_heads)
        )
        self.register_buffer(
            "video_rope_freqs",
            build_rope_freqs_1d(video_time_ids, attn_head_dim),
            persistent=False,
        )
        self.register_buffer(
            "action_rope_freqs",
            build_rope_freqs_1d(action_time_ids, attn_head_dim),
            persistent=False,
        )
        # Joint QKV width = n_heads * head_dim e.g. 12×64=768; need not equal
        # video hidden dit_dim e.g. 512. Expert Linear projects dim ↔ attn_dim.
        depth = int(config.get("dit_depth", 6))
        mlp_ratio = float(config.get("dit_mlp_ratio", 4.0))
        dropout = float(config.get("dit_dropout", 0.0))
        time_freq_dim = int(config.get("time_freq_dim", 256))
        # 0 / omitted → full Linear(D, 6D); >0 → bottleneck Linear(D, r)×Linear(r, 6D).
        adaln_rank = int(config.get("adaln_rank", 0))
        video_expert = Expert(
            dim=self.video_dim,
            depth=depth,
            num_heads=dit_num_heads,
            attn_head_dim=attn_head_dim,
            mlp_ratio=mlp_ratio,
            time_freq_dim=time_freq_dim,
            dropout=dropout,
            adaln_rank=adaln_rank,
        )
        action_expert = Expert(
            dim=self.action_dim,
            depth=depth,
            num_heads=dit_num_heads,
            attn_head_dim=attn_head_dim,
            mlp_ratio=mlp_ratio,
            time_freq_dim=time_freq_dim,
            dropout=dropout,
            adaln_rank=adaln_rank,
        )
        self.mot = MoT(video_expert=video_expert, action_expert=action_expert)
        # Keep names for param counting / freeze logic.
        self.video_expert = video_expert
        self.action_expert = action_expert

        allow = self._build_mot_attention_mask()
        self.register_buffer("mot_allow_mask", allow, persistent=False)

        loss_weight_mode = str(config.get("flow_loss_weight_mode", "beta"))
        beta_alpha = float(config.get("flow_beta_alpha", 1.5))
        beta_beta = float(config.get("flow_beta_beta", 1.0))
        # Same grid index u; σ_r = s_r·u / (1+(s_r-1)·u) per branch.
        self.video_scheduler = self._make_flow_scheduler(
            self.video_flow_shift,
            self.num_train_timesteps,
            training=True,
            loss_weight_mode=loss_weight_mode,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
        )
        self.action_scheduler = self._make_flow_scheduler(
            self.action_flow_shift,
            self.num_train_timesteps,
            training=True,
            loss_weight_mode=loss_weight_mode,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
        )
        self.scheduler = self.video_scheduler
        # Shift lives in Hydra config, not in the Lightning ckpt. Log the
        # constructed σ grids so eval cannot silently inherit flow_shift=1.
        print(
            "[DinoDiT] RF schedule "
            f"video_shift={self.video_flow_shift} "
            f"action_shift={self.action_flow_shift} "
            f"nfe={self.num_inference_steps} "
            f"heads={dit_num_heads}x{attn_head_dim} adaln_rank={adaln_rank}"
        )

    def _make_flow_scheduler(
        self,
        shift: float,
        num_steps: int,
        training: bool,
        loss_weight_mode: str = "uniform",
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
    ) -> FlowMatchScheduler:
        """Build a DriveVA RF scheduler and freeze its σ grid.

        Args:
            shift: SNR shift s in σ = s·u / (1+(s-1)·u).
            num_steps: Train grid size or inference NFE.
            training: If True, also build per-timestep loss weights.
            loss_weight_mode: ``beta`` or ``uniform``.
            beta_alpha / beta_beta: Beta(α,β) on σ when mode is beta.

        Returns:
            Scheduler with ``timesteps`` of length ``num_steps``.
        """
        sched = FlowMatchScheduler(
            num_inference_steps=num_steps,
            num_train_timesteps=self.num_train_timesteps,
            shift=shift,
            extra_one_step=True,
            loss_weight_mode=loss_weight_mode,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
        )
        sched.set_timesteps(num_steps, training=training, shift=shift)
        return sched

    def _build_mot_attention_mask(self) -> torch.Tensor:
        """Build True=attend joint mask for ``[H | F | A]``.

        Returns:
            Bool mask ``[S, S]`` e.g. ``[72, 72]``.
        """
        len_h = self.len_h
        len_f = self.len_scene
        len_a = self.len_action
        total = len_h + len_f + len_a
        mask = torch.zeros((total, total), dtype=torch.bool)
        # H → H only.
        mask[:len_h, :len_h] = True
        # F → H + F (not action).
        mask[len_h : len_h + len_f, : len_h + len_f] = True
        # A → all.
        mask[len_h + len_f :, :] = True
        return mask

    def load_drivor_backbone(self, checkpoint_path: str) -> Dict[str, int]:
        """Load and freeze DrivoR image-backbone weights into ``dino_register``."""
        stats = load_drivor_image_backbone(self.dino_register, checkpoint_path)
        self.dino_register.freeze()
        return stats

    def load_drivor_scorer(self, checkpoint_path: str) -> Dict[str, int]:
        """Load frozen DrivoR Scorer heads + 4-cam DINO from a Lightning ckpt.

        Args:
            checkpoint_path: DrivoR ``.ckpt`` / ``.pth`` containing
                ``agent._drivor_model.{pos_embed,scorer_attention,scorer,
                hist_encoding,scene_embeds,image_backbone}``.

        Returns:
            Loaded / missing / unexpected key counts, plus backbone stats.
        """
        if self.drivor_scorer is None:
            self.drivor_scorer = DrivoRFrozenScorer(self.config)
            self.use_drivor_scorer = True
        stats = self.drivor_scorer.load_from_drivor_ckpt(checkpoint_path)
        if stats["missing"] > 0:
            raise RuntimeError(
                f"DrivoR scorer missing {stats['missing']} keys from "
                f"{checkpoint_path}. Check scorer_ref_num / tf_d_model."
            )
        if self.scorer_dino_register is None:
            backbone_cfg = OmegaConf.create(dict(self.config["image_backbone"]))
            backbone_cfg.image_size = self.config["image_size"]
            backbone_cfg.num_scene_tokens = self.num_scene_tokens
            backbone_cfg.tf_d_model = self.dim
            self.scorer_dino_register = DINORegister(
                backbone_cfg, num_cams=self.scorer_num_cams
            )
        # Full 4-cam ``scene_embeds`` [1, 4, 16, 384]; do not slice to 1-cam.
        bb_stats = load_drivor_image_backbone(
            self.scorer_dino_register,
            checkpoint_path,
            strict_scene_embeds=True,
        )
        if int(bb_stats["missing_after"]) > 0:
            raise RuntimeError(
                f"DrivoR 4-cam scorer backbone missing "
                f"{bb_stats['missing_after']} keys from {checkpoint_path}."
            )
        self.scorer_dino_register.freeze()
        stats["backbone_loaded"] = int(bb_stats["loaded"])
        stats["backbone_skipped"] = int(bb_stats["skipped"])
        stats["backbone_missing"] = int(bb_stats["missing_after"])
        return stats

    def train(self, mode: bool = True) -> "DinoDiTModel":
        """Keep the frozen Scorer / 4-cam DINO in eval even when training."""
        super().train(mode)
        if self.drivor_scorer is not None:
            self.drivor_scorer.eval()
        if self.scorer_dino_register is not None:
            self.scorer_dino_register.eval()
        return self

    def _encode_obs(self, image: torch.Tensor) -> torch.Tensor:
        """Encode history + current images into clean H tokens.

        Args:
            image: ``[B, T_cond, C, 3, H, W]`` or legacy ``[B, C, 3, H, W]``.

        Returns:
            ``[B, len_h, dim]`` e.g. ``[2, 32, 256]`` for 2 frames × 16 slots.
        """
        if image.dim() == 5:
            image = image.unsqueeze(1)
        if image.dim() != 6:
            raise ValueError(
                f"Expected image [B,T,C,3,H,W] or [B,C,3,H,W], got {tuple(image.shape)}"
            )
        batch, num_frames, num_cams, channels, height, width = image.shape
        if num_frames != self.num_cond_frames:
            raise ValueError(
                f"image T_cond={num_frames} != num_cond_frames={self.num_cond_frames} "
                f"(history_frame_offsets={self.history_frame_offsets})."
            )
        flat = image.reshape(
            batch * num_frames, num_cams, channels, height, width
        )
        tokens = self.dino_register(flat).reshape(batch, self.len_h, self.dim)
        return tokens

    def _encode_scorer_obs(self, scorer_image: torch.Tensor) -> torch.Tensor:
        """Encode current-frame 4-cam images with the DrivoR Scorer backbone.

        Args:
            scorer_image: ``[B, 4, 3, H, W]`` e.g. ``[4, 4, 3, 672, 1148]``
                in order ``cam_f0, cam_b0, cam_l0, cam_r0``.

        Returns:
            Scene tokens ``[B, 64, 256]`` (4 cams × 16 slots).
        """
        if self.scorer_dino_register is None:
            raise RuntimeError(
                "scorer_dino_register is None; call load_drivor_scorer() first."
            )
        if scorer_image.dim() != 5:
            raise ValueError(
                f"scorer_image must be [B, 4, 3, H, W], got {tuple(scorer_image.shape)}"
            )
        num_cams = scorer_image.shape[1]
        if num_cams != self.scorer_num_cams:
            raise ValueError(
                f"scorer_image num_cams={num_cams} != "
                f"scorer_num_cams={self.scorer_num_cams} "
                f"(order={self.scorer_camera_order})."
            )
        # [B, 4*16, 256] e.g. [4, 64, 256]
        return self.scorer_dino_register(scorer_image)

    def _encode_future_gt(self, future_image: torch.Tensor) -> torch.Tensor:
        """Encode GT future images into DINO queries.

        Args:
            future_image: ``[B, num_future, C, 3, H, W]``.

        Returns:
            ``[B, num_future, len_obs_per_frame, dim]`` e.g. ``[2, 2, 16, 256]``.
        """
        batch, num_future, num_cams, channels, height, width = future_image.shape
        flat = future_image.reshape(
            batch * num_future, num_cams, channels, height, width
        )
        tokens = self.dino_register(flat)
        return tokens.reshape(
            batch, num_future, self.len_obs_per_frame, self.dim
        )

    def _dit_velocity(
        self,
        dino_cond: torch.Tensor,
        noisy_scenes: torch.Tensor,
        ego_emb: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        return_attn: bool = False,
    ):
        """Predict RF velocity for future scene tokens and action poses.

        Args:
            dino_cond: Clean H tokens ``[B, len_h, dim]`` e.g. ``[2, 32, 256]``.
            noisy_scenes: Noisy F queries
                ``[B, F, len_obs_per_frame, dim]`` e.g. ``[2, 2, 16, 256]``.
            ego_emb: Video AdaLN ego condition ``[B, Dv]`` e.g. ``[2, 512]``.
            noisy_action: Noisy traj ``[B, T, 3]`` in z-scored relative space.
            timestep_video: F-branch RF steps ``[B]`` e.g. ``[2]`` (H is t=0).
            timestep_action: A-branch RF steps ``[B]`` e.g. ``[2]``.
            return_attn: If True, also return per-layer joint maps.

        Returns:
            scene_vel ``[B, F, K, dim]``, action_vel ``[B, T, 3]``.
            When ``return_attn``, also ``attn_maps`` (list of ``[B, S, S]``).
        """
        batch = dino_cond.shape[0]
        h_tok = self.token_up(dino_cond)  # [B, H, Dv]
        f_tok = self.token_up(
            noisy_scenes.reshape(batch, self.len_scene, self.dim)
        )  # [B, Ftok, Dv]
        video_tokens = torch.cat([h_tok, f_tok], dim=1)  # [B, Sv, Dv] e.g. [2, 64, 512]
        # Fixed sinusoidal slot PE [Sv, Dv] e.g. [64, 512], added before MoT
        # QKV (standard Transformer PE). Not learned, so it does not
        # co-adapt with W_q/W_k. Time stays RoPE on Q/K after the linear.
        video_tokens = video_tokens + self.video_slot_pe
        action_tokens = self.traj_encoder(noisy_action)  # [B, Ta, Da] e.g. [2, 8, 256]

        # H uses t=0; F / A use their own RF timesteps (may differ by shift).
        t_video = torch.zeros(
            batch,
            self.len_video,
            device=timestep_video.device,
            dtype=timestep_video.dtype,
        )
        t_video[:, self.len_h :] = timestep_video.unsqueeze(1).expand(
            -1, self.len_scene
        )
        video_t_emb = self.video_expert.embed_timestep(t_video)
        video_t_emb = video_t_emb + ego_emb.unsqueeze(1)
        action_t_emb = self.action_expert.embed_timestep(timestep_action)
        action_t_emb = action_t_emb + self.ego_to_action(ego_emb)

        allow = self.mot_allow_mask.to(device=video_tokens.device)
        video_rope = self.video_rope_freqs.to(device=video_tokens.device)
        action_rope = self.action_rope_freqs.to(device=action_tokens.device)
        if return_attn:
            video_out, action_out, attn_maps = self.mot(
                video_tokens,
                action_tokens,
                video_t_emb,
                action_t_emb,
                allow,
                video_rope,
                action_rope,
                return_attn=True,
            )
        else:
            video_out, action_out = self.mot(
                video_tokens,
                action_tokens,
                video_t_emb,
                action_t_emb,
                allow,
                video_rope,
                action_rope,
                return_attn=False,
            )
            attn_maps = None

        # RF velocity is only defined on noisy F (drop clean H).
        scene_tok = video_out[:, self.len_h :]
        scene_tok = scene_tok.reshape(
            batch, self.num_future, self.len_obs_per_frame, self.video_dim
        )
        scene_vel = self.scene_down(scene_tok)
        action_vel = self.traj_decoder(action_out)
        if return_attn:
            return scene_vel, action_vel, attn_maps
        return scene_vel, action_vel

    def forward_train(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Rectified-Flow training step.

        Args:
            features: ``image`` ``[B,T_cond,C,3,H,W]``, ``ego_status`` ``[B,8]``.
            targets: ``future_image`` ``[B,F,C,3,H,W]``, ``trajectory`` ``[B,T,3]``.

        Returns:
            Dict with predicted velocities and RF targets for the loss.
        """
        image = features["image"]
        ego_status = features["ego_status"]
        batch = image.shape[0]
        device = image.device

        dino_cond = self._encode_obs(image)
        gt_future_dino = self._encode_future_gt(targets["future_image"])
        gt_action_abs = targets["trajectory"]
        gt_action = encode_trajectory_for_flow(gt_action_abs)

        # Shared grid index u; each branch maps u with its own shift.
        timestep_ids = torch.randint(
            0, len(self.video_scheduler.timesteps), (batch,), device=device
        )
        timestep_video = self.video_scheduler.timesteps.to(device)[timestep_ids]
        timestep_action = self.action_scheduler.timesteps.to(device)[timestep_ids]

        noise_scene = torch.randn_like(gt_future_dino)
        noise_action = torch.randn_like(gt_action)
        noisy_scenes = self.video_scheduler.add_noise(
            gt_future_dino, noise_scene, timestep_video
        )
        noisy_action = self.action_scheduler.add_noise(
            gt_action, noise_action, timestep_action
        )

        target_scene_vel = self.video_scheduler.training_target(
            gt_future_dino, noise_scene, timestep_video
        )
        target_action_vel = self.action_scheduler.training_target(
            gt_action, noise_action, timestep_action
        )
        loss_weight = self.action_scheduler.training_weight(timestep_action)

        ego_emb = self.ego_encoder(ego_status).squeeze(1)
        pred_scene_vel, pred_action_vel = self._dit_velocity(
            dino_cond,
            noisy_scenes,
            ego_emb,
            noisy_action,
            timestep_video,
            timestep_action,
        )

        return {
            "pred_scene_vel": pred_scene_vel,
            "pred_action_vel": pred_action_vel,
            "target_scene_vel": target_scene_vel.detach(),
            "target_action_vel": target_action_vel.detach(),
            "loss_weight": loss_weight.detach(),
            "gt_future_dino": gt_future_dino.detach(),
            "timestep": timestep_video,
            "timestep_action": timestep_action.detach(),
            "trajectory": gt_action_abs.detach(),
        }

    def _rf_denoise(
        self,
        dino_cond: torch.Tensor,
        ego_emb: torch.Tensor,
        noisy_scenes: torch.Tensor,
        noisy_action: torch.Tensor,
        infer_video: FlowMatchScheduler,
        infer_action: FlowMatchScheduler,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Run the Euler RF loop on an already-encoded batch.

        Args:
            dino_cond: Clean H tokens ``[B, len_h, 256]`` e.g. ``[4, 32, 256]``.
            ego_emb: AdaLN ego ``[B, Dv]`` e.g. ``[4, 512]``.
            noisy_scenes: F prior ``[B, F, K, 256]`` e.g. ``[4, 2, 16, 256]``.
            noisy_action: A prior in z-scored relative space ``[B, T, 3]``
                e.g. ``[4, 8, 3]``.
            infer_video / infer_action: Frozen σ grids for F / A.
            return_attn: Collect joint MoT maps at every step.

        Returns:
            Denoised F ``[B, F, K, 256]``, denoised A ``[B, T, 3]``,
            optional per-step attn list.
        """
        batch = dino_cond.shape[0]
        device = dino_cond.device
        step_self: List[torch.Tensor] = []
        n_steps = len(infer_video.timesteps)
        for i, (t_video_s, t_action_s) in enumerate(
            zip(infer_video.timesteps, infer_action.timesteps)
        ):
            t_video = t_video_s.expand(batch).to(device=device)
            t_action = t_action_s.expand(batch).to(device=device)
            if return_attn:
                pred_scene_vel, pred_action_vel, attn_maps = self._dit_velocity(
                    dino_cond,
                    noisy_scenes,
                    ego_emb,
                    noisy_action,
                    t_video,
                    t_action,
                    return_attn=True,
                )
                step_self.append(torch.stack(attn_maps, dim=0))
            else:
                pred_scene_vel, pred_action_vel = self._dit_velocity(
                    dino_cond,
                    noisy_scenes,
                    ego_emb,
                    noisy_action,
                    t_video,
                    t_action,
                )
            to_final = i == n_steps - 1
            noisy_scenes = infer_video.step(
                pred_scene_vel, t_video_s, noisy_scenes, to_final=to_final
            )
            noisy_action = infer_action.step(
                pred_action_vel, t_action_s, noisy_action, to_final=to_final
            )
        return noisy_scenes, noisy_action, step_self

    @torch.no_grad()
    def forward_infer(
        self,
        features: Dict[str, torch.Tensor],
        return_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Multi-step Rectified-Flow sampling from pure noise.

        When ``use_drivor_scorer`` and ``num_proposals>1``, encodes H once,
        draws ``num_proposals`` independent RF samples (chunked so expanded
        batch stays ``<= proposal_chunk_size``), optionally expands each
        sample with DiffusionDriveV2 multiplicative long/lat noise
        (``proposal_mul_noise_copies`` copies at ``proposal_mul_noise_std``),
        ranks the candidates with the frozen DrivoR Scorer whose KV is
        current-frame 4-cam DINO (not planner H tokens), and returns the
        argmax trajectory.

        Args:
            features: ``image`` ``[B,T_cond,C,3,H,W]``, ``ego_status`` ``[B,8]``.
            return_attn: If True, collect joint MoT attention at every RF step
                (single-sample path only).

        Returns:
            Dict with ``trajectory`` ``[B, T, 3]``. Scorer eval also returns
            ``proposals`` ``[B, M, T, 3]`` and ``pdm_score`` ``[B, M]``.
        """
        image = features["image"]
        ego_status = features["ego_status"]
        batch = image.shape[0]
        device = image.device
        dtype = image.dtype

        dino_cond = self._encode_obs(image)  # [B, len_h, 256] e.g. [4, 32, 256]
        ego_emb = self.ego_encoder(ego_status).squeeze(1)  # [B, Dv] e.g. [4, 512]

        infer_video = self._make_flow_scheduler(
            self.video_flow_shift, self.num_inference_steps, training=False
        )
        infer_action = self._make_flow_scheduler(
            self.action_flow_shift, self.num_inference_steps, training=False
        )

        scorer_ready = (
            self.use_drivor_scorer
            and self.drivor_scorer is not None
            and bool(getattr(self.drivor_scorer, "_loaded", False))
            and self.scorer_dino_register is not None
            and self.num_proposals > 1
            and not return_attn
        )
        if not scorer_ready:
            noisy_scenes = torch.randn(
                batch,
                self.num_future,
                self.len_obs_per_frame,
                self.dim,
                device=device,
                dtype=dtype,
            )
            noisy_action = torch.randn(
                batch, self.num_poses, 3, device=device, dtype=dtype
            )
            noisy_scenes, noisy_action, step_self = self._rf_denoise(
                dino_cond,
                ego_emb,
                noisy_scenes,
                noisy_action,
                infer_video,
                infer_action,
                return_attn=return_attn,
            )
            traj_abs = decode_trajectory_from_flow(noisy_action)
            out: Dict[str, torch.Tensor] = {
                "pred_future_dino": noisy_scenes,
                "trajectory": traj_abs,
            }
            if return_attn:
                # [S, L, B, N, N] e.g. [10, 5, 1, 72, 72]
                out["attn_maps"] = torch.stack(step_self, dim=0)
                out["infer_timesteps"] = infer_video.timesteps.detach().cpu()
                out["infer_timesteps_action"] = (
                    infer_action.timesteps.detach().cpu()
                )
            return out

        num_proposals = int(self.num_proposals)
        max_expand = max(int(self.proposal_chunk_size), 1)
        chunk_m = max(1, min(num_proposals, max_expand // max(batch, 1)))
        proposal_chunks: List[torch.Tensor] = []
        last_scenes: Optional[torch.Tensor] = None
        for start in range(0, num_proposals, chunk_m):
            num_modes = min(chunk_m, num_proposals - start)
            expand = batch * num_modes
            dino_m = dino_cond.repeat_interleave(num_modes, dim=0)
            # [B*m, len_h, 256] e.g. [32, 32, 256]
            ego_m = ego_emb.repeat_interleave(num_modes, dim=0)
            noisy_scenes = torch.randn(
                expand,
                self.num_future,
                self.len_obs_per_frame,
                self.dim,
                device=device,
                dtype=dtype,
            )
            noisy_action = torch.randn(
                expand, self.num_poses, 3, device=device, dtype=dtype
            )
            noisy_scenes, noisy_action, _ = self._rf_denoise(
                dino_m,
                ego_m,
                noisy_scenes,
                noisy_action,
                infer_video,
                infer_action,
            )
            traj_chunk = decode_trajectory_from_flow(noisy_action)
            # [B*m, T, 3] → [B, m, T, 3] e.g. [4, 8, 8, 3]
            proposal_chunks.append(
                traj_chunk.reshape(batch, num_modes, self.num_poses, 3)
            )
            last_scenes = noisy_scenes.reshape(
                batch,
                num_modes,
                self.num_future,
                self.len_obs_per_frame,
                self.dim,
            )[:, 0]

        proposals = torch.cat(proposal_chunks, dim=1)  # [B, K, T, 3] e.g. [4, 8, 8, 3]
        if self.proposal_mul_noise_copies > 0:
            # Keep the K RF originals, add C noisy copies each:
            # [B, K*(1+C), T, 3] e.g. [4, 64, 8, 3] for K=8, C=7, σ=0.1.
            proposals = expand_trajectories_mul_noise(
                proposals,
                num_copies=self.proposal_mul_noise_copies,
                noise_std=self.proposal_mul_noise_std,
            )
        scorer_image = features.get("scorer_image")
        if scorer_image is None:
            raise RuntimeError(
                "use_drivor_scorer requires features['scorer_image'] "
                "[B, 4, 3, H, W] (current-frame cam_f0/b0/l0/r0)."
            )
        # [B, 64, 256] e.g. [4, 64, 256] — DrivoR 4-cam current-frame tokens.
        scorer_scene = self._encode_scorer_obs(scorer_image)
        assert self.drivor_scorer is not None
        pdm_score, best_idx = self.drivor_scorer.score(
            proposals, scorer_scene, ego_status
        )  # [B, M], [B]
        gather = torch.arange(batch, device=device)
        traj_abs = proposals[gather, best_idx]  # [B, T, 3]
        out = {
            "pred_future_dino": last_scenes,
            "trajectory": traj_abs,
            "proposals": proposals,
            "pdm_score": pdm_score,
            "scorer_best_idx": best_idx,
        }
        return out

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Dispatch RF train/val (with targets) vs multi-step infer (no targets)."""
        if targets is not None:
            return self.forward_train(features, targets)
        return self.forward_infer(features)
