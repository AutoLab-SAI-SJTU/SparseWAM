# QueryEncoder architecture and train / inference pseudocode

> Code lives in `navsim/agents/drivoR/`.
>
> Main files:
> - `drivor_agent.py`: Lightning-style agent (training step, loss, optimizer, checkpoint)
> - `drivor_model.py`: `DrivoRModel`
> - `drivor_features.py`: feature and target builders
> - `transformer_decoder.py`: self-attention and cross-attention decoder
> - `layers/image_encoder/dinov2_lora.py`: DINOv2 + LoRA image and lidar backbone
> - `score_module/scorer.py`: six PDM sub-score heads
> - `score_module/compute_navsim_score.py`: online PDM simulation
> - `layers/losses/drivor_loss.py`: multi-task loss

---

## 1. Components

```
DrivoRAgent (AbstractAgent)
 ├─ DrivoRFeatureBuilder   → image / lidar / ego_status
 ├─ DrivoRTargetBuilder    → trajectory (+ long), token
 ├─ DrivoRModel
 │    ├─ ImgEncoder(DINOv2 + LoRA) x camera   → scene tokens
 │    ├─ ImgEncoder(DINOv2 + LoRA) x lidar    → scene tokens
 │    ├─ hist_encoding (Linear)               → ego_token
 │    ├─ init_feature  (trajectory prototypes) → traj_tokens
 │    ├─ traj_head[0..ref_num] (MLP)          → trajectory heads
 │    ├─ TransformerDecoder                   → stacked self/cross-attn refinement
 │    ├─ pos_embed (MLP)                      → proposal geometry
 │    ├─ TransformerDecoderScorer             → scorer cross-attn
 │    └─ Scorer (6 PDM heads + optional area/agent/bev)
 ├─ DrivoRLoss (trajectory / diversity / PDM / agent / bev)
 └─ compute_score: Ray workers run the PDM simulator for ground-truth sub-scores
```

---

## 2. `DrivoRModel.forward`

```python
class DrivoRModel(nn.Module):
    def __init__(cfg):
        D = cfg.tf_d_model

        if num_cams > 0:
            image_backbone = DINOv2_LoRA(image_size, num_features=F)
            scene_embeds   = Param[1, num_cams, num_scene_tokens, F]  # learnable

        if num_lidar > 0:
            lidar_backbone     = DINOv2_LoRA(lidar_image_size, ...)
            lidar_scene_embeds = Param[1, num_lidar, num_scene_tokens, F]

        hist_encoding = Linear(11 * 4 if full_history_status else 11, D)

        if one_token_per_traj:
            init_feature  = Embedding(proposal_num, D)             # one token per trajectory
            traj_out_size = num_poses * 3                          # one MLP emits the full trajectory
        else:
            init_feature  = Embedding(num_poses * proposal_num, D) # one token per pose
            traj_out_size = 3

        traj_head[i]       = MLP(D, ffn, traj_out_size) for i in 0..ref_num
        trajectory_decoder = TransformerDecoder(ref_num, D)        # returns every layer
        scorer_attention   = TransformerDecoderScorer(scorer_ref_num, D)
        pos_embed          = MLP(num_poses * 3 -> ffn -> D)
        scorer             = Scorer(cfg)  # 6 PDM heads plus optional heads

    def forward(features):
        # ego history -> ego_token
        ego_status = features["ego_status"].flatten(-2)            # [B, 11*4]
        ego_token  = hist_encoding(ego_status).unsqueeze(1)        # [B, 1, D]

        # trajectory prototypes
        traj_tokens = ego_token + init_feature.weight              # [B, N_or_NT, D]

        # cameras and lidar -> scene tokens
        scene_features = []
        if num_cams > 0:
            img = features["image"]                                # [B, num_cams, 3, H, W]
            st  = scene_embeds.expand(B, ...)                      # [B, num_cams, K, F]
            img_tokens = image_backbone(img, st)                   # [B, num_cams*K, D]
            scene_features.append(img_tokens)
        if num_lidar > 0:
            scene_features.append(lidar_backbone(features["lidar_feature"], ...))
        scene_features = cat(scene_features, dim=1)                # [B, S, D]

        # initial proposals (refinement layer 0)
        proposals     = traj_head[0](traj_tokens).reshape(B, N, num_poses, 3)
        proposal_list = [proposals]

        # refine for ref_num layers
        token_list = trajectory_decoder(traj_tokens, scene_features)  # [ref_num, B, ., D]
        for i in range(ref_num):
            tokens    = token_list[i]
            proposals = traj_head[i + 1](tokens).reshape(B, N, num_poses, 3)
            proposal_list.append(proposals)

        traj_tokens = token_list[-1]
        proposals   = proposal_list[-1]                           # [B, N, T, 3]

        # score from detached proposal geometry
        emb        = pos_embed(proposals.reshape(B, N, -1).detach())
        scorer_tok = scorer_attention(emb, scene_features) + ego_token

        # 6 PDM heads, plus optional area / agent / bev heads
        pred_logit, pred_logit2, pred_agents_states, pred_area_logit, \
        bev_semantic_map, agent_states, agent_labels = scorer(proposals, scorer_tok)

        # same mix used at inference to pick one trajectory
        pdm_score = ( noc * sigmoid(noc_logit).log()
                    + dac * sigmoid(dac_logit).log()
                    + ddc * sigmoid(ddc_logit).log()
                    + ( ttc * sigmoid(ttc_logit)
                      + ep  * sigmoid(ep_logit)
                      + comfort * sigmoid(comfort_logit) ).log() )

        best_idx   = argmax(pdm_score, dim=1)
        trajectory = proposals[range(B), best_idx]                # [B, T, 3]

        return {
            "proposals":     proposals,
            "proposal_list": proposal_list,
            "pred_logit":    pred_logit,    "pred_logit2": pred_logit2,
            "pred_agents_states": pred_agents_states,
            "pred_area_logit":    pred_area_logit,
            "bev_semantic_map":   bev_semantic_map,
            "agent_states":       agent_states,
            "agent_labels":       agent_labels,
            "pdm_score":          pdm_score,
            "trajectory":         trajectory,
        }
```

### 2.1 Transformer blocks (`transformer_decoder.py`)

```python
class Block:                      # Pre-LN: SelfAttn -> CrossAttn -> MLP
    def forward(x, kv):
        x = x + DropPath(LS(SelfAttn(LN(x))))
        x = x + DropPath(LS(CrossAttn(LN_q(x), LN_kv(kv))))
        x = x + DropPath(LS(MLP(LN(x))))
        return x

class TransformerDecoder:         # ref_num layers, returns [L, B, ., D]
class TransformerDecoderScorer:   # scorer_ref_num layers, returns the last layer
```

### 2.2 Backbone (`layers/image_encoder/dinov2_lora.py`)

```python
class ImgEncoder:
    def __init__(cfg):
        vit = timm.create_model("vit_*_dinov2", img_size=cfg.image_size)
        if cfg.use_lora:
            vit = LoRA_ViT_timm(vit, r=cfg.lora_rank)   # LoRA on qkv
        elif not cfg.finetune:
            freeze(vit)
        neck = Linear(F, D)
        grid_mask = GridMask(p=0.7)

    def forward(img[B,N,3,H,W], scene_tokens[B,N,K,F]):
        img = grid_mask(rearrange(img, "b n c h w -> (b n) c h w"))
        # Learnable scene tokens are prefix queries.
        tokens = vit.forward_features(img, scene_tokens)
        if focus_front_cam:                # keep all front tokens; keep K tokens for the other cameras
            front = tokens[:, 0]
            others = tokens[:, 1:, :K]
            tokens = cat([front, others])
        else:
            tokens = tokens[:, :K]
        return neck(tokens)               # [B, num_cams*K, D]
```

### 2.3 Scorer heads (`score_module/scorer.py`)

```python
class Scorer:
    pred_score = {                       # six independent MLPs
        "no_at_fault_collisions":         MLP(D -> ffn -> 1),
        "drivable_area_compliance":       MLP,
        "time_to_collision_within_bound": MLP,
        "ego_progress":                   MLP,
        "driving_direction_compliance":   MLP,
        "comfort":                        MLP,
    }
    if double_score: pred_score2 = MLP(D -> ffn -> 6)
    if agent_pred:   pred_col_agent = MLP(D -> ffn -> 2*T*9)      # train-only auxiliary
    if area_pred:    pred_area      = MLP(D -> ffn -> num_poses*2) # train-only auxiliary
    if bev_map:      map_head       = MapHead(cfg)                 # unused

    def forward(proposals, scorer_tok):
        pred_logit = {k: head(scorer_tok).squeeze(-1) for k, head in pred_score.items()}
        if self.training and area_pred:
            pred_area_logit = pred_area(scorer_tok)
        if self.training and agent_pred:
            pred_agents_states = pred_col_agent(scorer_tok)
        return pred_logit, ...
```

---

## 3. Features and targets (`drivor_features.py`)

```python
class DrivoRFeatureBuilder:
    def compute_features(agent_input):
        # Intrinsics, resize, and normalization -> [N_cam, 3, H, W].
        cameras = [cam_f0, cam_b0, cam_l0, cam_l1, cam_l2, cam_r0, cam_r1, cam_r2]
        image = stack([normalize(resize(cam.image)) for cam in cameras])

        # Up to 4 history frames: pose(3) + vel(2) + acc(2) + driving_command(4) = 11.
        ego_status = stack([
            cat([pose, vel, acc, cmd]) for ego in agent_input.ego_statuses
        ])
        return {"image": image, "ego_status": ego_status, ...}

class DrivoRTargetBuilder:
    def compute_targets(scene):
        trajectory = scene.get_future_trajectory(num_poses).poses  # [T, 3]
        # Optional longer horizon, resampled with a cubic spline.
        if long_trajectory_additional_poses > 0:
            trajectory_long = cubic_resample(scene.get_future_trajectory(T + dT))
        return {
            "trajectory": trajectory,
            "trajectory_long": trajectory_long,  # optional
            "token": scene.scene_metadata.initial_token,
        }
```

---

## 4. Training

### 4.1 Step

```python
for batch in dataloader:
    features, targets = batch
    pred = agent.forward(features)
    loss_dict = agent.compute_loss(features, targets, pred)
    loss_dict["loss"].backward()
    optimizer.step()
    scheduler.step()  # linear warmup, then cosine annealing
    optimizer.zero_grad()
```

### 4.2 Online PDM targets (`compute_score`)

```python
def compute_score(targets, proposals, test=False):
    # One Ray task per sample. Each task scores every proposal.
    data_points = [
        {"token": metric_cache_paths[token], "poses": poses, "test": test}
        for token, poses in zip(targets["token"], proposals.cpu().numpy())
    ]
    all_res = worker_map(ray_worker, get_scores, data_points)

    # get_scores runs PDMSimulator + PDMScorer:
    #   1. transform proposals into the global frame and simulate 40 steps
    #   2. compute the 6 sub-scores and the final score
    target_scores = stack([res[0] for res in all_res]).to(device)  # [B, N, 7]

    if test:
        return (
            final_scores[:, 0].mean(),
            best_scores.mean(),
            final_scores,
            l2_2s.mean(),
            target_scores[:, 0],
        )
    return (
        final_scores,
        best_scores,
        target_scores,
        key_agent_corners,
        key_agent_labels,
        all_ego_areas,
    )
```

### 4.3 Loss (`DrivoRLoss.forward`)

```python
def loss_fn(targets, pred, cfg, compute_score):
    final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = (
        compute_score(targets, pred["proposals"], test=False)
    )

    # Supervise every refinement layer: min-match L1 plus a diversity term.
    trajectory_loss = 0
    for proposals_i in pred["proposal_list"]:
        min_loss = l1(proposals_i, target_trajectory).mean(-1).amin(N).mean()
        if has(trajectory_long):
            min_loss += l1(proposals_i, trajectory_long).mean(-1).amin(N).mean()
        inter_loss = -pairwise_l1(proposals_i).amin().amin().mean()
        trajectory_loss = (
            prev_weight * trajectory_loss + min_loss + inter_weight * inter_loss
        )

    # BCE-with-logits on the 6 PDM sub-scores. TTC ignores the invalid label 2.
    sub_score_loss = (
        bce(dac_logit, gt_dac)
        + masked_bce(ttc_logit, gt_ttc, mask=gt_ttc != 2.0)
        + bce(noc_logit, three_to_two(gt_noc))
        + bce(ep_logit, gt_progress)
        + bce(ddc_logit, three_to_two(gt_ddc))
        + bce(comfort_logit, gt_comfort)
    )

    # Optional auxiliaries: Hungarian agent detection, BEV semantic CE.
    if pred["agent_states"] is not None:
        agent_class_loss, agent_box_loss = agent_loss(targets, pred)
    if pred["bev_semantic_map"] is not None:
        bev_semantic_loss = ce(pred["bev_semantic_map"], targets["bev_semantic_map"])

    loss = (
        trajectory_weight * trajectory_loss
        + sub_score_weight * sub_score_loss
        + pred_ce_weight * pred_ce_loss
        + pred_l1_weight * pred_l1_loss
        + pred_area_weight * pred_area_loss
        + agent_class_weight * agent_class_loss
        + agent_box_weight * agent_box_loss
        + bev_semantic_weight * bev_semantic_loss
    )
    chosen = argmax(pred["pdm_score"], dim=1)
    score = final_scores[arange(B), chosen].mean()
    return {"loss": loss, "score": score, "best_score": best_scores.mean(), ...}
```

### 4.4 Optimizer

```python
lr = base_lr * sqrt(global_bs / base_batch_size)
opt = Adam(model.parameters(), lr) or AdamW(model.parameters(), lr)

steps_per_epoch = ceil(dataset_size / global_bs)
T_max = steps_per_epoch * num_epochs
T_warmup = int(0.1 * T_max)
sched = SequentialLR(
    [
        LinearLR(start_factor=1e-6, total_iters=T_warmup),
        CosineAnnealingLR(T_max - T_warmup, eta_min=0),
    ],
    milestones=[T_warmup],
)
```

---

## 5. Inference

```python
def compute_trajectory(agent_input) -> Trajectory:
    agent.eval()
    features = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    features = {k: v.unsqueeze(0) for k, v in features.items()}  # [1, ...]

    with torch.no_grad():
        pred = agent._drivor_model(features)
        # pred["pdm_score"]:  [1, N]
        # pred["proposals"]:  [1, N, T, 3]
        # pred["trajectory"]: [1, T, 3], the argmax proposal
        poses = pred["trajectory"].squeeze(0).numpy()
    return Trajectory(poses)

def initialize():
    state = torch.load(checkpoint_path)["state_dict"]
    # Lightning stores weights under "agent._drivor_model".
    self.load_state_dict({
        k.replace("agent._drivor_model", "_drivor_model"): v
        for k, v in state.items()
    })
```

---

## 6. Design notes

- `init_feature` holds `proposal_num` trajectory prototypes. They are added to `ego_token` and refined for `ref_num` layers. Every layer has its own `traj_head` and is supervised with a min-match L1 loss plus a diversity term.
- Learnable `scene_embeds` are prefix queries to DINOv2 + LoRA, so the backbone emits a fixed set of scene tokens.
- `trajectory_decoder` generates proposals. `scorer_attention` scores them. Its query is `pos_embed(proposals).detach()`, so the score does not change the trajectory through that path.
- Training calls `PDMSimulator` and `PDMScorer` through Ray. The six NavSim sub-scores are BCE targets. `best_score` is the best-of-N simulator score.
- `pdm_score = noc * log(sigmoid) + dac * log(sigmoid) + ddc * log(sigmoid) + log(ttc * sigmoid + ep * sigmoid + comfort * sigmoid)`. Inference selects `argmax(pdm_score)`.
- The learning rate scales with `sqrt(global_batch_size / base_batch_size)`, then uses a 10% linear warmup and cosine decay.
