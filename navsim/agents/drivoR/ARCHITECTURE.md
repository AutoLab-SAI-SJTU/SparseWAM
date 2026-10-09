# DrivoR 模型架构与训练/推理伪代码

> 本文档对应代码路径：`DrivoR/navsim/agents/drivoR/`
>
> 主要文件：
> - `drivor_agent.py`：Lightning 风格的 Agent 封装（训练入口、loss、optimizer、checkpoint）
> - `drivor_model.py`：核心网络 `DrivoRModel`
> - `drivor_features.py`：特征 / 目标构建
> - `transformer_decoder.py`：Self+Cross-Attn 解码器
> - `layers/image_encoder/dinov2_lora.py`：DINOv2 + LoRA 图像/激光 backbone
> - `score_module/scorer.py`：6 个 PDM 子分头
> - `score_module/compute_navsim_score.py`：在线 PDM 仿真打分
> - `layers/losses/drivor_loss.py`：多任务联合损失

---

## 1. 顶层组件关系总览

```
DrivoRAgent (Lightning 风格的 AbstractAgent)
 ├─ FeatureBuilder (DrivoRFeatureBuilder)  → image / lidar / ego_status
 ├─ TargetBuilder  (DrivoRTargetBuilder)   → trajectory (+ long), token
 ├─ DrivoRModel  (核心网络)
 │    ├─ ImgEncoder(DINOv2 + LoRA) × camera   → 场景token
 │    ├─ ImgEncoder(DINOv2 + LoRA) × lidar    → 场景token
 │    ├─ hist_encoding (Linear)               → ego_token
 │    ├─ init_feature  (Embedding 轨迹原型)    → traj_tokens
 │    ├─ traj_head[0..ref_num] (MLP)          → 轨迹回归头
 │    ├─ TransformerDecoder  (refiner)        → 多层 self+cross-attn 迭代精化
 │    ├─ pos_embed (MLP)                      → 编码 proposal 几何
 │    ├─ TransformerDecoderScorer             → 评分专用 cross-attn
 │    └─ Scorer (6 个 PDM 子分头 + 可选 area/agent/bev)
 ├─ DrivoRLoss (轨迹/多样性/PDM子分/agent/bev)
 └─ compute_score : Ray 并行调用 PDM 模拟器，得到 GT 子分
```

---

## 2. 模型结构伪代码（`DrivoRModel.forward`）

```python
class DrivoRModel(nn.Module):
    # ---------- 初始化 ----------
    def __init__(cfg):
        D = cfg.tf_d_model

        if num_cams > 0:
            image_backbone = DINOv2_LoRA(image_size, num_features=F)
            scene_embeds   = Param[1, num_cams, num_scene_tokens, F]   # learnable

        if num_lidar > 0:
            lidar_backbone     = DINOv2_LoRA(lidar_image_size, ...)
            lidar_scene_embeds = Param[1, num_lidar, num_scene_tokens, F]

        hist_encoding = Linear(11*4 if full_history_status else 11, D)

        if one_token_per_traj:
            init_feature  = Embedding(proposal_num, D)            # 每条轨迹 1 个 token
            traj_out_size = num_poses * 3                          # MLP 一次出整条
        else:
            init_feature  = Embedding(num_poses * proposal_num, D) # 每个 pose 1 个 token
            traj_out_size = 3

        traj_head[i]        = MLP(D, ffn, traj_out_size) for i in 0..ref_num
        trajectory_decoder  = TransformerDecoder(ref_num, D)       # 返回每层中间结果
        scorer_attention    = TransformerDecoderScorer(scorer_ref_num, D)
        pos_embed           = MLP(num_poses*3 → ffn → D)
        scorer              = Scorer(cfg)   # 6 个 PDM 子分头 + 可选 head

    # ---------- 前向 ----------
    def forward(features):
        # (1) ego 历史 → ego_token
        ego_status = features["ego_status"].flatten(-2)            # [B, 11*4]
        ego_token  = hist_encoding(ego_status).unsqueeze(1)        # [B, 1, D]

        # (2) 轨迹原型 token = ego_token + Embedding
        traj_tokens = ego_token + init_feature.weight              # [B, N_or_NT, D]

        # (3) 场景特征：多相机 + 激光 → scene tokens
        scene_features = []
        if num_cams > 0:
            img = features["image"]                                # [B, num_cams, 3, H, W]
            st  = scene_embeds.expand(B, ...)                      # [B, num_cams, K, F]
            img_tokens = image_backbone(img, st)                   # [B, num_cams*K, D]
            scene_features.append(img_tokens)
        if num_lidar > 0:
            scene_features.append(lidar_backbone(features["lidar_feature"], ...))
        scene_features = cat(scene_features, dim=1)                # [B, S, D]

        # (4) 初始 proposals（refine 第 0 层）
        proposals      = traj_head[0](traj_tokens).reshape(B, N, num_poses, 3)
        proposal_list  = [proposals]

        # (5) Transformer decoder 多层 refine
        token_list = trajectory_decoder(traj_tokens, scene_features)  # [ref_num, B, ., D]
        for i in range(ref_num):
            tokens    = token_list[i]
            proposals = traj_head[i+1](tokens).reshape(B, N, num_poses, 3)
            proposal_list.append(proposals)

        traj_tokens = token_list[-1]
        proposals   = proposal_list[-1]                           # [B, N, T, 3]

        # (6) Scorer：用 proposal 的几何编码再 cross-attn 一次
        emb        = pos_embed(proposals.reshape(B, N, -1).detach())
        scorer_tok = scorer_attention(emb, scene_features) + ego_token

        # (7) 6 个 PDM 子分头 (+ 可选 area / agent / bev)
        pred_logit, pred_logit2, pred_agents_states, pred_area_logit, \
        bev_semantic_map, agent_states, agent_labels = scorer(proposals, scorer_tok)

        # (8) 用 PDM 公式聚合成单一 score，挑最优轨迹
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

### 2.1 Transformer 子模块（`transformer_decoder.py`）

```python
class Block:                      # 标准 Pre-LN: SelfAttn → CrossAttn → MLP
    def forward(x, kv):
        x = x + DropPath(LS(SelfAttn(LN(x))))
        x = x + DropPath(LS(CrossAttn(LN_q(x), LN_kv(kv))))
        x = x + DropPath(LS(MLP(LN(x))))
        return x

class TransformerDecoder:         # ref_num 层，返回 [L, B, ., D] 中间层堆叠
class TransformerDecoderScorer:   # scorer_ref_num 层，只返回最后一层
```

### 2.2 Backbone（`layers/image_encoder/dinov2_lora.py`）

```python
class ImgEncoder:
    def __init__(cfg):
        vit       = timm.create_model("vit_*_dinov2", img_size=cfg.image_size)
        if cfg.use_lora:  vit = LoRA_ViT_timm(vit, r=cfg.lora_rank)   # 注入 LoRA 到 qkv
        elif not cfg.finetune: freeze(vit)                            # 否则全冻结
        neck      = Linear(F, D)
        grid_mask = GridMask(p=0.7)

    def forward(img[B,N,3,H,W], scene_tokens[B,N,K,F]):
        img = grid_mask(rearrange(img, 'b n c h w -> (b n) c h w'))
        # 把 scene_tokens 当作可学习的 query 一同前向，作为前缀 token
        tokens = vit.forward_features(img, scene_tokens)
        if focus_front_cam:                # 前向相机保留全部 token，其他相机仅前 K 个
            front  = tokens[:, 0]
            others = tokens[:, 1:, :K]
            tokens = cat([front, others])
        else:
            tokens = tokens[:, :K]
        return neck(tokens)               # [B, num_cams*K, D]
```

### 2.3 Scorer 头（`score_module/scorer.py`）

```python
class Scorer:
    pred_score = {                       # 6 个独立 MLP
        'no_at_fault_collisions':         MLP(D→ffn→1),
        'drivable_area_compliance':       MLP,
        'time_to_collision_within_bound': MLP,
        'ego_progress':                   MLP,
        'driving_direction_compliance':   MLP,
        'comfort':                        MLP,
    }
    if double_score: pred_score2 = MLP(D→ffn→6)
    if agent_pred:   pred_col_agent = MLP(D→ffn→2*T*9)   # 训练时辅助
    if area_pred:    pred_area      = MLP(D→ffn→num_poses*2)  # 训练时辅助
    if bev_map:      map_head       = MapHead(cfg)            # (未启用)

    def forward(proposals, scorer_tok):
        pred_logit = { k: head(scorer_tok).squeeze(-1) for k,head in pred_score.items() }
        if self.training and area_pred:  pred_area_logit   = pred_area(scorer_tok)
        if self.training and agent_pred: pred_agents_states= pred_col_agent(scorer_tok)
        return pred_logit, ...
```

---

## 3. 特征 / 目标构建（`drivor_features.py`）

```python
class DrivoRFeatureBuilder:
    def compute_features(agent_input):
        # 多相机：内参缩放 / resize / 归一化 → [N_cam, 3, H, W]
        cameras = [cam_f0, cam_b0, cam_l0, cam_l1, cam_l2, cam_r0, cam_r1, cam_r2]
        image   = stack([normalize(resize(cam.image)) for cam in cameras])

        # ego 历史 (最多 4 帧): pose(3) + vel(2) + acc(2) + driving_command(4) = 11
        ego_status = stack([cat([pose, vel, acc, cmd]) for ego in agent_input.ego_statuses])

        return {"image": image, "ego_status": ego_status, ...}

class DrivoRTargetBuilder:
    def compute_targets(scene):
        trajectory = scene.get_future_trajectory(num_poses).poses        # [T, 3]
        # 可选：用长时刻轨迹 + CubicSpline 重采样，强化远端监督
        if long_trajectory_additional_poses > 0:
            trajectory_long = cubic_resample(scene.get_future_trajectory(T+ΔT))
        return {"trajectory": trajectory,
                "trajectory_long": trajectory_long,        # 可选
                "token": scene.scene_metadata.initial_token}
```

---

## 4. 训练流程

### 4.1 训练步主循环（PyTorch Lightning 由 Trainer 驱动）

```python
for batch in dataloader:
    features, targets = batch                  # 由 Feature/Target Builder 构造
    pred = agent.forward(features)             # 见 DrivoRModel.forward
    loss_dict = agent.compute_loss(features, targets, pred)
    loss_dict["loss"].backward()
    optimizer.step(); scheduler.step()         # LinearLR warmup + CosineAnnealing
    optimizer.zero_grad()
```

### 4.2 在线 PDM 子分计算（`compute_score`）

```python
def compute_score(targets, proposals, test=False):
    # 用 Ray 并行：对 batch 里每条 proposal 调用 PDM 模拟器
    data_points = [{"token": metric_cache_paths[token], "poses": poses, "test": test}
                   for token, poses in zip(targets["token"], proposals.cpu().numpy())]
    all_res = worker_map(ray_worker, get_scores, data_points)

    # get_scores -> 每个样本运行 PDMSimulator + PDMScorer
    #   1) 把模型 proposals 变换到全局坐标 -> 40 步 41 帧仿真
    #   2) 用 PDMScorer 算 6 个子分 + final score
    target_scores = stack([res[0] for res in all_res]).to(device)   # [B, N, 7]

    if test:                                # 验证：返回总分 / best 分 / l2 等指标
        return final_scores[:,0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:,0]
    else:                                   # 训练：再返回 key_agent / ego_area 监督信号
        return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas
```

### 4.3 总损失（`DrivoRLoss.forward`）

```python
def loss_fn(targets, pred, cfg, compute_score):
    # (A) 在线获得 GT 分数和 area / agent 监督
    final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas \
        = compute_score(targets, pred["proposals"], test=False)

    # (B) 轨迹回归 + 多样性损失（对 refine 每一层都监督）
    trajectory_loss = 0
    for proposals_i in pred["proposal_list"]:
        min_loss   = || proposals_i - target_trajectory ||_1.mean(-1).amin(N).mean()
        if has(trajectory_long):
            min_loss += ||proposals_i - trajectory_long||_1.mean(-1).amin(N).mean()
        # 多样性：在 N 条 proposals 之间互相排斥
        inter_loss = -pairwise_L1(proposals_i).amin.amin.mean()
        trajectory_loss = prev_weight*trajectory_loss + min_loss + inter_weight*inter_loss

    # (C) 6 个 PDM 子分的 BCEWithLogits (TTC 用 mask 屏蔽无效)
    da_loss     = BCE(dac_logit,   gt_dac)
    ttc_loss    = MaskedBCE(ttc_logit, gt_ttc, mask=gt_ttc != 2.0)
    noc_loss    = BCE(noc_logit,   three_to_two(gt_noc))
    progress    = BCE(ep_logit,    gt_progress)
    ddc_loss    = BCE(ddc_logit,   three_to_two(gt_ddc))
    comfort     = BCE(comfort_logit, gt_comfort)
    final_score_loss = sum(子分_loss * w)

    # (D) 可选辅助：agent 检测 (Hungarian + L1 + CE)、BEV 语义 CE
    if pred["agent_states"] is not None:
        agent_class_loss, agent_box_loss = _agent_loss(targets, pred, ...)
    if pred["bev_semantic_map"] is not None:
        bev_semantic_loss = CE(pred["bev_semantic_map"], targets["bev_semantic_map"])

    loss = trajectory_weight   * trajectory_loss
         + final_score_weight  * final_score_loss
         + pred_ce_weight      * pred_ce_loss
         + pred_l1_weight      * pred_l1_loss
         + pred_area_weight    * pred_area_loss
         + agent_class_weight  * agent_class_loss
         + agent_box_weight    * agent_box_loss
         + bev_semantic_weight * bev_semantic_loss

    # 监控：用 argmax(pdm_score) 选出的轨迹对应的 final_score
    score      = final_scores[arange(B), argmax(pred["pdm_score"], dim=1)].mean()
    best_score = best_scores.mean()
    return {"loss": loss, "score": score, "best_score": best_score, ...}
```

### 4.4 优化器 / 调度器（`get_optimizers`）

```python
lr = base_lr * sqrt(global_bs / base_batch_size)             # 线性缩放（按 sqrt）
opt = Adam(model.params, lr) | AdamW(model.params, lr)

T_max      = ceil(dataset_size / global_bs) * num_epochs
T_warmup   = int(0.1 * T_max)
sched = SequentialLR(
    [LinearLR(start_factor=1e-6, total_iters=T_warmup),       # warmup 10%
     CosineAnnealingLR(T_max - T_warmup, eta_min=0)],
    milestones=[T_warmup])

callbacks = [
    ModelCheckpoint(monitor='val/score_epoch', mode='max', save_top_k=1),
    ModelCheckpoint(save_last=True),
    LearningRateMonitor(interval='step'),
]
```

---

## 5. 推理流程

```python
# 入口由父类 AbstractAgent.compute_trajectory 提供
def compute_trajectory(agent_input) -> Trajectory:
    agent.eval()
    features = {}
    for builder in agent.get_feature_builders():              # DrivoRFeatureBuilder
        features.update(builder.compute_features(agent_input))
    features = {k: v.unsqueeze(0) for k,v in features.items()}# 加 batch 维

    with torch.no_grad():
        pred = agent._drivor_model(features)                  # 见第 2 节 forward
        # pred["pdm_score"]   : [1, N]    模型自评 PDM 分
        # pred["proposals"]   : [1, N, T, 3]
        # pred["trajectory"]  : [1, T, 3]  ← argmax(pdm_score) 选出的最佳轨迹
        poses = pred["trajectory"].squeeze(0).numpy()

    return Trajectory(poses)

# 加载 checkpoint：
def initialize():
    state = torch.load(checkpoint_path)["state_dict"]
    # 兼容 Lightning 训练时被加前缀 "agent._drivor_model" 的 key
    self.load_state_dict({k.replace("agent._drivor_model", "_drivor_model"): v
                          for k,v in state.items()})
```

---

## 6. 关键设计点速记

- **多轨迹原型 + 迭代精化**：`init_feature` 提供 `proposal_num` 条初始轨迹原型，叠加 `ego_token` 后送入 `TransformerDecoder` 的 `ref_num` 层，每层都有自己的 `traj_head` 回归一次，所有中间层都参与监督（最小匹配损失 + 多样性）。
- **场景 token 由 backbone 自身吐出**：用 `scene_embeds`（可学习）作为 ViT 的前缀 query，让 DINOv2 + LoRA 直接给出固定数量的 BEV/场景 token，避免重新设计 BEV 模块。
- **双解码器**：`trajectory_decoder` 负责"生成"，`scorer_attention` 负责"打分"，后者用 `pos_embed(proposals)` 作为 query（且 detach）防止评分梯度回传影响轨迹本身。
- **PDM 在线监督**：训练时通过 Ray 并行调用 `PDMSimulator + PDMScorer`，把 NavSim 的 PDM 六大子分作为 BCE 监督，同时给出 best-of-N 上界 (`best_score`) 作为评估参考。
- **PDM 公式聚合**：`pdm_score = noc·log σ + dac·log σ + ddc·log σ + log(ttc·σ + ep·σ + comfort·σ)`；推理时 `argmax(pdm_score)` 直接给出选用的轨迹。
- **优化器规模感知**：学习率随 `sqrt(global_batch_size / base_batch_size)` 缩放，并搭配 10% 线性 warmup + 余弦退火。
