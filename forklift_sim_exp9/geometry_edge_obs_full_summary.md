# 21D 几何边缘观测实验 · 全面总结（2026-04-27）

> 分支：`exp/geo-obs-impl`
> Commit：`5de517f9`（本总结撰写时 HEAD）
> 对照基线：`exp/geo-obs-base @ 61081469`（15D 特权观测，EMA = 0.9785）
> 设计文档：[`geometry_edge_obs_plan.md`](./geometry_edge_obs_plan.md)
> 迭代日志：[`geometry_edge_obs_result.md`](./geometry_edge_obs_result.md)

---

## 项目背景与任务目标

### 最终目标：叉车栈板搬运的 sim→real

本项目的终极目标是让一台**真实叉车**在真实仓储环境里自主完成"找到栈板 → 对准叉齿 → 插入 → 举升搬运"全流程作业。典型场景：

- 叉车从 3–5 m 外的任意位姿出发
- 栈板位置/朝向不完全已知（需由机载传感器估计）
- 要求亚厘米级的插入对齐精度（pocket 开口宽 ~20 cm，叉齿宽 ~12 cm，容错 ±4 cm）
- 插入后稳定举升 ≥ 30 cm 完成一次搬运动作

传统方案（SLAM + 全局规划 + 视觉标定 + 纯控制 pipeline）在亚厘米对齐上鲁棒性差、每个环节误差叠加。我们走的是 **Isaac Lab + RL + sim2real** 路线：在仿真里大规模并行训练策略，再迁移到真机。

### 仿真任务定义（本项目的 RL 环境）

在仿真里把上述流程抽象成一个强化学习任务（task id：`Isaac-Forklift-PalletInsertLift-Direct-v0` 及其几何观测变体）：

1. **初始**：叉车在栈板前方 3–4 m 的矩形区域内随机位姿（x ∈ [-4,-3], y ∈ [-0.6,+0.6], yaw ∈ [-0.25,+0.25] rad）；栈板固定于原点
2. **动作**：`[drive, steer, lift]` 三维连续，每个维度 [-1, +1]
3. **阶段目标**：
   - Approach：开到距 pocket 开口约 0.5 m
   - Insert：调整横向 + 偏航，将叉齿插入 pocket 至少 40% 深度
   - Lift：举升至少 30 cm 并保持 10 帧不跌出阈值
4. **成功判据**：5 维 Schmitt 滞回同时满足（横向误差 / 偏航误差 / 插入深度 / 举升高度 / 叉齿精对齐）持续 10 帧

### 路线图与本实验的位置

整个 sim→real 路线分三阶段：

```
┌────────────────────────────────────────────────────────────┐
│ Phase 0 · 15D 特权 RL 基线  ──────────────  已完成         │
│   直接读仿真 GT 位姿 (d_xy, dyaw, ...) 共 15 维            │
│   验证：奖励函数 + PPO 超参 + 任务可学性                   │
│   结果：2000 iter 收敛到 EMA 0.9785                        │
│                                                            │
│          ▼  问题：特权观测在真机不可得                     │
│                                                            │
│ Phase 1A · 21D 几何观测 RL (GT 投影)  ──── 本次实验        │
│   用"2 条 pocket 短边的 2D 像素端点 + proprio"             │
│   通过虚拟相机投影 GT 几何，模拟完美 CV 输出               │
│   验证：几何表征是否足以让 PPO 学会任务                    │
│   结果：v2 峰值 EMA 0.9040，验证通过                       │
│                                                            │
│          ▼  问题：还缺真实视觉噪声                         │
│                                                            │
│ Phase 1B · 真 CV pipeline + 域随机化  ──── 下一步          │
│   Canny + HoughLinesP 从渲染 RGB 里真实检出栈板边          │
│   与 Phase 1A 完全相同的 21D 观测接口 → 策略可直接加载     │
│   加光照/纹理/相机抖动/噪声域随机化                        │
│   量化 sim2real gap                                        │
│                                                            │
│          ▼                                                 │
│                                                            │
│ Phase 2 · 真机部署                                         │
│   真机相机标定 → C++/ROS2 的 CV node                       │
│   Zero-shot 或少量 fine-tune 部署                          │
└────────────────────────────────────────────────────────────┘
```

### 本实验要回答的 4 个问题

| # | 问题 | 验证方式 |
|---|---|---|
| Q1 | 哪种视觉表征既 sim2real 友好、又信息充分？ | 证明 "2 条 pocket 短边的 2D 像素端点 + 可见性/近远标签" 12 维 + 9 维 proprio 够用 |
| Q2 | MLP 策略能学会吗？还是必须上 CNN？ | 21 维纯向量 MLP 即可，**不需要 CNN、不需要渲染 RGB** |
| Q3 | 相对 15D 特权基线的性能损失有多大？ | 量化峰值 / 终盘 EMA 差距，作为 Phase 1B 的上界参考 |
| Q4 | 虚拟相机参数如何选才合理？ | 通过 v1 失败 / v2 成功，给出 "HFOV + 俯视角 + 相机高度" 的可行工作点 |

### 为什么选"几何边缘 + MLP"而不是"RGB 像素 + CNN"

1. **RGB 渲染训练的老坑**：先前的 `exp/exp9_0` 尝试过把 RGB 相机接到 RL 训练里，多次出现渲染崩溃、显存暴涨、训练阻塞，放弃
2. **Sim2Real gap 在像素域远大于几何域**：RGB 纹理/光照/相机噪声的域随机化成本极高；而 CV 检出的 "2D 线段端点" 在真机和仿真里都能稳定获得，**中层特征天然一致**
3. **传统 CV 对栈板直线边缘极度鲁棒**：Canny + `HoughLinesP` 几十年工业应用证明过
4. **训练吞吐保持**：纯向量 obs → 单次 rollout ~24k steps/s（RTX 5090, 1024 env），是 CNN+RGB 路线的 3–5 倍
5. **可解释性强**：policy 输入每一维都有物理含义（图像中的边端点像素坐标 + 可见性 flag），调试/失效分析容易

### 本文档覆盖的内容

本文档记录 **Phase 1A** 的完整执行过程：

- 观测设计（15D → 21D 的替换方案、投影管线、坐标系约定）
- 虚拟相机参数（内参、外参、FoV 容差）与 v1→v2 迭代中的关键教训
- 奖励函数（与基线 100% 一致，单变量控制）
- PPO 训练配置（同样与基线一致）
- 训练结果（v1 失败、v2 成功、学习曲线、与 15D 基线的逐指标对比）
- 结论与向 Phase 1B 的过渡计划

---

## 0. TL;DR

1. **用 21 维几何观测（2 条 pocket 短边 + 9 维 proprio）完全替代 15 维特权状态向量**，PPO 成功学会 approach + 对齐 + 插入 + 举升全流程。
2. **第一版（v1，HFOV 90°）失败** —— 根因是虚拟相机 FoV 在近场 0.58m 时覆盖不到近端短边，策略在关键的"插入瞬间"丢观测；2000 iter 计划中 iter 1139 提前人工终止。
3. **第二版（v2，HFOV 120°, cam_z 1.30, pitch 25°）成功** —— 完整跑完 2000 iter，**峰值 EMA 0.9040 @ iter 1531**，**终盘 EMA 0.8063**。
4. 相比 15D 特权基线（峰值 0.9792），峰值差距仅 7.5 pp，**几何观测的可学习性得到验证**，可作为 Phase 1B 真 CV pipeline 的上界参考。

---

## 1. 任务与环境

### 1.1 任务定义

叉车从 3-4 米外随机起点，驶向固定位姿的欧标托盘（1.2m×0.8m×0.145m，按 1.8× 缩放到 2.16m×1.44m×0.261m），将叉齿插入托盘底部 pocket，然后举升至少 0.30m 并保持 10 帧（0.33 s）才算成功。

- **动作空间**：`[drive, steer, lift]` ∈ [-1, +1]³，分别对应车轮速度（±20 rad/s）、转向角（±0.6 rad）、举升速度（±0.5 m/s）。
- **Episode 长度**：36 s × 30 Hz ÷ decimation 4 = 1080 steps。
- **初始位姿随机化**：
  - 叉车 x ∈ [-4.0, -3.0] m、y ∈ [-0.6, +0.6] m、yaw ∈ [-0.25, +0.25] rad
  - 托盘位姿固定（与 15D 基线一致）
- **并行环境**：`num_envs = 1024`（训练）/ 128（smoke）
- **Seed**：42

### 1.2 成功判定（Schmitt 三维 + hold counter）

同时满足以下条件持续 10 帧：

| 维度 | Entry 阈值 | Exit 阈值（1.2× 滞回） |
|---|---|---|
| 横向误差 `y_err` | ≤ 0.15 m | > 0.18 m |
| 偏航误差 `yaw_err` | ≤ 5.0° | > 6.0° |
| 插入深度 `insert_depth` | ≥ `0.40 × pallet_depth_m` | < 阈值 - 0.02 m |
| 举升高度 `lift_height` | ≥ 0.30 m | < 阈值 - 0.08 m |
| 叉齿横向误差 `tip_y_err`（近场 < 2.2m 时） | ≤ 0.12 m | > 0.16 m |

越界时 `hold_counter *= 0.8`（衰减而非清零），累计到 10 则判为成功。

---

## 2. 观测空间对比

### 2.1 15D 特权观测（基线）

```
[d_xy_r (2), cos_dyaw (1), sin_dyaw (1), v_xy_r (2),
 yaw_rate (1), lift_pos (1), lift_vel (1), insert_norm (1),
 actions (3), y_err_obs (1), yaw_err_obs (1)]   # = 15
```

关键点：`d_xy_r / cos_dyaw / sin_dyaw / y_err_obs / yaw_err_obs` 都是**直接读取仿真 GT 的相对位姿**，是"上帝视角"特权信息。

### 2.2 21D 几何边观测（本实验）

```
obs = [edge_obs (12), proprio (9)]   # = 21
```

#### 2.2.1 边部分（12 维）— `env._get_edge_obs()`

2 条托盘**入口短边（pocket 开口两侧）** × 6 特征/边：

| 字段 | 含义 | 取值 |
|---|---|---|
| `u1, v1` | 端点 1 的归一化像素坐标 | [-1, +1]，图像中心为 0 |
| `u2, v2` | 端点 2 的归一化像素坐标 | [-1, +1] |
| `visible` | 两端点是否都在 FoV 内（含 10% 容差）且 `z_cam > 0.05` | {0, 1} |
| `is_near` | 两条短边里，中点到相机更近的一条为 1 | one-hot {0, 1} |

**边身份排序**：按 **pallet local frame 固定**——
- `edge[0]` = pallet local `x = -depth/2` 一侧短边（−X 端）
- `edge[1]` = pallet local `x = +depth/2` 一侧短边（+X 端）
- `is_near` 标签随相机当前距离自动判定。

**端点 3D 局部坐标**（取托盘顶面，相机视角清晰）：
```
edge[0]: (-1.08, -0.72, +0.131), (-1.08, +0.72, +0.131)
edge[1]: (+1.08, -0.72, +0.131), (+1.08, +0.72, +0.131)
```

**不可见端点处理**：`u = v = 0`，`visible = 0`（策略网络第 1 层 MLP 会学到"特定 6 维模式" = "看不见"）。

#### 2.2.2 Proprio 部分（9 维）

```
[v_xy_r (2), yaw_rate (1), lift_pos / scale (1),
 lift_vel (1), insert_norm (1), actions[0:3] (3)]   # = 9
```

只保留"与外部几何无关、但策略仍需的"本体特征。其中 `insert_norm` 是**唯一无法从几何边反推**的特权位（需知道叉齿 tip 相对 pocket 的深度），保留它等价于"叉齿有一个深度接触传感器"，物理合理。

### 2.3 投影管线（env 内部，替代"真相机 + CV"）

Phase 1A **不开真相机渲染**，而是用 GT 几何 + pinhole 投影**模拟"完美 CV pipeline 的输出"**：

```
P_w  (world)                                      # GT 托盘边端点
  ↓ world → robot body:  P_r = R_robot^T (P_w - robot_pos)
  ↓ robot body → camera: P_c = R_body→cam (P_r - cam_pos_local)
  ↓ pinhole projection:  u = fx·x/z + cx,  v = fy·y/z + cy
  ↓ normalize:           u_norm = (u-W/2)/(W/2),  v_norm = (v-H/2)/(H/2)
  ↓ FoV 裁切:            visible = (z>0.05) ∧ (|u_norm|≤1.1) ∧ (|v_norm|≤1.1)
```

这样**策略看到的 obs 数据形式和真 CV pipeline（OpenCV `HoughLinesP`）输出一致**，Phase 1B 只需把"GT 投影"换成"真 CV"即可完成 sim2real 对接。

---

## 3. 虚拟相机参数

### 3.1 内参（pinhole）

| 参数 | v1（失败） | **v2（采用）** | 说明 |
|---|---|---|---|
| 图像宽×高 | 256×256 | 256×256 | 对齐 exp9 |
| HFOV | **90°** | **120°** | 宽视角，近距离能框住整盘 |
| focal fx=fy | 128 | **73.9** | `W/2 / tan(HFOV/2)` |
| principal point (cx, cy) | (128, 128) | (128, 128) | 图像中心 |

### 3.2 外参（相对叉车 root body 的固定安装）

| 参数 | v1 | **v2** | 说明 |
|---|---|---|---|
| `camera_pos_local` | (0.30, 0, **1.80**) m | (0.30, 0, **1.30**) m | 向前 0.3m，离地高度 |
| `camera_pitch_deg` | **+15** | **+25** | ROS 约定，正值 = nose down |
| roll / yaw | 0 / 0 | 0 / 0 | 无 roll / yaw |

**坐标系约定**：
- 叉车 body frame：Isaac 约定，X forward, Y left, Z up
- 相机 frame：OpenCV 约定，X right, Y down, Z forward
- 两者之间用固定的 axis-swap 矩阵 + 动态 pitch 旋转对接（见 `env._init_geometry_edge_obs`）

### 3.3 可见性容差

`camera_fov_margin = 1.1`：`|u_norm|, |v_norm| ≤ 1.1` 才判为可见（允许 10% 图像边缘越界宽容）。

### 3.4 v1 失败的几何根因

叉车接近到叉齿前 0.58 m 时（成功插入前的关键阶段），**近端短边左/右端点的半视角 = atan(0.72 / 0.58) ≈ 51.2°**：

- v1 HFOV 90° → 半视角 45°，51.2° > 45° → `visible = 0`，坐标被置零 → 策略失去"pocket 开口离我多近 / 横向对齐如何"的全部信号。
- v2 HFOV 120° → 半视角 60°，51.2° < 60° → `u_norm ≈ ±0.60`，仍在 FoV 内。

v1 的 `success_rate_ema` 全程 0.0000，v2 iter 1531 峰值 0.9040，差距的根因**只在这 3 个相机参数**。

---

## 4. 奖励函数（与 15D 基线完全一致，**单变量控制**）

几何观测实验**没有改任何奖励系数**，目标是"观测表征的公平对比"。下面按 reward 组件列出完整构成（源自 `env._get_rewards`）。

### 4.1 三阶段势函数 shaping（主体）

```
phi_total = phi1 + phi2 + phi_ins + phi_lift
r_pot = phi_total_t - phi_total_{t-1}       # 差分 shaping，gamma=1.0
```

| 阶段 | 势函数 | 激活条件 | 物理含义 |
|---|---|---|---|
| Stage 1 `phi1` | `k_phi1 / (1 + E1)`，`k_phi1=6.0` | 距离带 [2.0, 3.0] m + 粗对齐 | 让叉车进入环形距离带 |
| Stage 2 `phi2` | `(k_phi2 / (1 + E2)) × w_band × w_align2`，`k_phi2=10.0` | 穿出 d1_min 后 | 微调对准 pocket 前 |
| Stage 3 `phi_ins` | `k_ins × (0.2 + 0.8·w_align3) × insert_norm × w3 × w_lat_gate`，`k_ins=18.0` | `insert_norm > 0.10` | 插入深化 |
| 举升 `phi_lift` | `k_lift × w_lift × lift_height`，`k_lift=20.0` | `insert_norm > 0.35` | 举升 |

其中 `E1 = e_band/0.5 + y_err/0.15 + yaw_err_deg/10`，`E2 = stage_dist/1.0 + y_err/0.08 + yaw_err_deg/5`。

### 4.2 里程碑奖励（一次性触发，防重放）

| 里程碑 | 条件 | 奖励 |
|---|---|---|
| `approach` | `dist_front ≤ 0.25 m` | +1.0 |
| `coarse_align` | `y ≤ 0.20 m & yaw ≤ 10°` | +2.0 |
| `gate_align` | `y ≤ 0.15 m & yaw ≤ 8°`（+ approach flag） | +2.5 |
| `insert_10` | `insert_norm ≥ 0.10` | +5.0 |
| `insert_30` | `insert_norm ≥ 0.30` | +10.0 |
| `fine_align` | `y ≤ 0.10 m & yaw ≤ 5°` | +5.0 |
| `precise_align` | `y ≤ 0.05 m & yaw ≤ 3°` | +8.0 |
| `lift_10cm` / `lift_20cm` / `lift_50cm` / `lift_75cm` | 举升阈值穿越（带 insert 门控） | +3 / +5 / +6 / +8 |

### 4.3 Delta shaping（连续梯度）

| 组件 | 公式 | 权重 | 用途 |
|---|---|---|---|
| `r_hold_align` | `k·Δphi_align`，`phi_align = exp(-(y/0.25)² - (yaw/8°)²)` | `k_hold_align=0.3` | 对齐区间梯度 |
| `r_lift_progress` | `k·Δ(w_lift_base × exp(-(Δh/0.15)²))` | `k_lift_progress=1.2` | 举升进度 |
| `r_lat_fine` | `k·Δphi_lat`（`insert_norm > 0.05` 激活） | `k_lat_fine=0.0`（当前未激活） | 横向精调 |
| `r_far_lat_fix` | `k·Δy_err`（`y > 0.4 & insert < 0.1`） | `k_far_lat=0.0` | 远场大横偏修正 |

### 4.4 惩罚项

| 组件 | 条件/权重 | 目的 |
|---|---|---|
| `pen_dense` | `rew_time_penalty + rew_action_l2·‖a‖² - k_dist_cont·d_xy` | 常驻，防磨蹭 |
| `pen_premature` | `-k_pre·(1-premature_fade)·max(lift_height, 0)`，`k_pre=5.0` | 禁止未插先举 |
| `pen_dead_zone` | `-k·(insert_norm-0.30)·(y-0.20)`（clamp），`k=0.5` | 深插 + 大横偏的死区 |
| `pen_tip_proximity` | `-k·(tip_y_err - 0.15)+`（近场激活），`k=1.5` | 防叉齿撞盘 |
| `pen_pallet_push` | `-k·push_gate·(pallet_disp - 0.05)+`（Exp-A2），`k=1.0` | 防未插稳推走托盘 |
| `pen_global_stall` | `-1.5`（120 步 `phi_total` 无变化） | 防策略卡死 |
| `early_stop_penalty` | `rew_early_stop_fly=-2.0 / stall=-1.0 / dz_stuck=-2.0` | 失败早停 |

### 4.5 终局奖励

| 事件 | 奖励 |
|---|---|
| 成功（hold_counter ≥ 10） | `+100.0 + rew_success_time·(1-time_ratio)`（最高 +130） |
| 超时 | `-10.0` |
| 成功后静止 | `+0.5 / step` |

### 4.6 总奖励

```
rew = r_pot + pen_premature + pen_dense + r_terminal
    + milestone_reward + r_hold_align + r_lift_progress
    + early_stop_penalty + pen_dead_zone + r_retreat
    + r_lat_fine + r_far_lat_fix + pen_global_stall
    + pen_tip_proximity + pen_pallet_push + r_stay_still
```

---

## 5. PPO 训练配置（与基线完全一致）

### 5.1 Runner

| 参数 | 值 | 说明 |
|---|---|---|
| `num_envs` | 1024 | 并行环境数（smoke 128） |
| `num_steps_per_env` | 64 | 每次 rollout 步数 |
| `max_iterations` | 2000 | 总迭代数 |
| `save_interval` | 50 | 每 50 iter 存 checkpoint |
| `seed` | 42 | 随机种子 |
| `experiment_name` | `forklift_pallet_insert_lift_geo_edge` | 独立日志目录 |

### 5.2 Policy（MLP Actor-Critic，无 CNN）

| 参数 | 值 | 说明 |
|---|---|---|
| `class_name` | `ClampedActorCritic` | clamp `log_std ≥ ln(0.05)` 防 std 塌缩 |
| `actor_hidden_dims` | [256, 256, 128] | Actor MLP |
| `critic_hidden_dims` | [256, 256, 128] | Critic MLP |
| `activation` | elu | |
| `init_noise_std` | 0.5 | 初始探索强度 |
| `noise_std_type` | log | log 空间梯度更新 |
| `actor_obs_normalization` | True | 观测归一化（running mean/std） |
| `critic_obs_normalization` | True | |

### 5.3 PPO Algorithm

| 参数 | 值 | 说明 |
|---|---|---|
| `num_learning_epochs` | 5 | 每 iter 更新轮数 |
| `num_mini_batches` | 4 | |
| `learning_rate` | 3e-4 | |
| `schedule` | adaptive | adaptive KL 调度 |
| `gamma` / `lam` | 0.99 / 0.95 | GAE |
| `clip_param` | 0.2 | PPO clip |
| `entropy_coef` | 0.0005 | 低探索，精修模式 |
| `desired_kl` | 0.008 | 保守更新步幅 |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `value_loss_coef` | 1.0 | |
| `use_clipped_value_loss` | True | |

---

## 6. 实验迭代与训练结果

### 6.1 v1（HFOV 90°, cam_z 1.80, pitch 15°）— 失败

| 项 | 值 |
|---|---|
| 日志 | `logs/20260427_155157_train_geo_edge_n1024.log` |
| 启动 commit | `fcdc732c` |
| 训练时长 | 约 46 分钟后人工 kill（iter 1139 / 2000） |
| `success_rate_ema` | 全程 0.0000 |
| `Mean episode length` | 1079（顶满） |
| 根因 | 近场 FoV 覆盖不足，`visible=0` 导致策略失观测 |

### 6.2 v2（HFOV 120°, cam_z 1.30, pitch 25°）— 成功

| 项 | 值 |
|---|---|
| 日志 | `logs/20260427_171538_train_geo_edge_v2_n1024.log` |
| 启动 commit | `a519019b` |
| 训练时长 | **1h 24min 04s**（iter 0 → 1999 完整跑完） |
| 吞吐 | ~24.3k steps/s（1024 env × 64 steps/iter，RTX 5090） |
| Checkpoint 目录 | `third_party/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift_geo_edge/2026-04-27_17-15-41/` |

**冒烟验证（30 iter, num_envs=128）**：

| iter | v1 smoke `frac_inserted` | v2 smoke `frac_inserted` |
|---|---|---|
| 25 | 0 | 0 |
| 27 | 0 | 0.031 |
| 29 | 0 | **0.070** |

v2 smoke iter 29 已出现 v1 全程未见过的插入行为，第一次直接证据。

### 6.3 v2 学习曲线（full 训练）

| EMA 里程碑 | v2 首次达到 | 15D 基线首次达到 | v2 延迟 |
|---|---|---|---|
| 0.10 | iter **715** | iter ~500 | +215 |
| 0.50 | iter **938** | iter **699** | +239 |
| 0.70 | iter **1232** | iter ~900 | +332 |
| 0.80 | iter **1313** | iter **1099** | +214 |
| 0.90（峰值） | iter **1531** | iter ~1400 | +131 |

**v2 vs v1 vs 15D 基线 · 终盘（iter 1999）**：

| 指标 | v1 | **v2** | 15D 基线 |
|---|---|---|---|
| `success_rate_ema` | 0.0000 | **0.8063** | 0.9785 |
| 峰值 EMA | 0.0000 | **0.9040 @ iter 1531** | 0.9792 |
| `success_rate_total`（累计） | 0.0000 | 0.5431 | 0.5xxx（接近） |
| `Mean reward` 尾段 | -1370 | **+21.69** | +73.54 |
| `Mean episode length` 尾段 | 1079（超时） | **505** | 440 |
| `err/lateral_mean` 尾段 | 0.28 | 0.28 | 0.18 |
| `err/yaw_deg_mean` 尾段 | 4.6° | 5.2° | 4.2° |
| `err/insert_norm_mean` 尾段 | 0.02 | 0.09 | 0.18 |
| `diag/pallet_disp_xy_mean` 尾段 | 0.26 | 0.08 | 0.22 |
| `phase/frac_inserted` 尾段 | 0 | 0.13 | 0.12 |
| `phase/frac_lifted` 尾段 | 0 | 0.05 | 0.11 |

### 6.4 尾段 regression 现象

从 iter 1531（EMA 0.9040）到 iter 1999（EMA 0.8063），策略出现约 10 pp 下滑。观察 `Mean action noise std` 在末期降至 **0.05**（初始 0.5），同时 `entropy_coef=0.0005` 较低，推测是 std 过度收敛后对初始位姿分布的抗噪能力下降。

---

## 7. 结论

1. **方案可行性验证完毕**：21D 几何边缘观测 + 9D proprio + MLP 策略**能学会**完整的 forklift-pallet-insert-lift 任务，峰值 EMA 0.9040。
2. **相机参数是关键可调变量**：仅改 HFOV（90→120°）+ cam_z（1.80→1.30）+ pitch（15→25°）就从"完全学不了"翻到"能收敛到 90%"。未来设计真相机硬件方案时，必须先用 FoV 覆盖几何验算"最近工况是否仍在视角内"。
3. **相对 15D 特权观测的代价可控**：
   - 收敛晚 130–330 iter（~10–17% 训练预算）
   - 峰值 EMA 差 7.5 pp（0.9040 vs 0.9792）
   - 终盘 EMA 差 17 pp（0.8063 vs 0.9785），受尾段 regression 影响更大
4. **观测表征是信息充分的**：策略能学习证明 2 条 pocket 短边 + proprio 构成了**任务状态的可用编码**，为 Phase 1B（真 CV pipeline）提供了上界参考。

---

## 8. 下一步计划

### 8.1 短期（v3，预估 1 天内可出结果）

**优先级 1：评估 peak checkpoint 真实成功率**
- 用 `model_1550.pt` 跑独立 evaluation（关闭 EMA 采样，统计真实 episode 成功率）
- 预期会更接近 0.90（EMA 低估当前策略）
- 命令：`isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Forklift-PalletInsertLift-GeometryEdgeObs-v0 --num_envs 256 --checkpoint ..._17-15-41/model_1550.pt`

**优先级 2：v3 缓解尾段 regression**

候选方案（选一项试验）：
1. **提高 `entropy_coef`**：0.0005 → 0.002，保留探索到更晚期
2. **提高 `init_noise_std` clamp 下界**：目前 ln(0.05)，改为 ln(0.10) 防 std 过度收敛
3. **Resume from `model_1550.pt`**：用更小 `learning_rate`（1e-4）精修 500 iter

### 8.2 中期（Phase 1B，真 CV sim2real 桥接）

1. **渲染 pipeline 接入**：启用 `TiledCameraCfg`，每 step 渲染 256×256 RGB（预估每 step 多 3-5 ms，训练时长 ×1.3 左右）
2. **CV 检出替换 GT 投影**：
   - Canny + `HoughLinesP` 提候选线段
   - 按平行性、长度比、连通性筛出 4 条托盘边
   - 对短边做 pocket 检测（线段之间低亮度矩形区域 = 有 pocket）
3. **对齐 21D 接口**：输出格式与本实验完全一致，策略权重直接加载
4. **域随机化**：光照、纹理、相机抖动、噪声
5. **迁移评估**：直接用 Phase 1A peak model 在 Phase 1B 环境评估 → 期望看到明显退化，量化 sim2real gap
6. **Distillation 兜底**：若 gap > 20 pp，做 teacher（GT 投影）→ student（真 CV）蒸馏

### 8.3 长期（sim→real）

1. 真机相机标定（内外参、畸变）写入 env_cfg
2. 真机 CV pipeline 工程化（C++/ROS2 node，< 20 ms / frame）
3. 固定相机参数、渲染质量对齐后，直接 zero-shot 或少量 fine-tune 部署

---

## 9. 文件清单（本实验产出）

### 9.1 代码改动（`exp/geo-obs-impl` 分支）

| 文件 | 作用 |
|---|---|
| `forklift_pallet_insert_lift_project/isaaclab_patch/.../env_cfg.py` | 新增 `ForkliftPalletInsertLiftGeoEdgeEnvCfg`、相机内外参、托盘尺寸字段 |
| `forklift_pallet_insert_lift_project/isaaclab_patch/.../env.py` | 新增 `_init_geometry_edge_obs`、`_get_pallet_edges_world`、`_get_edge_obs`，修改 `_get_observations` 分支 |
| `forklift_pallet_insert_lift_project/isaaclab_patch/.../agents/rsl_rl_ppo_cfg.py` | 新增 `ForkliftInsertLiftGeoEdgePPORunnerCfg` |
| `forklift_pallet_insert_lift_project/isaaclab_patch/.../__init__.py` | 注册 `Isaac-Forklift-PalletInsertLift-GeometryEdgeObs-v0` |

### 9.2 文档

| 文件 | 作用 |
|---|---|
| `docs/0427-now/geometry_edge_obs_plan.md` | 设计方案（前期审计） |
| `docs/0427-now/geometry_edge_obs_result.md` | 迭代 v1/v2 的数据与失败分析 |
| `docs/0427-now/geometry_edge_obs_full_summary.md` | **本文档** |
| `docs/0427-now/baseline_replica_summary.md` | 15D 基线复现总结（前置工作） |

### 9.3 训练日志

| 文件 | 内容 |
|---|---|
| `logs/20260427_133516_train_s1.0zB_replica_n1024.log` | 15D 基线 2000 iter（EMA 0.9785） |
| `logs/20260427_154813_smoke_train_geo_edge.log` | v1 smoke 30 iter |
| `logs/20260427_155157_train_geo_edge_n1024.log` | v1 full（1139 iter 人工 kill） |
| `logs/20260427_171300_smoke_train_geo_edge_v2.log` | v2 smoke 30 iter |
| `logs/20260427_171538_train_geo_edge_v2_n1024.log` | **v2 full 2000 iter（peak 0.9040）** |

### 9.4 Model Checkpoints

- 目录：`third_party/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift_geo_edge/2026-04-27_17-15-41/`
- 每 50 iter 存一份：`model_0.pt` ~ `model_1999.pt`
- **推荐使用**：`model_1550.pt`（peak EMA ~0.90 附近）

### 9.5 Git 历史

```
5de517f9 docs: finalize Phase 1A v2 result (EMA peak 0.90, final 0.81)
a519019b feat(obs): tune virtual camera for 21D edge obs (Phase 1A v2)
7877ce0b docs: fix internal path reference in geometry_edge_obs_plan
66d1c002 docs: rename docs/0427- to docs/0427-now
fcdc732c feat(obs): add 21D geometry edge observation variant (Phase 1A)
61081469 docs: add 0427 baseline replica summary on exp/geo-obs-base
```
