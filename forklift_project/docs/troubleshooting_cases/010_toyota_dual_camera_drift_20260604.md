# 010 Toyota 双摄相机漂移问题复盘（2026-06-04）

## 背景

目标是在 `outputs/topdown_dual_camera_fork_visible_20260604/overview.png` 所代表的第三方俯视视角和双摄安装参数基础上，启动 Isaac teleop，让人用 WASD 控制叉车，并观察模型实际输入的左右双摄画面是否足够支撑学习。

运行入口：

```bash
cd /data/jianshi/projects/forklift_sim_exp9

scripts/toyota_pipeline/run_isaaclab_env.sh \
  -p scripts/toyota_pipeline/teleop_dual_camera.py \
  --task Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0 \
  --num_envs 1 \
  --final_rect_init \
  --overview_camera_20260604 \
  --device cuda:0
```

启动后应该确认日志里出现：

```text
[teleop] dual camera prims: left=/World/envs/env_0/CameraLeft right=/World/envs/env_0/CameraRight
```

主要看 `Forklift Dual Camera Model Input` 窗口；这是模型实际吃到的双摄 tensor。普通 Isaac viewport 标题栏里显示 `Perspective` 的窗口不能作为训练图像判断依据。

## 关键参数

`--final_rect_init` 对应训练初始位置范围：

```text
x    = [-4.0, -3.0] m
y    = [-0.6, 0.6] m
yaw  = [-14.32394487827058, +14.32394487827058] deg
```

`--overview_camera_20260604` 对应双摄和场景参数：

```text
env_spacing             = 20.0 m
dual_camera_far_clip_m  = 8.0 m
dual_camera_hfov_deg    = 100.0 deg
left_pos_local          = (150.0, 75.0, 140.0) cm = (1.50, 0.75, 1.40) m
right_pos_local         = (150.0, -75.0, 140.0) cm = (1.50, -0.75, 1.40) m
left_rpy_local_deg      = (0.0, 40.0, -20.0)
right_rpy_local_deg     = (0.0, 40.0, 20.0)
vision_room_enable      = False
stage_1_mode            = True
action_space            = 2
lift                    = disabled by default
action_guard/noise      = disabled for manual visual check
```

## 现象

俯视视角看起来正常，叉车和托盘相对关系没有明显问题。

但在 `Forklift Dual Camera Model Input` 中，左右双摄画面不对：叉车虽然会动，但视觉上像相机相对叉车发生漂移，甚至像相机运动速度比叉车快。前进过程中货叉/车体在双摄图像里会出现不符合固定车载相机的相对运动。

这说明问题不应简单归结为第三方俯视 viewport 或普通 Perspective 视口问题。

## 为什么四元数检查一开始没查出问题

诊断脚本曾打印相机相对叉车的局部位置、局部四元数和 forward/up 轴：

```text
left actual_local_wxyz=(+0.925417, +0.059391, +0.336824, -0.163176)
left actual_rpy_deg=(+0.0000, +40.0000, -20.0000)
left forward_in_body_xyz=(+0.7198, -0.2620, -0.6428)

right actual_local_wxyz=(+0.925417, -0.059391, +0.336824, +0.163176)
right actual_rpy_deg=(+0.0000, +40.0000, +20.0000)
right forward_in_body_xyz=(+0.7198, +0.2620, -0.6428)
```

这些数值本身是合理的：相机前向轴在车体坐标系中是向前且向下看。

漏诊原因：这个检查读的是 `raw_env._camera_left._view.get_world_poses([0])` 合成后的相机 world pose，只能证明“某个 Camera prim 的最终 world pose 数学上等于预期”。它不能证明 TiledCamera render product 在渲染时没有受到父级 transform 和手动 world pose 写入的双重影响。

换句话说：

```text
Camera prim 的 get_world_poses 正确
!=
RTX/TiledCamera 实际渲染图像没有父子变换竞争或双重运动
```

这个案例里，人的视觉观察比只看 pose 数值更能暴露问题。

## 核心修复

真实环境文件：

```text
/data/jianshi/projects/forklift_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py
```

同步镜像文件：

```text
/data/jianshi/projects/forklift_sim_exp9/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py
```

### 1. 双摄 prim 从 Robot/body 子节点改到 env 根节点

修改前：

```text
/World/envs/env_.*/Robot/body/CameraLeft
/World/envs/env_.*/Robot/body/CameraRight
```

修改后：

```text
/World/envs/env_.*/CameraLeft
/World/envs/env_.*/CameraRight
```

原因：双摄相机此前挂在 `Robot/body` 下，同时代码又每个 render 前手动 `set_world_poses()`。这可能让 TiledCamera render product 看到“父级 body 运动 + 手动 world pose 更新”的组合，表现成相机漂移或相机运动速度不对。

现在相机 prim 放在 env 根下，由代码每次 render 前显式用叉车 PhysX root pose 写入 world pose，避免父级车体 transform 再参与一次。

### 2. `_sync_dual_camera_poses()` 尊重传入的最新 PhysX pose

之前 `env_ids is None` 分支会忽略 `root_pos/root_quat` 参数，继续使用 `self.robot.data.root_pos_w/root_quat_w`。

修复后：

```python
if env_ids is None:
    env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
    base_pos = self.robot.data.root_pos_w if root_pos is None else root_pos
    base_quat = self.robot.data.root_quat_w if root_quat is None else root_quat
```

### 3. render 前从 PhysX 读取最新车体 root pose

新增：

```python
def _latest_robot_root_pose_from_physx(self):
    root_pose = self.robot.root_physx_view.get_root_transforms().clone()
    root_quat_xyzw = root_pose[:, 3:7]
    root_quat_wxyz = torch.cat((root_quat_xyzw[:, 3:4], root_quat_xyzw[:, :3]), dim=-1)
    return root_pose[:, :3], root_quat_wxyz
```

在 `step()` 中：

```python
if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
    root_pos, root_quat = self._latest_robot_root_pose_from_physx()
    self._sync_dual_camera_poses(root_pos=root_pos, root_quat=root_quat)
    self.sim.render()
```

## 辅助脚本改动

### `scripts/toyota_pipeline/teleop_dual_camera.py`

- 默认任务改为 `Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0`。
- 支持 `--final_rect_init` 和 `--overview_camera_20260604`。
- 默认显示 `Forklift Dual Camera Model Input`，这是 raw env0 left/right model input。
- teleop 启动时打印实际双摄 prim path：

```text
[teleop] dual camera prims: left=/World/envs/env_0/CameraLeft right=/World/envs/env_0/CameraRight
```

- 可选 Isaac dual viewport 的路径也同步改为 env-root camera。

### `scripts/toyota_pipeline/check_dual_camera_mount_sync.py`

- 严格在 `env.step(action)` 后读取实际 camera prim pose，不在 step 后手动补同步。
- 打印：
  - camera prim path
  - left/right local position
  - actual/expected local quaternion
  - actual/expected local RPY
  - camera forward/up axes in body frame
  - max position/orientation error

### `scripts/toyota_pipeline/run_isaaclab_env.sh`

- 修复 `-p scripts/...` 相对路径：wrapper 会在切到 IsaacLab 目录前把项目内相对脚本路径转换成绝对路径，避免 IsaacLab 在错误目录找脚本。

## 复测命令和结果

严格同步检查：

```bash
cd /data/jianshi/projects/forklift_sim_exp9

scripts/toyota_pipeline/run_isaaclab_env.sh \
  -p scripts/toyota_pipeline/check_dual_camera_mount_sync.py \
  --task Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0 \
  --num_envs 1 \
  --steps 120 \
  --drive 1.0 \
  --device cuda:0
```

关键输出：

```text
[camera_mount] prim_paths=left=/World/envs/env_0/CameraLeft right=/World/envs/env_0/CameraRight
[camera_mount] step=0   left_local=(+1.5000, +0.7500, +1.4000) right_local=(+1.5000, -0.7500, +1.4000)
[camera_mount] step=30  left_local=(+1.5000, +0.7500, +1.4000) right_local=(+1.5000, -0.7500, +1.4000)
[camera_mount] step=60  left_local=(+1.5000, +0.7500, +1.4000) right_local=(+1.5000, -0.7500, +1.4000)
[camera_mount] step=90  left_local=(+1.5000, +0.7500, +1.4000) right_local=(+1.5000, -0.7500, +1.4000)
[camera_mount] step=119 left_local=(+1.5000, +0.7500, +1.4000) right_local=(+1.5000, -0.7500, +1.4000)
[camera_mount] max_pos_err_m=0.000000 max_angle_err_rad=0.001381
```

teleop smoke：

```bash
scripts/toyota_pipeline/run_isaaclab_env.sh \
  -p scripts/toyota_pipeline/teleop_dual_camera.py \
  --task Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0 \
  --num_envs 1 \
  --final_rect_init \
  --overview_camera_20260604 \
  --scripted_smoke_steps 1 \
  --headless \
  --device cuda:0
```

关键输出：

```text
[teleop] task=Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0
[teleop] init_distribution=rect x=[-4.000,-3.000] m, y=[-0.600,0.600] m, yaw=[-14.323945,14.323945] deg
[teleop] camera=hfov=100.0 far=8.0 left_pos=(150.0, 75.0, 140.0) left_rpy=(0.0, 40.0, -20.0) room=False
[teleop] dual camera prims: left=/World/envs/env_0/CameraLeft right=/World/envs/env_0/CameraRight
```

## 接手模型的注意事项

1. 不要只用 `get_world_poses()` 和四元数判断视觉一定正确。这个案例说明 render product 的父子 transform 链路可能和读出来的最终 pose 不一致。
2. 重新跑 teleop 时必须确认日志里是 env-root camera path，不是旧的 `/Robot/body/CameraLeft|Right`。
3. 观察窗口应为 `Forklift Dual Camera Model Input`，不是普通 `Perspective` viewport。
4. 如果用户仍报告漂移，下一步应保存同一帧的：
   - 双摄 RGB tensor
   - topdown 截图
   - robot PhysX root pose
   - camera `_view.get_world_poses`
   - camera `data.pos_w/quat_w_world`
   - render frame index
5. 若用户运行的是旧 Isaac 进程，必须完全关闭后重启。Isaac 进程内已加载的 Python 模块不会自动热更新。

