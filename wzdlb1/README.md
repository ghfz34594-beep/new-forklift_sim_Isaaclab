# 二阶倒立摆强化学习项目

基于 Isaac Lab 的二阶倒立摆（Acrobot / Double Pendulum）强化学习训练项目。

## 模型结构

```
                    ┌─────────┐
                    │  桌面   │
                    │ (固定)  │
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │  电机   │ ← 驱动关节 (joint1)
                    │ (蓝色)  │   绕 Y 轴旋转
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │ 摆杆 1  │ ← 第一段摆杆 (绿色)
                    │ (绿色)  │
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │  轴承   │ ← 被动关节 (joint2)
                    │ (银色)  │   绕 Y 轴旋转
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │ 摆杆 2  │ ← 第二段摆杆 (橙色)
                    │ (橙色)  │
                    └─────────┘
```

**特点：**
- 所有关节在同一平面内（XZ 平面）
- 驱动关节（电机）安装在桌面侧面
- 两段摆杆初始状态自然下垂（朝下）
- 目标：控制驱动关节，使两段摆杆稳定在竖直向上位置

## 快速开始

### 1. 查看模型结构

首先查看模型初始结构，确认设计正确：

```bash
./view_model.sh
```

这会启动 Isaac Sim 并显示模型，你可以：
- 用鼠标旋转/缩放视角
- 观察摆杆自然下垂的初始状态
- 关闭窗口或按 Ctrl+C 退出

### 2. 开始训练

确认模型结构后，开始强化学习训练：

```bash
# 带 GUI 训练（可以看到训练过程）
./run_train_rotary_double_pendulum.sh

# 无头模式训练（更快）
./run_train_rotary_double_pendulum.sh --headless
```

训练参数可通过命令行调整：
```bash
./run_train_rotary_double_pendulum.sh --num_envs 8192 --headless
```

### 3. 查看训练结果

训练完成后，使用保存的模型进行回放：

```bash
# 使用最新的 checkpoint
./run_play_rotary_double_pendulum.sh --num_envs 1 --use_last_checkpoint

# 或指定具体的 checkpoint
./run_play_rotary_double_pendulum.sh --num_envs 1 --checkpoint /path/to/checkpoint.pth
```

## 目录结构

```
wzdlb1/
├── README.md                              # 本文件
├── view_model.sh                          # 查看模型结构脚本
├── run_train_rotary_double_pendulum.sh    # 训练脚本
├── run_play_rotary_double_pendulum.sh     # 回放脚本
├── scripts/
│   └── view_double_pendulum.py            # 模型查看器 Python 脚本
└── isaaclab_patch/                        # Isaac Lab 补丁文件
    └── rotary_double_pendulum/
        ├── assets/
        │   └── rotary_double_pendulum.urdf  # 模型 URDF 文件
        ├── rotary_double_pendulum_cfg.py    # 环境配置
        ├── rotary_double_pendulum_env.py    # 环境实现
        └── agents/
            └── rl_games_ppo_cfg.yaml        # PPO 训练配置
```

## 环境信息

- **Isaac Sim 目录**: `/home/uniubi/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/`
- **Isaac Lab 目录**: `/home/uniubi/projects/forklift_sim/IsaacLab`
- **Conda 环境**: `env_isaaclab`
- **训练日志**: `IsaacLab/logs/rl_games/rotary_double_pendulum_direct/`

## 奖励函数设计

```
奖励 = 高度奖励 + 平衡奖励 + 速度惩罚 + 动作惩罚 + 存活奖励

- 高度奖励: 摆杆末端越高越好
- 平衡奖励: 接近竖直向上时给予额外奖励
- 速度惩罚: 接近平衡时希望速度小
- 动作惩罚: 节省能量
- 存活奖励: 鼓励存活更长时间
```

## 常见问题

1. **训练时 reward 为负值？**
   - 这是正常的，二阶倒立摆是困难的控制问题
   - 继续训练，reward 会逐渐提升

2. **GUI 无法显示？**
   - 使用 `--headless` 参数进行无头训练
   - 或通过 NoMachine 等远程桌面软件连接

3. **如何调整训练参数？**
   - 编辑 `agents/rl_games_ppo_cfg.yaml` 修改 PPO 参数
   - 编辑 `rotary_double_pendulum_cfg.py` 修改环境参数
