# UNI Dog Stereo Camera Demo

这个 demo 会加载本地机器狗 USD，并在机身 `BASE_LINK` 上挂载一组双目相机。

## 文件位置

- 运行脚本：`/home/uniubi/projects/forklift_sim/IsaacLab/scripts/demos/sensors/dog_camera.py`
- 机器狗资产：`/home/uniubi/xuanyuan/dog/uni_0428.usd`

## 运行命令

推荐使用当前资产目录下的运行脚本：

```bash
cd /home/uniubi/xuanyuan/dog
./run_dog_camera.sh
```

也可以手动切环境后运行：

```bash
conda deactivate
source /home/uniubi/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

cd /home/uniubi/xuanyuan/dog
/home/uniubi/projects/forklift_sim/IsaacLab/isaaclab.sh -p dog_camera.py --enable_cameras
```

启动后确认日志里使用的是：

```text
/home/uniubi/miniconda3/envs/env_isaaclab/bin/python
```

不要使用 `/home/uniubi/test_0313/miniconda3/bin/python`，那个是 base 环境，不能导入 IsaacLab。

## 相机路径

左眼：

```text
/World/envs/env_0/Robot/BASE_LINK/left_cam
```

右眼：

```text
/World/envs/env_0/Robot/BASE_LINK/right_cam
```

## 在 UI 里显示相机画面

1. 启动脚本后，主 Viewport 保持 Perspective，用来看整体机器狗。
2. 在菜单里新建一个 Viewport。
3. 在新 Viewport 顶部的相机下拉菜单里选择 `left_cam` 或 `right_cam`。
4. 如果要同时看左右目，可以再新建一个 Viewport，分别选择 `left_cam` 和 `right_cam`。

## 调整相机位置

当前相机参数在 `dog_camera.py` 顶部：

```python
LEFT_CAMERA_POS = (0.23286, -0.00665, 0.01343)
STEREO_BASELINE_M = 0.06
```

右眼位置由左眼位置和双目基线自动计算：

```python
RIGHT_CAMERA_POS = (LEFT_CAMERA_POS[0], LEFT_CAMERA_POS[1] - STEREO_BASELINE_M, LEFT_CAMERA_POS[2])
```

如果 UI 里手动移动了相机，退出后不会自动保存。需要把 UI 中的 Transform 数值回填到 `LEFT_CAMERA_POS` 或 `STEREO_BASELINE_M`。

## Headless 验证

如果只想验证双目相机是否能输出图像，可以运行：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p scripts/demos/sensors/dog_camera.py --headless --enable_cameras
```

正常情况下日志会持续输出类似：

```text
left_camera rgb=(1, 480, 640, 3), depth=(1, 480, 640, 1)
right_camera rgb=(1, 480, 640, 3), depth=(1, 480, 640, 1)
```
