"""
叉车仿真主入口 — Isaac Sim 5.1 独立脚本。

启动顺序：
  1. 创建 SimulationApp（必须最先执行）
  2. 导入 Isaac 相关模块
  3. 构建仿真场景（地面 + 叉车 + 托盘 + 灯光）
  4. 在后台线程启动 FastAPI Web 服务
  5. 运行仿真主循环，每帧写入关节目标

运行方式：
  /home/uniubi/miniconda3/envs/env_isaaclab/bin/python main.py [--headless] [--port 8080]
"""
from __future__ import annotations

import argparse
import logging
import sys

# ------------------------------------------------------------------
# 命令行参数（必须在 SimulationApp 之前解析）
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="叉车 Web 控制仿真")
parser.add_argument("--headless", action="store_true", help="无窗口模式运行")
parser.add_argument("--port", type=int, default=4161, help="Web 服务端口")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Web 服务监听地址")
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("forklift.main")

# ------------------------------------------------------------------
# Step 1: 初始化 Isaac Sim（必须在任何 omni 导入之前）
# ------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({
    "headless": args.headless,
    "renderer": "RaytracedLighting",
    "window_width": 1920,
    "window_height": 1080,
})
logger.info("Isaac Sim 已启动")

# ------------------------------------------------------------------
# Step 2: SimulationApp 启动后导入 omni / pxr 模块
# ------------------------------------------------------------------
import numpy as np  # noqa: E402
import carb  # noqa: E402
import omni.isaac.core.utils.stage as stage_utils  # noqa: E402

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.articulations import ArticulationView  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, PhysxSchema  # noqa: E402

# ------------------------------------------------------------------
# Step 3: 项目模块
# ------------------------------------------------------------------
import os  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import ForkliftController  # noqa: E402
from web_server import set_controller, start_web_server  # noqa: E402

# ------------------------------------------------------------------
# 常量
# ------------------------------------------------------------------
FORKLIFT_PRIM_PATH = "/World/Forklift"
PALLET_PRIM_PATH   = "/World/Pallet"

FORKLIFT_INIT_POS  = np.array([-3.0, 0.0, 0.05])
PALLET_INIT_POS    = np.array([0.5,  0.0, 0.0])

# ForkliftC 关节名称（与 env_cfg.py 保持一致）
FRONT_WHEEL_JOINTS = ["left_front_wheel_joint", "right_front_wheel_joint"]
BACK_WHEEL_JOINTS  = ["left_back_wheel_joint",  "right_back_wheel_joint"]
ROTATOR_JOINTS     = ["left_rotator_joint",      "right_rotator_joint"]
LIFT_JOINT         = "lift_joint"


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def get_isaac_nucleus_dir() -> str:
    """
    获取 Isaac 资产目录，与 isaaclab.utils.assets.ISAAC_NUCLEUS_DIR 逻辑相同。
    读取 carb settings: /persistent/isaac/asset_root/cloud，再拼接 /Isaac。
    """
    try:
        settings = carb.settings.get_settings()
        cloud_root = settings.get("/persistent/isaac/asset_root/cloud")
        if cloud_root:
            return f"{cloud_root}/Isaac"
    except Exception:
        pass
    fallback = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
        "/Assets/Isaac/5.1/Isaac"
    )
    logger.warning("未能读取 Nucleus carb 配置，回退到云端地址: %s", fallback)
    return fallback


def ensure_pallet_physics(stage, prim_path: str) -> None:
    """
    为托盘 prim 强制应用 RigidBodyAPI（部分 pallet.usd 以纯视觉 prop 发布）。
    托盘设为动态刚体，受重力，可被叉臂物理顶起。
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        logger.warning("托盘 prim 不存在: %s", prim_path)
        return

    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

    rb = UsdPhysics.RigidBodyAPI(prim)
    rb.GetRigidBodyEnabledAttr().Set(True)
    rb.GetKinematicEnabledAttr().Set(False)   # 动态刚体，受物理驱动

    phx = PhysxSchema.PhysxRigidBodyAPI(prim)
    phx.GetDisableGravityAttr().Set(False)
    phx.GetMaxDepenetrationVelocityAttr().Set(2.0)

    # 质量（欧标托盘约 25 kg）
    if not prim.HasAPI(UsdPhysics.MassAPI):
        UsdPhysics.MassAPI.Apply(prim)
    UsdPhysics.MassAPI(prim).GetMassAttr().Set(25.0)

    logger.info("已为托盘应用物理 API: %s", prim_path)


def get_joint_indices(art_view: ArticulationView, names: list[str]) -> list[int]:
    """从 ArticulationView 的 dof_names 中查找关节索引。"""
    all_names: list[str] = list(art_view.dof_names)
    indices = []
    for name in names:
        if name in all_names:
            indices.append(all_names.index(name))
        else:
            raise RuntimeError(
                f"关节 '{name}' 未在叉车中找到。\n"
                f"可用关节: {all_names}"
            )
    return indices


# ------------------------------------------------------------------
# 场景构建
# ------------------------------------------------------------------

def build_scene(world: World, isaac_dir: str) -> None:
    """添加地面、叉车、托盘和灯光到场景。"""
    stage = world.stage

    # 地面
    world.scene.add_default_ground_plane()

    # 灯光 — 半球光
    dome = stage.DefinePrim("/World/DomeLight", "DomeLight")
    UsdLux.DomeLight(dome).GetIntensityAttr().Set(800)

    # 灯光 — 方向光（模拟阳光）
    distant = stage.DefinePrim("/World/SunLight", "DistantLight")
    sun = UsdLux.DistantLight(distant)
    sun.GetIntensityAttr().Set(3000)
    sun.GetAngleAttr().Set(0.5)
    UsdGeom.Xformable(distant).AddRotateXYZOp().Set(Gf.Vec3f(-60, 0, 30))

    # 叉车
    forklift_usd = f"{isaac_dir}/Robots/IsaacSim/ForkliftC/forklift_c.usd"
    logger.info("加载叉车 USD: %s", forklift_usd)
    stage_utils.add_reference_to_stage(forklift_usd, FORKLIFT_PRIM_PATH)

    fk_prim = stage.GetPrimAtPath(FORKLIFT_PRIM_PATH)
    fk_xf = UsdGeom.Xformable(fk_prim)
    fk_xf.ClearXformOpOrder()
    fk_xf.AddTranslateOp().Set(Gf.Vec3d(*FORKLIFT_INIT_POS.tolist()))

    # 托盘
    pallet_usd = f"{isaac_dir}/Props/Pallet/pallet.usd"
    logger.info("加载托盘 USD: %s", pallet_usd)
    stage_utils.add_reference_to_stage(pallet_usd, PALLET_PRIM_PATH)

    pa_prim = stage.GetPrimAtPath(PALLET_PRIM_PATH)
    pa_xf = UsdGeom.Xformable(pa_prim)
    pa_xf.ClearXformOpOrder()
    pa_xf.AddTranslateOp().Set(Gf.Vec3d(*PALLET_INIT_POS.tolist()))

    logger.info("场景资产加载完成")


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("  叉车 Web 控制仿真  (Isaac Sim 5.1)")
    logger.info("=" * 60)

    isaac_dir = get_isaac_nucleus_dir()
    logger.info("Isaac Nucleus 目录: %s", isaac_dir)

    # 创建物理世界
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 30.0,
    )

    # 构建场景（仅添加 USD 引用，不运行物理）
    build_scene(world, isaac_dir)

    # 第一次 reset：初始化物理引擎，prim 树完成构建
    world.reset()
    simulation_app.update()

    # 为托盘补全物理 API（reset 后 prim 已完整展开）
    ensure_pallet_physics(world.stage, PALLET_PRIM_PATH)

    # 创建 ArticulationView 并注册到 scene
    art_view = ArticulationView(
        prim_paths_expr=FORKLIFT_PRIM_PATH,
        name="forklift_view",
    )
    world.scene.add(art_view)

    # 第二次 reset：让 ArticulationView 完成内部初始化
    world.reset()
    simulation_app.update()

    if not art_view.is_initialized():
        logger.error("ArticulationView 初始化失败，请检查叉车 USD 路径和物理配置。")
        simulation_app.close()
        sys.exit(1)

    all_joints: list[str] = list(art_view.dof_names)
    logger.info("叉车关节列表 (%d): %s", len(all_joints), all_joints)

    # 获取关节索引
    try:
        front_ids  = get_joint_indices(art_view, FRONT_WHEEL_JOINTS)
        back_ids   = get_joint_indices(art_view, BACK_WHEEL_JOINTS)
        rotat_ids  = get_joint_indices(art_view, ROTATOR_JOINTS)
        lift_id    = get_joint_indices(art_view, [LIFT_JOINT])[0]
    except RuntimeError as e:
        logger.error("关节索引获取失败: %s", e)
        simulation_app.close()
        sys.exit(1)

    logger.info(
        "关节索引 — 前轮:%s 后轮:%s 转向:%s 起升:%d",
        front_ids, back_ids, rotat_ids, lift_id,
    )

    # 初始化控制器
    ctrl = ForkliftController()
    ctrl.register_articulation(
        articulation=art_view,
        front_wheel_indices=front_ids,
        back_wheel_indices=back_ids,
        rotator_indices=rotat_ids,
        lift_index=lift_id,
    )

    # 注入 Web 服务并在后台启动
    set_controller(ctrl)
    start_web_server(host=args.host, port=args.port)
    logger.info("Web 控制界面已启动: http://localhost:%d", args.port)

    # 仿真主循环
    logger.info("仿真运行中，按 Ctrl+C 退出…")
    try:
        while simulation_app.is_running():
            ctrl.apply_to_articulation()
            world.step(render=True)
    except KeyboardInterrupt:
        logger.info("收到退出信号")
    finally:
        simulation_app.close()
        logger.info("仿真已关闭。")


if __name__ == "__main__":
    main()
