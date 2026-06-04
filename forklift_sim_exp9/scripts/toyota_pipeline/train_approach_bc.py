"""Behavior cloning warm start for Toyota dual-camera approach policy."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


parser = argparse.ArgumentParser(description="Train BC policy for Toyota dual-camera approach")
parser.add_argument("--dataset_dir", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument(
    "--action_source",
    choices=("action", "teacher", "relabel"),
    default="relabel",
    help="Use action_drive/action_steer, action_*_teacher, or action_*_relabel as BC labels.",
)
parser.add_argument(
    "--action_loss_space",
    choices=("tanh", "raw"),
    default="raw",
    help="Use tanh(pred) MSE for clipped labels, or raw pred MSE to match raw actor eval.",
)
parser.add_argument(
    "--prev_action_source",
    choices=("metadata", "label"),
    default="label",
    help=(
        "Source for proprio previous_drive/previous_steer. "
        "metadata uses recorded teacher previous actions; label derives previous actions from the selected BC labels."
    ),
)
parser.add_argument("--max_frac_abs_drive_gt_095", type=float, default=0.05)
parser.add_argument("--allow_saturated_labels", action="store_true")
parser.add_argument("--max_abs_lift_action", type=float, default=0.05, help="Drop rows after manual lift starts.")
parser.add_argument("--max_lift_joint_m", type=float, default=0.04, help="Drop rows where lift joint is already raised.")
parser.add_argument("--max_lift_height_m", type=float, default=0.04, help="Drop rows where fork lift height is already raised.")
parser.add_argument("--max_pallet_disp_xy_m", type=float, default=0.20, help="Drop heavily pushed rows from BC.")
parser.add_argument("--min_insert_depth_m", type=float, default=0.0, help="Keep rows at or above this insert depth.")
parser.add_argument("--episode_min_insert_depth_m", type=float, default=0.45, help="Keep only episodes that reach this insert depth.")
parser.add_argument(
    "--episode_max_pallet_disp_xy_m",
    type=float,
    default=0.20,
    help="Keep only episodes below this max episode-relative pallet displacement.",
)
parser.add_argument(
    "--clean_episode_max_pallet_disp_xy_m",
    type=float,
    default=None,
    help="Optional stricter episode max pallet displacement, e.g. 0.05 for clean-only BC.",
)
parser.add_argument(
    "--drop_hard_lateral_high_disp",
    action="store_true",
    help=(
        "Drop rows from hard-lateral episodes whose episode max pallet displacement exceeds "
        "--hard_lateral_max_episode_pallet_disp_xy_m. Requires dataset columns written by "
        "collect_teacher_approach_dataset.py."
    ),
)
parser.add_argument(
    "--hard_lateral_abs_init_y_m",
    type=float,
    default=0.40,
    help="Fallback threshold for hard-lateral tagging when episode_hard_lateral is absent.",
)
parser.add_argument(
    "--hard_lateral_max_episode_pallet_disp_xy_m",
    type=float,
    default=0.030,
    help="Visual-clean displacement threshold used with --drop_hard_lateral_high_disp.",
)
parser.add_argument(
    "--keep_all_episodes",
    action="store_true",
    help="Disable successful/episode-level filtering and keep rows using only row-level filters.",
)
parser.add_argument(
    "--require_metadata",
    action="store_true",
    help="Fail if a child session under dataset_dir does not contain metadata.csv.",
)
parser.add_argument(
    "--allow_legacy_dataset",
    action="store_true",
    help="Allow training from frozen legacy datasets for diagnostics only.",
)
parser.add_argument(
    "--train_backbone",
    action="store_true",
    help="Fine-tune the visual backbone instead of keeping ImageNet features frozen.",
)
args = parser.parse_args()

if Path(args.dataset_dir).expanduser().name == "progress_v311_multi_env_clean_v1" and not args.allow_legacy_dataset:
    raise RuntimeError(
        "Refusing to train from frozen legacy dataset progress_v311_multi_env_clean_v1. "
        "Use the CleanView45 pipeline for new student training, or pass --allow_legacy_dataset only for diagnostics."
    )


def _load_vision_actor_critic():
    """Load the policy module without importing IsaacLab/Omni task packages."""
    candidates = [
        Path("/data/jianshi/projects/forklift_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift"),
        Path(__file__).resolve().parents[2]
        / "forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift",
    ]
    task_dir = next((path for path in candidates if (path / "vision_actor_critic.py").is_file()), None)
    if task_dir is None:
        raise FileNotFoundError("Could not locate forklift_pallet_insert_lift/vision_actor_critic.py")

    package_name = "_forklift_bc_policy"
    package = types.ModuleType(package_name)
    package.__path__ = [str(task_dir)]
    sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.vision_actor_critic",
        task_dir / "vision_actor_critic.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load policy module from {task_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.VisionActorCritic


VisionActorCritic = _load_vision_actor_critic()


class TeleopApproachDataset(Dataset):
    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        csv_paths = self._discover_csvs(self.dataset_dir)
        if not csv_paths:
            raise FileNotFoundError(f"No metadata.csv found under: {self.dataset_dir}")
        self.action_key_drive, self.action_key_steer = self._action_keys_for_source()
        rows = []
        dropped = 0
        for csv_path in csv_paths:
            with csv_path.open(newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))
                self._validate_action_keys(csv_rows, csv_path)
                self._annotate_prev_actions(csv_rows)
                keep_episode_ids = self._kept_episode_ids(csv_rows)
                for row in csv_rows:
                    if not row.get("image_left") or not row.get("image_right"):
                        dropped += 1
                        continue
                    row["_session_dir"] = str(csv_path.parent)
                    episode_id = self._row_episode_id(row)
                    if keep_episode_ids is not None and episode_id not in keep_episode_ids:
                        dropped += 1
                        continue
                    if not self._keep_row(row):
                        dropped += 1
                        continue
                    rows.append(row)
        if not rows:
            raise RuntimeError(f"No usable approach rows found in {self.dataset_dir}; dropped={dropped}")
        self.rows = rows
        self.csv_paths = csv_paths
        self.dropped = dropped
        self.label_stats = self._label_stats(rows)
        self.prev_action_stats = self._prev_action_stats(rows)
        self._validate_label_stats()
        self.image_tf = transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def _discover_csvs(dataset_dir: Path) -> list[Path]:
        direct = dataset_dir / "metadata.csv"
        if direct.is_file():
            return [direct]
        csvs = sorted(path for path in dataset_dir.glob("*/metadata.csv") if path.is_file())
        if args.require_metadata:
            session_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
            missing = [str(path) for path in session_dirs if not (path / "metadata.csv").is_file()]
            if missing:
                raise FileNotFoundError("Sessions missing metadata.csv: " + ", ".join(missing[:10]))
        return csvs

    @staticmethod
    def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)

    @staticmethod
    def _bool(row: dict[str, str], key: str, default: bool = False) -> bool:
        value = str(row.get(key, "")).strip().lower()
        if value in ("true", "1", "yes"):
            return True
        if value in ("false", "0", "no"):
            return False
        return default

    @staticmethod
    def _row_episode_id(row: dict[str, str]) -> int:
        try:
            return int(float(row.get("_episode_id", row.get("episode_id", 0))))
        except ValueError:
            return 0

    def _split_episodes(self, rows: list[dict[str, str]]) -> list[tuple[int, list[dict[str, str]]]]:
        if not rows:
            return []
        if "episode_id" in rows[0]:
            buckets: dict[int, list[dict[str, str]]] = {}
            order: list[int] = []
            for row in rows:
                try:
                    episode_id = int(float(row.get("episode_id", 0)))
                except ValueError:
                    episode_id = 0
                row["_episode_id"] = str(episode_id)
                if episode_id not in buckets:
                    buckets[episode_id] = []
                    order.append(episode_id)
                buckets[episode_id].append(row)
            return [(episode_id, buckets[episode_id]) for episode_id in order]

        episodes: list[tuple[int, list[dict[str, str]]]] = []
        current: list[dict[str, str]] = []
        episode_id = 0
        for row in rows:
            row["_episode_id"] = str(episode_id)
            current.append(row)
            if self._bool(row, "done"):
                episodes.append((episode_id, current))
                episode_id += 1
                current = []
        if current:
            episodes.append((episode_id, current))
        return episodes

    def _kept_episode_ids(self, rows: list[dict[str, str]]) -> set[int] | None:
        episodes = self._split_episodes(rows)
        if args.keep_all_episodes:
            return {episode_id for episode_id, _ in episodes}
        kept: set[int] = set()
        max_disp_limit = float(args.episode_max_pallet_disp_xy_m)
        if args.clean_episode_max_pallet_disp_xy_m is not None:
            max_disp_limit = float(args.clean_episode_max_pallet_disp_xy_m)
        for episode_id, episode_rows in episodes:
            max_insert = max((self._float(row, "insert_depth_m") for row in episode_rows), default=0.0)
            disp_values = [self._float(row, "pallet_disp_xy_m") for row in episode_rows]
            initial_disp = disp_values[0] if disp_values else 0.0
            max_disp = max((value - initial_disp for value in disp_values), default=0.0)
            if max_insert >= float(args.episode_min_insert_depth_m) and max_disp <= max_disp_limit:
                kept.add(episode_id)
        return kept

    def _keep_row(self, row: dict[str, str]) -> bool:
        if abs(self._float(row, "action_lift")) > float(args.max_abs_lift_action):
            return False
        if self._float(row, "lift_joint_m") > float(args.max_lift_joint_m):
            return False
        if self._float(row, "lift_height_m") > float(args.max_lift_height_m):
            return False
        if self._float(row, "pallet_disp_xy_m") > float(args.max_pallet_disp_xy_m):
            return False
        if self._float(row, "insert_depth_m") < float(args.min_insert_depth_m):
            return False
        if bool(args.drop_hard_lateral_high_disp):
            hard_lateral = self._bool(row, "episode_hard_lateral", False)
            if "episode_hard_lateral" not in row:
                hard_lateral = abs(self._float(row, "init_y_m", 0.0)) >= float(args.hard_lateral_abs_init_y_m)
            episode_max_disp = self._float(
                row,
                "episode_max_pallet_disp_xy_m",
                self._float(row, "pallet_disp_xy_m", 0.0),
            )
            if hard_lateral and episode_max_disp > float(args.hard_lateral_max_episode_pallet_disp_xy_m):
                return False
        return True

    @staticmethod
    def _action_keys_for_source() -> tuple[str, str]:
        if args.action_source == "action":
            return ("action_drive", "action_steer")
        if args.action_source == "teacher":
            return ("action_drive_teacher", "action_steer_teacher")
        return ("action_drive_relabel", "action_steer_relabel")

    def _validate_action_keys(self, rows: list[dict[str, str]], csv_path: Path) -> None:
        if not rows:
            return
        keys = (self.action_key_drive, self.action_key_steer)
        missing = [key for key in keys if key not in rows[0]]
        if missing:
            raise KeyError(
                f"Requested --action_source {args.action_source}, but {csv_path} is missing columns: {missing}"
            )

    def _annotate_prev_actions(self, rows: list[dict[str, str]]) -> None:
        if args.prev_action_source == "metadata":
            for row in rows:
                row["_prev_drive_for_proprio"] = str(self._float(row, "prev_drive"))
                row["_prev_steer_for_proprio"] = str(self._float(row, "prev_steer"))
            return

        for _, episode_rows in self._split_episodes(rows):
            prev_drive = 0.0
            prev_steer = 0.0
            for row in episode_rows:
                row["_prev_drive_for_proprio"] = str(prev_drive)
                row["_prev_steer_for_proprio"] = str(prev_steer)
                prev_drive = self._float(row, self.action_key_drive)
                prev_steer = self._float(row, self.action_key_steer)

    def _label_stats(self, rows: list[dict[str, str]]) -> dict[str, float]:
        drive_key, steer_key = self.action_key_drive, self.action_key_steer
        drives = torch.tensor([self._float(row, drive_key) for row in rows], dtype=torch.float32)
        steers = torch.tensor([self._float(row, steer_key) for row in rows], dtype=torch.float32)
        return {
            "samples": float(len(rows)),
            "mean_drive": float(drives.mean().item()),
            "mean_abs_drive": float(drives.abs().mean().item()),
            "max_abs_drive": float(drives.abs().max().item()),
            "frac_abs_drive_gt_095": float((drives.abs() > 0.95).float().mean().item()),
            "mean_steer": float(steers.mean().item()),
            "mean_abs_steer": float(steers.abs().mean().item()),
            "max_abs_steer": float(steers.abs().max().item()),
            "frac_abs_steer_gt_095": float((steers.abs() > 0.95).float().mean().item()),
        }

    def _prev_action_stats(self, rows: list[dict[str, str]]) -> dict[str, float | str]:
        prev_drive = torch.tensor([self._float(row, "_prev_drive_for_proprio") for row in rows], dtype=torch.float32)
        prev_steer = torch.tensor([self._float(row, "_prev_steer_for_proprio") for row in rows], dtype=torch.float32)
        metadata_prev_drive = torch.tensor([self._float(row, "prev_drive") for row in rows], dtype=torch.float32)
        metadata_prev_steer = torch.tensor([self._float(row, "prev_steer") for row in rows], dtype=torch.float32)
        drive_delta = (prev_drive - metadata_prev_drive).abs()
        steer_delta = (prev_steer - metadata_prev_steer).abs()
        return {
            "prev_action_source": str(args.prev_action_source),
            "mean_abs_prev_drive": float(prev_drive.abs().mean().item()),
            "mean_abs_prev_steer": float(prev_steer.abs().mean().item()),
            "mean_abs_prev_drive_delta_from_metadata": float(drive_delta.mean().item()),
            "mean_abs_prev_steer_delta_from_metadata": float(steer_delta.mean().item()),
            "frac_prev_drive_delta_gt_0_1": float((drive_delta > 0.1).float().mean().item()),
            "frac_prev_steer_delta_gt_0_1": float((steer_delta > 0.1).float().mean().item()),
        }

    def _validate_label_stats(self) -> None:
        if args.max_frac_abs_drive_gt_095 is None or args.allow_saturated_labels:
            return
        frac = float(self.label_stats["frac_abs_drive_gt_095"])
        limit = float(args.max_frac_abs_drive_gt_095)
        if frac > limit:
            raise RuntimeError(
                f"BC label saturation gate failed: frac_abs_drive_gt_095={frac:.4f} > {limit:.4f}. "
                "Use teacher relabeling/filtering, or pass --allow_saturated_labels only for diagnostics."
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        session_dir = Path(row["_session_dir"])
        left = self.image_tf(Image.open(session_dir / row["image_left"]).convert("RGB"))
        right = self.image_tf(Image.open(session_dir / row["image_right"]).convert("RGB"))
        proprio = torch.tensor(
            [
                float(row["vx_mps"]),
                float(row["vy_mps"]),
                float(row["yaw_rate_radps"]),
                float(row["_prev_drive_for_proprio"]),
                float(row["_prev_steer_for_proprio"]),
            ],
            dtype=torch.float32,
        )
        action = torch.tensor(
            [float(row[self.action_key_drive]), float(row[self.action_key_steer])],
            dtype=torch.float32,
        )
        return {"image_left": left, "image_right": right, "proprio": proprio, "action": action}


def _make_policy(device: torch.device) -> VisionActorCritic:
    obs = {
        "image_left": torch.zeros((1, 3, 224, 224), device=device),
        "image_right": torch.zeros((1, 3, 224, 224), device=device),
        "proprio": torch.zeros((1, 5), device=device),
        "critic": torch.zeros((1, 15), device=device),
    }
    return VisionActorCritic(
        obs=obs,
        obs_groups={"policy": ["image_left", "image_right", "proprio"], "critic": ["critic"]},
        num_actions=2,
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        init_noise_std=0.18,
        noise_std_type="log",
        backbone_type="resnet34",
        pretrained_backbone_path=None,
        freeze_backbone=not bool(args.train_backbone),
        freeze_backbone_updates=0,
        imagenet_backbone_init=True,
        dual_camera=True,
    ).to(device)


def main() -> None:
    device = torch.device(args.device)
    dataset = TeleopApproachDataset(args.dataset_dir)
    print(
        "[bc] label_stats "
        + json.dumps(
            {
                **dataset.label_stats,
                "action_source": args.action_source,
                "action_keys": [dataset.action_key_drive, dataset.action_key_steer],
                "action_loss_space": args.action_loss_space,
                "prev_action_source": args.prev_action_source,
                "prev_action_stats": dataset.prev_action_stats,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    policy = _make_policy(device)
    policy.train()
    loss_fn = nn.MSELoss()
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    history = []
    for epoch in range(int(args.epochs)):
        total_loss = 0.0
        samples = 0
        for batch in loader:
            obs = {
                "image_left": batch["image_left"].to(device),
                "image_right": batch["image_right"].to(device),
                "proprio": batch["proprio"].to(device),
                "critic": torch.zeros((batch["proprio"].shape[0], 15), device=device),
            }
            target = batch["action"].to(device)
            pred = policy.act_inference(obs)
            if args.action_loss_space == "raw":
                loss = loss_fn(pred, target)
            else:
                loss = loss_fn(torch.tanh(pred), target.clamp(-1.0, 1.0))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(target.shape[0])
            samples += int(target.shape[0])
        epoch_loss = total_loss / max(samples, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"[bc] epoch={epoch} loss={epoch_loss:.6f} samples={samples}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": policy.state_dict(),
        "metadata": {
            "dataset_dir": str(args.dataset_dir),
            "samples": len(dataset),
            "sessions": len(dataset.csv_paths),
            "dropped_rows": int(dataset.dropped),
            "epochs": int(args.epochs),
            "loss": history[-1]["loss"] if history else None,
            "policy_class": "VisionActorCritic",
            "action_source": args.action_source,
            "action_keys": [dataset.action_key_drive, dataset.action_key_steer],
            "action_loss_space": args.action_loss_space,
            "prev_action_source": args.prev_action_source,
            "prev_action_stats": dataset.prev_action_stats,
            "label_stats": dataset.label_stats,
            "filters": {
                "max_abs_lift_action": float(args.max_abs_lift_action),
                "max_lift_joint_m": float(args.max_lift_joint_m),
                "max_lift_height_m": float(args.max_lift_height_m),
                "max_pallet_disp_xy_m": float(args.max_pallet_disp_xy_m),
                "min_insert_depth_m": float(args.min_insert_depth_m),
                "episode_min_insert_depth_m": float(args.episode_min_insert_depth_m),
                "episode_max_pallet_disp_xy_m": float(args.episode_max_pallet_disp_xy_m),
                "clean_episode_max_pallet_disp_xy_m": (
                    None if args.clean_episode_max_pallet_disp_xy_m is None else float(args.clean_episode_max_pallet_disp_xy_m)
                ),
                "drop_hard_lateral_high_disp": bool(args.drop_hard_lateral_high_disp),
                "hard_lateral_abs_init_y_m": float(args.hard_lateral_abs_init_y_m),
                "hard_lateral_max_episode_pallet_disp_xy_m": float(args.hard_lateral_max_episode_pallet_disp_xy_m),
                "keep_all_episodes": bool(args.keep_all_episodes),
            },
            "train_backbone": bool(args.train_backbone),
        },
    }
    torch.save(payload, output)
    output.with_suffix(".json").write_text(json.dumps({"history": history, **payload["metadata"]}, indent=2), encoding="utf-8")
    print(f"[bc] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
