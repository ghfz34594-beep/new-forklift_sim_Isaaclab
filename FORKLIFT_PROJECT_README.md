# Forklift Pallet Insert Lift Project

This repository contains an Isaac Lab based forklift pallet insertion and lifting project.  The Isaac Lab framework is kept at the repository root, and the forklift-specific environment, assets, scripts, and notes are bundled with it so another machine can clone the repository and inspect or run the project without copying the original `/data/jianshi/projects` workspace layout.

## What Is Included

- `source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/`: active Isaac Lab task code.
- `assets/`: local USD assets used by the task, including `forklift_c.usd` and pallet USD files.
- `forklift_project/forklift_pallet_insert_lift_project/`: patch-style project source and helper scripts from the standalone forklift task workspace.
- `forklift_project/scripts/`: experiment, evaluation, diagnostic, and Toyota pipeline scripts.
- `forklift_project/docs/`: experiment notes, pipeline notes, troubleshooting, and paper-oriented documentation.
- `forklift_project/deployment/`: deployment notes and inference scaffold.
- `forklift_project/forklift_web_control/`: small web-control helper.

## What Is Not Included

Large generated artifacts are intentionally excluded from git:

- training logs and Isaac Lab run directories
- `outputs/`
- dataset dumps under `data/`
- checkpoints such as `*.pt`, `*.pth`, `*.onnx`, and `*.h5`
- cache directories and compiled Python files

If a script references a checkpoint path under `logs/rsl_rl`, provide the checkpoint separately or adjust the script to point at your local checkpoint.

## Basic Usage

Install Isaac Lab dependencies for this checkout first, following the normal Isaac Lab installation flow for the Isaac Sim version used by your machine.

From the repository root:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --help
```

The active forklift environment code is registered inside:

```text
source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/
```

The task configuration now prefers local repository assets from `assets/`.  If those files are present, the environment does not require the original local path:

```text
/data/jianshi/projects/forklift_sim/assets
```

## Useful Starting Points

- `forklift_project/docs/README.md`
- `forklift_project/docs/clean_teacher_visual_pipeline_20260529.md`
- `forklift_project/docs/toyota_reference_curve_exploration_20260602.md`
- `forklift_project/deployment/README.md`
- `forklift_project/forklift_pallet_insert_lift_project/README.md`

## Notes For New Users

This repository contains the project code and assets, not a full trained model release.  To reproduce a training run, start from the task code and scripts, then create fresh outputs locally.  To reproduce a specific old result, copy the relevant checkpoint or dataset from the original machine separately.
