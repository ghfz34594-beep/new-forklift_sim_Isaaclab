# Forklift Simulation Projects

This repository mirrors the project workspace layout used on the development machine.

## Top-Level Layout

- `codex_orchestrator/`: local orchestration helper.
- `docs/`: shared project notes and quickstart docs.
- `forklift_pallet_insert_lift_project/`: standalone patch-style forklift task project.
- `forklift_sim/`: IsaacLab checkout plus required local assets and web-control helper.
- `forklift_sim_exp9/`: Exp9 experiment workspace, scripts, docs, deployment notes, and assets.
- `outputs/`: kept as an empty output placeholder; generated runs are not committed.
- `wzdlb1/`: rotary double pendulum helper project.
- `2.zip`, `2D`: small files from the original workspace.

## Not Committed

Large generated data is intentionally excluded:

- `logs/`
- `outputs/` contents
- `data/`
- model checkpoints such as `*.pt`, `*.pth`, `*.onnx`, `*.h5`
- Python caches and local runtime logs

The main Isaac Lab checkout is under:

```text
forklift_sim/IsaacLab
```

The active forklift task code is under:

```text
forklift_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift
```

Exp9 materials are under:

```text
forklift_sim_exp9
```
