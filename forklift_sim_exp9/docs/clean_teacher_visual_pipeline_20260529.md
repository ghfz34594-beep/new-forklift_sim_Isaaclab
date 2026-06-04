# Clean Teacher / Visual Pipeline

This runbook implements the clean troubleshooting order for the forklift task:

1. Confirm keyboard physical reachability.
2. Retrain the privileged geo-edge Teacher.
3. Inspect Teacher success/loss curves and checkpoint evals.
4. Audit the dual-camera + 5D proprio visual tensor path.
5. Train a new visual policy from scratch for 500-1000 iterations.

Old `direct_visual_v41_curriculum_20260529` checkpoints are historical reference only. Do not use them as warm starts for this clean run.

## Physical Check

Launch keyboard teleop:

```bash
RUN_TELEOP=1 \
/data/jianshi/projects/forklift_sim_exp9/scripts/toyota_pipeline/run_clean_teacher_visual_pipeline.sh
```

Confirm W/S drive, A/D steering, Q/E lift, and manual pallet insertion. Then continue only after the check is manually confirmed.

## Teacher First

Run Teacher training and evaluation:

```bash
CONFIRM_PHYSICS_OK=1 \
RUN_VISUAL_TRAIN=0 \
/data/jianshi/projects/forklift_sim_exp9/scripts/toyota_pipeline/run_clean_teacher_visual_pipeline.sh
```

The Teacher uses `21D = 12D edge_obs + 9D proprio` and outputs 2D drive/steer. The script keeps `model_500.pt`, `model_1000.pt` when available, and the final checkpoint, then evaluates them with `eval_geoedge_checkpoint.py`.

Inspect:

```text
teacher/training_summary.json
teacher/eval/*_summary.json
```

## Visual Audit And Training

After Teacher curves look healthy:

```bash
CONFIRM_PHYSICS_OK=1 \
CONFIRM_TEACHER_OK=1 \
RUN_TEACHER=0 \
TEACHER_RUN_DIR=<clean_teacher_run_dir> \
VISUAL_ITERS=501 \
/data/jianshi/projects/forklift_sim_exp9/scripts/toyota_pipeline/run_clean_teacher_visual_pipeline.sh
```

The visual audit enforces:

```text
left camera image  -> ResNet34 -> 512D
right camera image -> ResNet34 -> 512D
concat -> 1024D
image projection -> 256D
5D Toyota proprio -> proprio encoder -> 128D
actor input -> 384D
actor output -> 2D drive/steer
```

The visual policy uses two RGB cameras plus 5D Toyota proprio. It does not receive the Teacher-only 9D privileged proprio.

## Reward Debugging

Only start reward sweeps if all three upstream checks pass:

1. Keyboard physical reachability.
2. Teacher success/loss behavior.
3. Visual shape audit.
4. Visual data transforms, BC labels, and hard-lateral filtering.

Change one reward factor per run, keep seed/env/iterations/eval fixed, and only combine factors after each one has shown an individual gain. For example, if a reward weight is suspected, sweep only that value such as `10 -> 8 -> 7 -> 6`; record success rate, loss, pallet displacement, dirty insert, and push-no-insert before trying combinations.

## Teacher Guidance Fallback

Do not start from Teacher/reference-trajectory guidance. It is a fallback only.

If physical reachability, Teacher, visual pipeline, data filtering, BC/PPO, single-factor reward sweeps, and small reward combinations all pass but visual PPO still fails, use Teacher/reference trajectory guidance. Align this with the guidance-curve idea in the Toyota paper:

https://arxiv.org/abs/2412.11503

The useful reference is the paper's intermediate clothoid-style path reward from forklift start to pallet. In this project it should be added only as an auxiliary guide, not as a replacement for the fresh visual pipeline.
