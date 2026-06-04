# Toyota Dual-Camera Pipeline

This directory contains the paper-aligned workflow:

- `teleop_dual_camera.py`: keyboard validation for drive/steer/lift and dual camera visibility.
- `collect_teleop_approach_dataset.py`: collect API/keyboard approach demonstrations for BC warm start.
- `rollout_recorder.py`: shared non-web API rollout recorder.
- `validate_teleop_dataset.py`: validate formal teleop sessions before BC training.
- `train_approach_bc.py`: train behavior-cloning warm start for the approach actor.
- `collect_decision_dataset.py`: collect simulated dual-camera snapshots for the loading decision classifier.
- `train_loading_decision.py`: train the supervised lift / do-not-lift classifier.
- `run_approach_decision_lift.py`: run approach policy, stop, decision hook, then scripted lift/reverse/lower.
- `forklift_api.py`: non-web imperative API wrapper for scripts.

Typical commands:

```bash
# All commands below can be run through this wrapper so isaaclab.sh uses the
# known-good env_isaaclab conda environment instead of whatever shell happens
# to be active.
RUN=/home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/run_isaaclab_env.sh

# Student/RL visual training now uses a wide fixed forklift-mounted dual-camera
# view plus unobstructed third-person top-down review.  Multi-env RGB
# interference is solved as a scene invariant before training, not learned by
# the student and not defined as "16 env passed".  The concrete solution is:
# - isolate envs spatially with env_spacing=20;
# - keep camera_far=8 and fail scene setup unless far_clip <= 0.45 * env_spacing
#   when per-env rooms are disabled;
# - keep vision rooms off by default so third-person/top-down review can see the
#   forklift, forks, and pallet;
# - use the wide camera config (HFOV=100, pos=(50,+/-45,190), rpy=(0,65,+/-5))
#   so the pallet remains visible at long range and large yaw;
# - validate with visual_isolation_summary.json: far-clip isolation, mosaic hash
#   uniqueness, full env coverage, and pallet_visibility_audit must pass.

# Room60 pressure test: validate foreign leakage under the Toyota-style camera.
# This writes visual_isolation_summary.json with foreign_leakage_pass and
# camera_learnability_pass separated.
python /home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/validate_room60_visual_isolation.py \
  --output_dir /home/uniubi/projects/forklift_sim_exp9/outputs/room60_visual_isolation_v1 \
  --task Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0 \
  --num_envs 16 --env_spacing 20 --camera_far 8 \
  --dual_camera_hfov_deg 100 \
  --pallet_visibility_audit --preinsert_pose_sweep \
  --coverage_mode stratified --coverage_count 16 \
  --mosaic_coverage_mode all --require_full_mosaic_coverage \
  --overwrite --headless

# Check dual-camera image quality before spending GPU time on visual training.
$RUN -p /home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/check_dual_camera_quality.py \
  --task Isaac-Forklift-PalletApproach-ToyotaDualCamera-v0 \
  --headless --num_envs 1 \
  --output_dir /home/uniubi/projects/forklift_sim_exp9/outputs/toyota_camera_check

# Validate the non-web API: reset, left/right cameras, drive, steer, lift, stop.
$RUN -p /home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/check_api_control.py \
  --task Isaac-Forklift-PalletApproach-ToyotaDualCamera-v0 \
  --headless --num_envs 1 \
  --output_dir /home/uniubi/projects/forklift_sim_exp9/outputs/toyota_api_check

# Validate PushSafe displacement, dual-camera API fields, and push termination.
$RUN -p /home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/check_pushsafe_api.py \
  --task Isaac-Forklift-PalletApproach-ToyotaDualCameraPushSafe-v0 \
  --headless --enable_cameras --num_envs 4

# Train Toyota-style approach PPO.
$RUN -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletApproach-ToyotaDualCamera-v0 \
  --headless --enable_cameras --num_envs 32 --max_iterations 2000

# Collect API/keyboard approach demonstrations for BC warm start.
$RUN -p /home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/collect_teleop_approach_dataset.py \
  --task Isaac-Forklift-PalletApproach-ToyotaDualCameraPushSafe-v0 \
  --output_dir /home/uniubi/projects/forklift_sim_exp9/data/toyota_approach_bc/formal_v1/session_001 \
  --num_envs 1 --flush_every 10

# Validate 20 formal sessions before BC. Existing session_001/session_002
# without metadata.csv are reachability evidence only, not BC training data.
python /home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/validate_teleop_dataset.py \
  --dataset_dir /home/uniubi/projects/forklift_sim_exp9/data/toyota_approach_bc/formal_v1 \
  --min_sessions 20 --min_clean_sessions 10 --require_summary

# Train BC warm-start weights from recorded demonstrations.
$RUN -p /home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/train_approach_bc.py \
  --dataset_dir /home/uniubi/projects/forklift_sim_exp9/data/toyota_approach_bc/formal_v1 \
  --output /home/uniubi/projects/forklift_sim_exp9/data/toyota_approach_bc/approach_bc_v1.pt \
  --epochs 10 --batch_size 32 --num_workers 0 --device cuda:0

# Primary CleanView45 RGB student path: validate, collect, train, and run a
# 10-episode closed-loop BC smoke eval.  This starts from a new output/data
# directory and does not mix old progress_v311_multi_env_clean_v1 data.
/home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/run_student_cleanview_training_pipeline.sh

# Diagnostic only: one Isaac process per visual env.  Use this to prove the
# teacher can insert cleanly when cross-env RGB contamination is impossible, not
# as the main fix for single-process multi-env interference.
python /home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/collect_teacher_distill_multiprocess.py \
  --output_dir /home/uniubi/projects/forklift_sim_exp9/data/toyota_teacher_distill/progress_v311_mp_diagnostic_v1 \
  --num_workers 4 --max_parallel 2 \
  --episodes_per_worker 40 --attempts_per_worker 80 \
  --dual_camera_hfov_deg 100 \
  --pallet_visibility_audit --preinsert_pose_sweep \
  --sanity_only \
  --headless

# Visualize a BC checkpoint before PPO; do not run long PPO if this still
# saturates drive/steer or pushes the pallet.
$RUN -p /home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/record_toyota_checkpoint_visual_eval.py \
  --task Isaac-Forklift-PalletApproach-ToyotaDualCameraPushSafe-v0 \
  --checkpoint_type bc \
  --checkpoint /home/uniubi/projects/forklift_sim_exp9/data/toyota_approach_bc/approach_bc_v1.pt \
  --output_dir /home/uniubi/projects/forklift_sim_exp9/outputs/toyota_bc_v1_eval \
  --headless --enable_cameras --episodes 3 --steps 720 --record_every 3

# PushSafe PPO from BC warm start.
$RUN -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0 \
  --bc_checkpoint /home/uniubi/projects/forklift_sim_exp9/data/toyota_approach_bc/approach_bc_v1.pt \
  --vision_acceptance_summary /home/uniubi/projects/forklift_sim_exp9/outputs/room60_visual_isolation_v1/visual_isolation_summary.json \
  --headless --enable_cameras --num_envs 16 --max_iterations 2000

# End-to-end gated helper: validate data, train BC, run PPO smoke, and only
# start main PPO when START_MAIN=1 is explicitly set.
DATASET_DIR=/home/uniubi/projects/forklift_sim_exp9/data/toyota_approach_bc/formal_v1 \
START_MAIN=0 \
/home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/run_pushsafe_training_from_bc.sh

# Manual physical/API validation.
$RUN -p /home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/teleop_dual_camera.py \
  --task Isaac-Forklift-PalletApproach-ToyotaDualCamera-v0 --num_envs 1

# Collect simulated loading-decision data from an approach checkpoint.
$RUN -p /home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/collect_decision_dataset.py \
  --checkpoint /path/to/model.pt \
  --output /home/uniubi/projects/forklift_sim_exp9/data/toyota_decision/decision_v1.pt \
  --headless --enable_cameras --num_envs 32 --steps 2000

# Train decision classifier outside IsaacSim.
PYTHONPATH=/home/uniubi/projects/forklift_sim/IsaacLab/source/isaaclab_tasks:/home/uniubi/projects/forklift_sim_exp9/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks:$PYTHONPATH \
python /home/uniubi/projects/forklift_sim_exp9/scripts/toyota_pipeline/train_loading_decision.py \
  --dataset /home/uniubi/projects/forklift_sim_exp9/data/toyota_decision/decision_v1.pt \
  --output /home/uniubi/projects/forklift_sim_exp9/data/toyota_decision/decision_v1_model.pt
```
