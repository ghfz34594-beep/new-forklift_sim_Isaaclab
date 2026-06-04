#!/bin/bash
set -e
export ISAACLAB_PATH=/data/jianshi/projects/forklift_sim/IsaacLab
export PYTHONPATH=/data/jianshi/projects/forklift_sim_exp9/forklift_expert_policy_project:$PYTHONPATH

LOGDIR=/data/jianshi/projects/forklift_sim_exp9/logs/smoke_s1.0v_phase4/50ep
mkdir -p $LOGDIR
PY=$ISAACLAB_PATH/_isaac_sim/python.sh
SC=/data/jianshi/projects/forklift_sim_exp9/forklift_expert_policy_project/scripts/play_expert.py

: > $LOGDIR/progress.log

for S in 0 7 13 42 55 88 99 123 177 200; do
  echo "=== seed=$S start $(date +%H:%M:%S) ===" >> $LOGDIR/progress.log
  $PY $SC --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
    --num_envs 1 --headless --episodes 5 --seed $S \
    --log_file $LOGDIR/s${S}.log > $LOGDIR/f${S}.log 2>&1
  echo "=== seed=$S done exit=$? $(date +%H:%M:%S) ===" >> $LOGDIR/progress.log
done
echo "ALL_DONE" >> $LOGDIR/progress.log
