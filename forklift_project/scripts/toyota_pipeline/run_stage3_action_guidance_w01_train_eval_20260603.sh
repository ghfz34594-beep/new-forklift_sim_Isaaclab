#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/jianshi/projects/forklift_sim_exp9}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/data/jianshi/projects/forklift_sim/IsaacLab}"
PIPELINE_DIR="${PROJECT_ROOT}/scripts/toyota_pipeline"
RUN_WRAPPER="${PIPELINE_DIR}/run_isaaclab_env.sh"

TASK="Isaac-Forklift-PalletApproach-V311LegacyAcceptedTeacherVisualFreshActionGuidanceW01-v0"
EXPERIMENT_NAME="v311_legacy_visual_teacher_guidance"
RUN_NAME="action_guidance_w01_500iter_20260603"

OUTPUT_ROOT="${PROJECT_ROOT}/outputs/accepted_teacher_visual_goal_20260601/v311_legacy_visual_teacher_guidance/action_guidance_w01"
STDOUT_DIR="${OUTPUT_ROOT}/train_stdout"
VISUAL_SUMMARY="${OUTPUT_ROOT}/visual_acceptance_train_task/visual_isolation_summary.json"
TRAIN_STDOUT="${STDOUT_DIR}/train_500iter_20260603.log"
TRAIN_PID="${STDOUT_DIR}/train_500iter_20260603.pid"
TRAIN_EXIT="${STDOUT_DIR}/train_500iter_20260603.exit_code"
TRAIN_COMMAND="${STDOUT_DIR}/train_500iter_20260603.command.sh"
EVAL_STDOUT="${STDOUT_DIR}/eval_model_499_64ep_20260603.log"
EVAL_OUTPUT="${OUTPUT_ROOT}/visual_eval_model_499_64ep_novideo_20260603"

mkdir -p "${STDOUT_DIR}" "${EVAL_OUTPUT}"
printf '%s\n' "$$" > "${TRAIN_PID}"

train_cmd=(
  "${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py
  --task "${TASK}"
  --num_envs 64
  --seed 20260601
  --max_iterations 500
  --experiment_name "${EXPERIMENT_NAME}"
  --run_name "${RUN_NAME}"
  --vision_acceptance_summary "${VISUAL_SUMMARY}"
  --headless
  --enable_cameras
  --device cuda:0
)

eval_cmd=(
  "${RUN_WRAPPER}" -p "${PIPELINE_DIR}/record_toyota_checkpoint_visual_eval.py"
  --task "${TASK}"
  --checkpoint "__CHECKPOINT__"
  --checkpoint_type ppo
  --output_dir "${EVAL_OUTPUT}"
  --num_envs 1
  --episodes 64
  --steps 720
  --record_every 720
  --fps 30
  --seed 20260601
  --disable_teacher_reference_reset
  --visual_clean_max_pallet_disp_xy_m 0.030
  --hard_lateral_abs_init_y_m 0.40
  --dual_camera_hfov_deg 100
  --dual_camera_left_pos 150 75 140
  --dual_camera_right_pos 150 -75 140
  --dual_camera_left_rpy_deg 0 40 -20
  --dual_camera_right_rpy_deg 0 40 20
  --camera_far 8
  --no_video
  --headless
  --enable_cameras
  --device cuda:0
)

{
  printf '#!/usr/bin/env bash\n'
  printf '%q ' "${train_cmd[@]}"
  printf '\n'
} > "${TRAIN_COMMAND}"
chmod +x "${TRAIN_COMMAND}"

on_exit() {
  local code=$?
  printf '%s\n' "${code}" > "${TRAIN_EXIT}"
}
trap on_exit EXIT

printf '[stage3-action-guidance] pid=%s\n' "$$"
printf '[stage3-action-guidance] train stdout: %s\n' "${TRAIN_STDOUT}"
printf '[stage3-action-guidance] eval stdout: %s\n' "${EVAL_STDOUT}"
printf '[stage3-action-guidance] run_name: %s\n' "${RUN_NAME}"

"${train_cmd[@]}" 2>&1 | tee "${TRAIN_STDOUT}"

run_dir="$(find "${ISAACLAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}" -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1)"
if [[ -z "${run_dir}" ]]; then
  echo "[stage3-action-guidance] could not find run dir for ${RUN_NAME}" >&2
  exit 2
fi
checkpoint="${run_dir}/model_499.pt"
if [[ ! -f "${checkpoint}" ]]; then
  checkpoint="$(find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"
fi
if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
  echo "[stage3-action-guidance] no checkpoint found in ${run_dir}" >&2
  exit 3
fi

printf '[stage3-action-guidance] eval checkpoint: %s\n' "${checkpoint}"
for i in "${!eval_cmd[@]}"; do
  if [[ "${eval_cmd[$i]}" == "__CHECKPOINT__" ]]; then
    eval_cmd[$i]="${checkpoint}"
  fi
done

"${eval_cmd[@]}" 2>&1 | tee "${EVAL_STDOUT}"
printf '[stage3-action-guidance] eval summary: %s/summary.json\n' "${EVAL_OUTPUT}"
