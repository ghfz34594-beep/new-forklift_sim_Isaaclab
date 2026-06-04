#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-floor0}"

PROJECT_ROOT="${PROJECT_ROOT:-/data/jianshi/projects/forklift_sim_exp9}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/data/jianshi/projects/forklift_sim/IsaacLab}"
PIPELINE_DIR="${PROJECT_ROOT}/scripts/toyota_pipeline"
RUN_WRAPPER="${PIPELINE_DIR}/run_isaaclab_env.sh"

EXPERIMENT_NAME="v311_legacy_visual_curve_guidance"
SEED="${SEED:-20260601}"
NUM_ENVS="${NUM_ENVS:-64}"
DEVICE="${DEVICE:-cuda:0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
SMOKE_ITERATIONS="${SMOKE_ITERATIONS:-2}"

case "${VARIANT}" in
  floor0)
    TASK="Isaac-Forklift-PalletApproach-V311LegacyAcceptedTeacherVisualFreshCurvePhaseGatePreAlignW18R1Floor0-v0"
    RUN_SLUG="curve_phase_gate_pre_align_w18_r1_floor0"
    ;;
  soft)
    TASK="Isaac-Forklift-PalletApproach-V311LegacyAcceptedTeacherVisualFreshCurvePhaseGatePreAlignW18R1Soft-v0"
    RUN_SLUG="curve_phase_gate_pre_align_w18_r1_soft"
    ;;
  latch)
    TASK="Isaac-Forklift-PalletApproach-V311LegacyAcceptedTeacherVisualFreshCurvePhaseLatchPreAlignV5-v0"
    RUN_SLUG="curve_phase_latch_pre_align_v5"
    ;;
  *)
    echo "[stage2c-r1-ab] unknown variant: ${VARIANT}; expected floor0, soft, or latch" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_SLUG}_500iter_20260603"
SMOKE_RUN_NAME="${RUN_SLUG}_smoke_2iter_20260603"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/accepted_teacher_visual_goal_20260601/v311_legacy_visual_curve_guidance/${RUN_SLUG}"
STDOUT_DIR="${OUTPUT_ROOT}/train_stdout"
VISUAL_DIR="${OUTPUT_ROOT}/visual_acceptance_train_task"
SMOKE_DIR="${OUTPUT_ROOT}/smoke_2iter_20260603"
EVAL_OUTPUT="${OUTPUT_ROOT}/visual_eval_best_64ep_novideo_20260603"

VISUAL_SUMMARY="$(realpath -m "${VISUAL_DIR}/visual_isolation_summary.json")"
VISUAL_STDOUT="${VISUAL_DIR}/visual_isolation_stdout.log"
SMOKE_STDOUT="${SMOKE_DIR}/train.log"
TRAIN_STDOUT="${STDOUT_DIR}/train_500iter_20260603.log"
TRAIN_PID="${STDOUT_DIR}/train_500iter_20260603.pid"
TRAIN_EXIT="${STDOUT_DIR}/train_500iter_20260603.exit_code"
TRAIN_COMMAND="${STDOUT_DIR}/train_500iter_20260603.command.sh"
EVAL_STDOUT="${STDOUT_DIR}/eval_best_64ep_novideo_20260603.log"

mkdir -p "${VISUAL_DIR}" "${SMOKE_DIR}" "${STDOUT_DIR}" "${EVAL_OUTPUT}"
printf '%s\n' "$$" > "${TRAIN_PID}"

visual_cmd=(
  python3 "${PIPELINE_DIR}/validate_room60_visual_isolation.py"
  --output_dir "${VISUAL_DIR}"
  --task "${TASK}"
  --num_envs "${NUM_ENVS}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --steps 120
  --record_every 2
  --fps 20
  --dual_camera_hfov_deg 100
  --dual_camera_left_pos 150 75 140
  --dual_camera_right_pos 150 -75 140
  --dual_camera_left_rpy_deg 0 40 -20
  --dual_camera_right_rpy_deg 0 40 20
  --camera_far 8
  --red_component_gate 3
  --max_second_red_area_px 2600
  --min_fork_red_area_px 250
  --max_red_area_fraction 0.20
  --record_mosaic
  --mosaic_max_envs "${NUM_ENVS}"
  --mosaic_cols 4
  --coverage_mode stratified
  --coverage_count 4
  --mosaic_coverage_mode all
  --mosaic_chunk_size 128
  --require_full_mosaic_coverage
  --no_mosaic_save_frames
  --overwrite
)

smoke_cmd=(
  "${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py
  --task "${TASK}"
  --num_envs "${NUM_ENVS}"
  --seed "${SEED}"
  --max_iterations "${SMOKE_ITERATIONS}"
  --experiment_name "${EXPERIMENT_NAME}"
  --run_name "${SMOKE_RUN_NAME}"
  --vision_acceptance_summary "${VISUAL_SUMMARY}"
  --headless
  --enable_cameras
  --device "${DEVICE}"
)

train_cmd=(
  "${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py
  --task "${TASK}"
  --num_envs "${NUM_ENVS}"
  --seed "${SEED}"
  --max_iterations "${MAX_ITERATIONS}"
  --experiment_name "${EXPERIMENT_NAME}"
  --run_name "${RUN_NAME}"
  --vision_acceptance_summary "${VISUAL_SUMMARY}"
  --headless
  --enable_cameras
  --device "${DEVICE}"
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
  --seed "${SEED}"
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
  --device "${DEVICE}"
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

printf '[stage2c-r1-ab] variant=%s\n' "${VARIANT}"
printf '[stage2c-r1-ab] task=%s\n' "${TASK}"
printf '[stage2c-r1-ab] visual stdout=%s\n' "${VISUAL_STDOUT}"
printf '[stage2c-r1-ab] smoke stdout=%s\n' "${SMOKE_STDOUT}"
printf '[stage2c-r1-ab] train stdout=%s\n' "${TRAIN_STDOUT}"
printf '[stage2c-r1-ab] visual summary=%s\n' "${VISUAL_SUMMARY}"
printf '[stage2c-r1-ab] run_name=%s\n' "${RUN_NAME}"

"${visual_cmd[@]}" 2>&1 | tee "${VISUAL_STDOUT}"
"${smoke_cmd[@]}" 2>&1 | tee "${SMOKE_STDOUT}"
"${train_cmd[@]}" 2>&1 | tee "${TRAIN_STDOUT}"

run_dir="$(find "${ISAACLAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}" -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1)"
if [[ -z "${run_dir}" ]]; then
  echo "[stage2c-r1-ab] could not find run dir for ${RUN_NAME}" >&2
  exit 3
fi
checkpoint="${run_dir}/model_499.pt"
if [[ ! -f "${checkpoint}" ]]; then
  checkpoint="$(find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"
fi
if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
  echo "[stage2c-r1-ab] no checkpoint found in ${run_dir}" >&2
  exit 4
fi

printf '[stage2c-r1-ab] eval checkpoint=%s\n' "${checkpoint}"
for i in "${!eval_cmd[@]}"; do
  if [[ "${eval_cmd[$i]}" == "__CHECKPOINT__" ]]; then
    eval_cmd[$i]="${checkpoint}"
  fi
done

"${eval_cmd[@]}" 2>&1 | tee "${EVAL_STDOUT}"
printf '[stage2c-r1-ab] eval summary=%s/summary.json\n' "${EVAL_OUTPUT}"
