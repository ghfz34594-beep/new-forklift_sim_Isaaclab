#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/jianshi/projects/forklift_sim_exp9}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/data/jianshi/projects/forklift_sim/IsaacLab}"
PIPELINE_DIR="${PROJECT_ROOT}/scripts/toyota_pipeline"
RUN_WRAPPER="${RUN_WRAPPER:-${PIPELINE_DIR}/run_isaaclab_env.sh}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-accepted_teacher_visual_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/${RUN_NAME}}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/toyota_teacher_distill/${RUN_NAME}}"

ACCEPTED_TEACHER_CHECKPOINT="${ACCEPTED_TEACHER_CHECKPOINT:-/data/jianshi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_toyota_geoedge_progress_teacher_v311_legacy_exact_freeze_actor450_v13/2026-06-01_00-44-50_v311_legacy_v9c_model450_freeze_to1999_20260601_004446_seed42_1024env_to2000/model_1999.pt}"
ACCEPTED_TEACHER_REPORT="${ACCEPTED_TEACHER_REPORT:-/data/jianshi/projects/forklift_sim_exp9/outputs/v311_legacy_v9c_model450_freeze_to1999_20260601_004446_seed42_1024env_to2000/eval_2048_final/v311_legacy_v9c_freeze450_model_1999_eval2048_seed20260427_acceptance.json}"

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-20260601}"
NUM_ENVS="${NUM_ENVS:-16}"
ENV_SPACING="${ENV_SPACING:-20}"
CAMERA_FAR="${CAMERA_FAR:-8}"
HFOV="${HFOV:-60}"
LEFT_POS=(${LEFT_POS:-120 55 150})
RIGHT_POS=(${RIGHT_POS:-120 -55 150})
LEFT_RPY=(${LEFT_RPY:-0 68 -8})
RIGHT_RPY=(${RIGHT_RPY:-0 68 8})

COLLECT_TASK="${COLLECT_TASK:-Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCollect-v0}"
VISUAL_TASK="${VISUAL_TASK:-Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV40Direct-v0}"

TARGET_CLEAN_EPISODES="${TARGET_CLEAN_EPISODES:-160}"
MAX_ATTEMPTED_EPISODES="${MAX_ATTEMPTED_EPISODES:-260}"
MAX_STEPS="${MAX_STEPS:-900}"
IMAGE_EVERY="${IMAGE_EVERY:-1}"
BC_EPOCHS="${BC_EPOCHS:-20}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-32}"
BC_LR="${BC_LR:-1e-4}"
VISUAL_ITERS="${VISUAL_ITERS:-500}"
VISUAL_NUM_ENVS="${VISUAL_NUM_ENVS:-64}"
SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-${VISUAL_NUM_ENVS}}"
SMOKE_ITERS="${SMOKE_ITERS:-2}"
RUN_COLLECT="${RUN_COLLECT:-1}"
RUN_BC="${RUN_BC:-1}"
RUN_VISUAL_TRAIN="${RUN_VISUAL_TRAIN:-1}"
RUN_VISUAL_EVAL="${RUN_VISUAL_EVAL:-1}"
ISO_COVERAGE_COUNT="${ISO_COVERAGE_COUNT:-4}"
ISO_MOSAIC_CHUNK_SIZE="${ISO_MOSAIC_CHUNK_SIZE:-128}"
RED_COMPONENT_GATE="${RED_COMPONENT_GATE:-3}"
MAX_SECOND_RED_AREA_PX="${MAX_SECOND_RED_AREA_PX:-2600}"
MIN_FORK_RED_AREA_PX="${MIN_FORK_RED_AREA_PX:-250}"
MAX_RED_AREA_FRACTION="${MAX_RED_AREA_FRACTION:-0.45}"

HARD_LATERAL_ABS_INIT_Y_M="${HARD_LATERAL_ABS_INIT_Y_M:-0.40}"
HARD_LATERAL_MAX_DISP_M="${HARD_LATERAL_MAX_DISP_M:-0.030}"
DROP_HARD_LATERAL_HIGH_DISP="${DROP_HARD_LATERAL_HIGH_DISP:-1}"

VIS_DIR="${OUTPUT_ROOT}/visual_acceptance_collect_task"
VIS_TRAIN_DIR="${OUTPUT_ROOT}/visual_acceptance_train_task"
SHAPE_AUDIT_DIR="${OUTPUT_ROOT}/visual_shape_audit"
BC_OUTPUT="${OUTPUT_ROOT}/approach_student_bc.pt"
VISUAL_RUN_NAME="${VISUAL_RUN_NAME:-${RUN_NAME}_ppo${VISUAL_ITERS}}"

mkdir -p "${OUTPUT_ROOT}" "${DATASET_DIR}" "${SHAPE_AUDIT_DIR}"

log() {
  printf '[accepted-teacher-visual] %s\n' "$*"
}

write_command() {
  local path="$1"
  shift
  printf '%q ' "$@" > "${path}"
  printf '\n' >> "${path}"
  chmod +x "${path}"
}

find_latest_run_dir() {
  local run_name="$1"
  find "${ISAACLAB_DIR}/logs/rsl_rl" -mindepth 2 -maxdepth 2 -type d -name "*${run_name}" | sort | tail -n 1
}

find_latest_checkpoint() {
  local run_dir="$1"
  find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1
}

python3 - "${ACCEPTED_TEACHER_REPORT}" "${ACCEPTED_TEACHER_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
if not checkpoint.is_file():
    raise SystemExit(f"accepted teacher checkpoint missing: {checkpoint}")
payload = json.loads(report.read_text(encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit(f"accepted teacher report is not passing: {report}")
PY

FRESH_CAMERA_ARGS=(
  --env_spacing "${ENV_SPACING}"
  --camera_far "${CAMERA_FAR}"
  --dual_camera_hfov_deg "${HFOV}"
  --dual_camera_left_pos "${LEFT_POS[@]}"
  --dual_camera_right_pos "${RIGHT_POS[@]}"
  --dual_camera_left_rpy_deg "${LEFT_RPY[@]}"
  --dual_camera_right_rpy_deg "${RIGHT_RPY[@]}"
  --no_vision_room
)

cat > "${OUTPUT_ROOT}/plan.md" <<'EOF'
# 已验收 Teacher -> Fresh Visual Pipeline Goal

## 执行约束
- 只继承已验收 privileged teacher checkpoint: `__ACCEPTED_TEACHER_CHECKPOINT__`。
- visual pipeline 必须 fresh restart：不加载旧 visual checkpoint，不复用旧 visual run 结果，不把旧 visual 路线当作依据。
- visual actor 使用双目 RGB + 5D Toyota proprio；9D privileged proprio 只属于 teacher/特权信息侧，不进入 visual actor。
- Teacher 引导 / Toyota 论文引导曲线不放在主线前面，只能作为最终兜底方案。

## 主线步骤
1. 确认已验收 teacher 可用，并只用它生成 fresh visual 数据。
2. 逐项核对视觉链路：两路 RGB -> ResNet34 -> 双目拼接 -> 图像投影 -> 5D proprio 编码 -> actor 融合输入 -> 2D action。
3. visual gate 必须通过：相机可见性、无串环境、room、sentinel、geometry、mosaic coverage 都保持开启。
4. 采集 fresh teacher RGB/action 数据，并标记 hard-lateral 与高托盘位移样本。
5. 过滤 hard-lateral 且 max pallet displacement > __HARD_LATERAL_MAX_DISP_M__ m 的样本，再训练 fresh BC warm start。
6. 使用 fresh BC checkpoint 启动 visual PPO __VISUAL_ITERS__ iter。

## 维度验收
- 每路摄像头输入应为 [N, 3, 224, 224]。
- 每路 ResNet34 输出应为 [N, 512]。
- 双目拼接应为 [N, 1024]，图像投影应为 [N, 256]。
- visual actor proprio 应为 [N, 5]，proprio encoder 输出应为 [N, 128]。
- actor fused input 应为 [N, 384]。
- actor 输出应为 [N, 2]，语义为前进/后退 + 转向。

## 失败排查顺序
前提：底层物理可达性和视觉输入数据通路都已经确认无误。

1. 基础物理可达性 OK：键盘能插、能转向，目标 reset 分布内物理上可完成。
2. Teacher 能正常训练和独立验收。
3. 视觉输入 pipeline 的数据变换无误：相机张量、image transform、5D proprio、BC label、hard-lateral 标记/过滤都和 audit 一致。
4. 若以上都 OK 但最终仍训不出来，优先排查奖励函数。

## 奖励函数调试方法
- 单因素实验：一次只改一个影响因素。例如分析出 3 个可疑因素时，每次只动其中一个；权重可以按 10 -> 8 -> 7 -> 6 逐步试。
- 每次记录成功率、loss、max pallet displacement、dirty insert、push-no-insert。
- 单因素试完后，若有多个因素分别改善不同失败模式，再做少量组合实验。

## 最后兜底
- 不允许一开始直接上 Teacher 引导或 Toyota 论文引导曲线。
- 触发条件：物理可达性、teacher、visual 通路、数据质量、BC/PPO、奖励单因素和少量组合实验都排查后仍失败。
- 兜底方案：参考 Toyota 论文 https://arxiv.org/abs/2412.11503 的中间引导曲线思路，或直接使用 teacher 引导；它只能作为辅助 guide，不替代主线 fresh visual pipeline。
EOF
python3 - "${OUTPUT_ROOT}/plan.md" "${ACCEPTED_TEACHER_CHECKPOINT}" "${HARD_LATERAL_MAX_DISP_M}" "${VISUAL_ITERS}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
replacements = {
    "__ACCEPTED_TEACHER_CHECKPOINT__": sys.argv[2],
    "__HARD_LATERAL_MAX_DISP_M__": sys.argv[3],
    "__VISUAL_ITERS__": sys.argv[4],
}
text = path.read_text(encoding="utf-8")
for old, new in replacements.items():
    text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
PY

log "running shape audit"
shape_cmd=(
  "${RUN_WRAPPER}" -p "${PIPELINE_DIR}/audit_visual_actor_shapes.py"
  --task "${VISUAL_TASK}"
  --num_envs 2
  --seed "${SEED}"
  --output "${SHAPE_AUDIT_DIR}/summary.json"
  --dual_camera_hfov_deg "${HFOV}"
  --dual_camera_left_pos "${LEFT_POS[@]}"
  --dual_camera_right_pos "${RIGHT_POS[@]}"
  --dual_camera_left_rpy_deg "${LEFT_RPY[@]}"
  --dual_camera_right_rpy_deg "${RIGHT_RPY[@]}"
  --camera_far "${CAMERA_FAR}"
  --headless --enable_cameras --device "${DEVICE}"
)
write_command "${SHAPE_AUDIT_DIR}/audit_command.sh" "${shape_cmd[@]}"
"${shape_cmd[@]}" 2>&1 | tee "${SHAPE_AUDIT_DIR}/audit.log"

log "running fresh collection-task visual isolation"
python3 "${PIPELINE_DIR}/validate_room60_visual_isolation.py" \
  --output_dir "${VIS_DIR}" \
  --task "${VISUAL_TASK}" \
  --num_envs "${NUM_ENVS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --steps 120 \
  --record_every 2 \
  --fps 20 \
  "${FRESH_CAMERA_ARGS[@]}" \
  --red_component_gate "${RED_COMPONENT_GATE}" \
  --max_second_red_area_px "${MAX_SECOND_RED_AREA_PX}" \
  --min_fork_red_area_px "${MIN_FORK_RED_AREA_PX}" \
  --max_red_area_fraction "${MAX_RED_AREA_FRACTION}" \
  --sentinel_room_probes_all_envs \
  --record_mosaic \
  --mosaic_max_envs "${NUM_ENVS}" \
  --mosaic_cols 4 \
  --coverage_mode stratified \
  --coverage_count "${ISO_COVERAGE_COUNT}" \
  --mosaic_coverage_mode all \
  --mosaic_chunk_size "${ISO_MOSAIC_CHUNK_SIZE}" \
  --require_full_mosaic_coverage \
  --no_mosaic_save_frames \
  --overwrite

VIS_SUMMARY="${VIS_DIR}/visual_isolation_summary.json"

log "running visual-training task isolation"
python3 "${PIPELINE_DIR}/validate_room60_visual_isolation.py" \
  --output_dir "${VIS_TRAIN_DIR}" \
  --task "${VISUAL_TASK}" \
  --num_envs "${VISUAL_NUM_ENVS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --steps 120 \
  --record_every 2 \
  --fps 20 \
  "${FRESH_CAMERA_ARGS[@]}" \
  --red_component_gate "${RED_COMPONENT_GATE}" \
  --max_second_red_area_px "${MAX_SECOND_RED_AREA_PX}" \
  --min_fork_red_area_px "${MIN_FORK_RED_AREA_PX}" \
  --max_red_area_fraction "${MAX_RED_AREA_FRACTION}" \
  --sentinel_room_probes_all_envs \
  --record_mosaic \
  --mosaic_max_envs "${VISUAL_NUM_ENVS}" \
  --mosaic_cols 4 \
  --coverage_mode stratified \
  --coverage_count "${ISO_COVERAGE_COUNT}" \
  --mosaic_coverage_mode all \
  --mosaic_chunk_size "${ISO_MOSAIC_CHUNK_SIZE}" \
  --require_full_mosaic_coverage \
  --no_mosaic_save_frames \
  --overwrite

VISUAL_TRAIN_SUMMARY="${VIS_TRAIN_DIR}/visual_isolation_summary.json"

if [[ "${RUN_COLLECT}" == "1" ]]; then
  collect_cmd=(
    "${RUN_WRAPPER}" -p "${PIPELINE_DIR}/collect_teacher_approach_dataset.py"
    --task "${COLLECT_TASK}"
    --checkpoint "${ACCEPTED_TEACHER_CHECKPOINT}"
    --output_dir "${DATASET_DIR}"
    --num_envs "${NUM_ENVS}"
    --target_clean_episodes "${TARGET_CLEAN_EPISODES}"
    --episodes "${MAX_ATTEMPTED_EPISODES}"
    --max_steps "${MAX_STEPS}"
    --image_every "${IMAGE_EVERY}"
    --flush_every 25
    --seed "${SEED}"
    --vision_acceptance_summary "${VIS_SUMMARY}"
    --relabel_teacher_actions
    --hard_lateral_abs_init_y_m "${HARD_LATERAL_ABS_INIT_Y_M}"
    --hard_lateral_max_episode_pallet_disp_xy_m "${HARD_LATERAL_MAX_DISP_M}"
    "${FRESH_CAMERA_ARGS[@]}"
    --device "${DEVICE}" --enable_cameras --headless
  )
  if [[ "${DROP_HARD_LATERAL_HIGH_DISP}" == "1" ]]; then
    collect_cmd+=(--drop_hard_lateral_high_disp_episodes)
  fi
  write_command "${OUTPUT_ROOT}/collect_command.sh" "${collect_cmd[@]}"
  log "collecting accepted-teacher visual data"
  "${collect_cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/collect.log"
fi

python3 - "${DATASET_DIR}" "${TARGET_CLEAN_EPISODES}" <<'PY'
import json
import sys
from pathlib import Path

dataset_dir = Path(sys.argv[1])
target_clean = int(sys.argv[2])
summary_path = dataset_dir / "summary.json"
metadata_path = dataset_dir / "metadata.csv"
if not summary_path.is_file():
    raise SystemExit(f"dataset summary missing after collection: {summary_path}")
if not metadata_path.is_file():
    raise SystemExit(f"dataset metadata missing after collection: {metadata_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
kept = int(summary.get("kept_episodes", 0))
attempted = int(summary.get("attempted_episodes", 0))
steps = int(summary.get("steps", 0))
if kept < target_clean:
    raise SystemExit(
        "teacher visual dataset is incomplete; refusing BC. "
        f"kept_episodes={kept}, target_clean_episodes={target_clean}, "
        f"attempted_episodes={attempted}, steps={steps}, dataset={dataset_dir}"
    )
print(
    "[accepted-teacher-visual] dataset gate passed: "
    f"kept_episodes={kept}, attempted_episodes={attempted}, steps={steps}",
    flush=True,
)
PY

if [[ "${RUN_BC}" == "1" ]]; then
  bc_cmd=(
    "${RUN_WRAPPER}" -p "${PIPELINE_DIR}/train_approach_student_from_teacher.py"
    --dataset_dir "${DATASET_DIR}"
    --output "${BC_OUTPUT}"
    --epochs "${BC_EPOCHS}"
    --batch_size "${BC_BATCH_SIZE}"
    --lr "${BC_LR}"
    --device "${DEVICE}"
    --action_source relabel
    --action_loss_space raw
    --prev_action_source label
    --max_frac_abs_drive_gt_095 0.05
    --clean_episode_max_pallet_disp_xy_m 0.05
    --drop_hard_lateral_high_disp
    --hard_lateral_abs_init_y_m "${HARD_LATERAL_ABS_INIT_Y_M}"
    --hard_lateral_max_episode_pallet_disp_xy_m "${HARD_LATERAL_MAX_DISP_M}"
    --train_backbone
  )
  write_command "${OUTPUT_ROOT}/bc_command.sh" "${bc_cmd[@]}"
  log "training BC warm start"
  "${bc_cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/bc.log"
fi

if [[ "${RUN_VISUAL_TRAIN}" == "1" ]]; then
  smoke_cmd=(
    "${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py
    --task "${VISUAL_TASK}"
    --num_envs "${SMOKE_NUM_ENVS}"
    --max_iterations "${SMOKE_ITERS}"
    --seed "${SEED}"
    --run_name "${VISUAL_RUN_NAME}_smoke"
    --bc_checkpoint "${BC_OUTPUT}"
    --vision_acceptance_summary "${VISUAL_TRAIN_SUMMARY}"
    --headless --enable_cameras --device "${DEVICE}"
  )
  write_command "${OUTPUT_ROOT}/visual_smoke_command.sh" "${smoke_cmd[@]}"
  log "running visual PPO smoke"
  "${smoke_cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/visual_smoke.log"

  train_cmd=(
    "${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py
    --task "${VISUAL_TASK}"
    --num_envs "${VISUAL_NUM_ENVS}"
    --max_iterations "${VISUAL_ITERS}"
    --seed "${SEED}"
    --run_name "${VISUAL_RUN_NAME}"
    --bc_checkpoint "${BC_OUTPUT}"
    --vision_acceptance_summary "${VISUAL_TRAIN_SUMMARY}"
    --headless --enable_cameras --device "${DEVICE}"
  )
  write_command "${OUTPUT_ROOT}/visual_train_command.sh" "${train_cmd[@]}"
  log "starting visual PPO: ${VISUAL_RUN_NAME}"
  "${train_cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/visual_train.log"

  visual_run_dir="$(find_latest_run_dir "${VISUAL_RUN_NAME}")"
  echo "${visual_run_dir}" > "${OUTPUT_ROOT}/visual_run_dir.txt"
  visual_checkpoint="$(find_latest_checkpoint "${visual_run_dir}")"
  echo "${visual_checkpoint}" > "${OUTPUT_ROOT}/visual_latest_checkpoint.txt"

  if [[ "${RUN_VISUAL_EVAL}" == "1" ]]; then
    eval_dir="${OUTPUT_ROOT}/visual_eval_$(basename "${visual_checkpoint}" .pt)"
    eval_cmd=(
      "${RUN_WRAPPER}" -p "${PIPELINE_DIR}/record_toyota_checkpoint_visual_eval.py"
      --task "${VISUAL_TASK}"
      --checkpoint "${visual_checkpoint}"
      --checkpoint_type ppo
      --output_dir "${eval_dir}"
      --num_envs 1
      --episodes 12
      --steps 720
      --record_every 3
      --fps 30
      --seed "${SEED}"
      --disable_teacher_reference_reset
      --visual_clean_max_pallet_disp_xy_m "${HARD_LATERAL_MAX_DISP_M}"
      --hard_lateral_abs_init_y_m "${HARD_LATERAL_ABS_INIT_Y_M}"
      --dual_camera_hfov_deg "${HFOV}"
      --dual_camera_left_pos "${LEFT_POS[@]}"
      --dual_camera_right_pos "${RIGHT_POS[@]}"
      --dual_camera_left_rpy_deg "${LEFT_RPY[@]}"
      --dual_camera_right_rpy_deg "${RIGHT_RPY[@]}"
      --camera_far "${CAMERA_FAR}"
      --save_raw_camera_frames
      --save_frame_metadata
      --headless --enable_cameras --device "${DEVICE}"
    )
    write_command "${OUTPUT_ROOT}/visual_eval_command.sh" "${eval_cmd[@]}"
    log "evaluating visual checkpoint"
    "${eval_cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/visual_eval.log"
  fi
fi

log "done: ${OUTPUT_ROOT}"
