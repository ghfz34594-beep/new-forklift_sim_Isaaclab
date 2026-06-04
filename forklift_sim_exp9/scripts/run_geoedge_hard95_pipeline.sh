#!/usr/bin/env bash
# End-to-end runner for the GeoEdge Stage-A hard95 training plan.
#
# Typical usage:
#   MODE=smoke bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=smoke_v10 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe45 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v4 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v5 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v6 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v7 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v8 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v9 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v10 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v11 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v12 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v13 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v14 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v15 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v16 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v17 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v18 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=probe_v19 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v4 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v5 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v6 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v7 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v8 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v9 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v10 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v11 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v12 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v13 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v14 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v15 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v16 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v17 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v18 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main_v19 bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=main bash scripts/run_geoedge_hard95_pipeline.sh
#   MODE=confirm BEST_CHECKPOINT=/abs/path/model_XXX.pt bash scripts/run_geoedge_hard95_pipeline.sh
#
# MODE=all runs smoke -> probe -> main. Confirmation is intentionally separate
# because it should use the best checkpoint selected from the main evaluations.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB="${ISAACLAB:-/data/jianshi/projects/forklift_sim/IsaacLab}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/uniubi/miniconda3/envs/env_isaaclab}"
TASK="${TASK:-Isaac-Forklift-PalletInsertLift-GeometryEdgeObs-v0}"
MODE="${MODE:-smoke}"
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-1024}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-256}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/geoedge_hard95}"
PROBE_MIN_STRICT="${PROBE_MIN_STRICT:-0.89}"
PROBE_MIN_NEAR_STRICT="${PROBE_MIN_NEAR_STRICT:-0.60}"
PROBE_MAX_DIRTY="${PROBE_MAX_DIRTY:-0.34}"
PROBE_MAX_TIMEOUT="${PROBE_MAX_TIMEOUT:-0.16}"
BASE_RESUME_RUN="${BASE_RESUME_RUN:-2026-05-19_15-40-53_geoedge_recovery_v2_full_insert_seed42_iter30}"
BASE_RESUME_CHECKPOINT="${BASE_RESUME_CHECKPOINT:-model_67.pt}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$ISAACLAB/logs/rsl_rl/forklift_pallet_insert_lift_geo_edge/$BASE_RESUME_RUN/$BASE_RESUME_CHECKPOINT}"
PROFILE_MAIN="legacy_clean_pf_success_max_dirty_gate_recovery_v3_hard95"
PROFILE_FRAC45="legacy_clean_pf_success_max_dirty_gate_recovery_v3_hard95_frac45"
PROFILE_V4="legacy_clean_pf_success_max_dirty_gate_recovery_v4_turnroom_hard95"
PROFILE_V5="legacy_clean_pf_success_max_dirty_gate_recovery_v5_turnroom_shortfall_hard95"
PROFILE_V6="legacy_clean_pf_success_max_dirty_gate_recovery_v6_centerdist_gate_hard95"
PROFILE_V7="legacy_clean_pf_success_max_dirty_gate_recovery_v7_centerdist_harddirty_hard95"
PROFILE_V8="legacy_clean_pf_success_max_dirty_gate_recovery_v8_rootdist_successgate_hard95"
PROFILE_V9="legacy_clean_pf_success_max_dirty_gate_recovery_v9_rootdist_reverse_hard95"
PROFILE_V10="legacy_clean_pf_success_max_dirty_gate_recovery_v10_active_reverse_hard95"
PROFILE_V11="legacy_clean_pf_success_max_dirty_gate_recovery_v11_turnspace_reverse_hard95"
PROFILE_V12="legacy_clean_pf_success_max_dirty_gate_recovery_v12_narrow_turn_hard95"
PROFILE_V13="legacy_clean_pf_success_max_dirty_gate_recovery_v13_tipspace_turn_hard95"
PROFILE_V14="legacy_clean_pf_success_max_dirty_gate_recovery_v14_inithard_turn_hard95"
PROFILE_V15="legacy_clean_pf_success_max_dirty_gate_recovery_v15_guarded_turn_hard95"
PROFILE_V16="legacy_clean_pf_success_max_dirty_gate_recovery_v16_actionguard_turnroom_hard95"
PROFILE_V17="legacy_clean_pf_success_max_dirty_gate_recovery_v17_narrow_actionguard_hard95"
PROFILE_V18="legacy_clean_pf_success_max_dirty_gate_recovery_v18_pulse_turnroom_hard95"
PROFILE_V19="legacy_clean_pf_success_max_dirty_gate_recovery_v19_pulse_steer_hard95"
PROFILE_V20="legacy_clean_pf_success_max_dirty_gate_recovery_v20_tipdist_turnroom_hard95"
EVAL_OVERRIDES_DEFAULT=()
EVAL_OVERRIDES_ACTIONGUARD=(
  "env.preinsert_action_guard_enable=true"
  "env.preinsert_action_guard_stateful_enable=true"
  "env.preinsert_action_guard_initial_near_hard_only=false"
  "env.preinsert_action_guard_trigger_dist_m=0.62"
  "env.preinsert_action_guard_release_dist_m=0.82"
  "env.preinsert_action_guard_center_m=0.22"
  "env.preinsert_action_guard_tip_m=0.15"
  "env.preinsert_action_guard_yaw_deg=7.0"
  "env.preinsert_action_guard_insert_frac_max=0.24"
  "env.preinsert_action_guard_max_forward_action=0.0"
  "env.preinsert_action_guard_force_reverse=true"
  "env.preinsert_action_guard_reverse_action=0.18"
  "env.preinsert_action_guard_steer_scale=1.0"
  "env.preinsert_action_guard_min_abs_steer=0.0"
)
EVAL_OVERRIDES_NARROW_ACTIONGUARD=(
  "env.preinsert_action_guard_enable=true"
  "env.preinsert_action_guard_stateful_enable=true"
  "env.preinsert_action_guard_initial_near_hard_only=true"
  "env.preinsert_action_guard_trigger_dist_m=0.42"
  "env.preinsert_action_guard_release_dist_m=0.58"
  "env.preinsert_action_guard_center_m=0.32"
  "env.preinsert_action_guard_tip_m=0.22"
  "env.preinsert_action_guard_yaw_deg=10.0"
  "env.preinsert_action_guard_insert_frac_max=0.18"
  "env.preinsert_action_guard_max_forward_action=0.15"
  "env.preinsert_action_guard_force_reverse=false"
  "env.preinsert_action_guard_reverse_action=0.0"
  "env.preinsert_action_guard_steer_scale=1.0"
  "env.preinsert_action_guard_min_abs_steer=0.0"
)
EVAL_OVERRIDES_PULSE_ACTIONGUARD=(
  "env.preinsert_action_guard_enable=true"
  "env.preinsert_action_guard_stateful_enable=true"
  "env.preinsert_action_guard_initial_near_hard_only=true"
  "env.preinsert_action_guard_trigger_dist_m=0.40"
  "env.preinsert_action_guard_release_dist_m=0.62"
  "env.preinsert_action_guard_center_m=0.32"
  "env.preinsert_action_guard_tip_m=0.22"
  "env.preinsert_action_guard_yaw_deg=10.0"
  "env.preinsert_action_guard_insert_frac_max=0.16"
  "env.preinsert_action_guard_max_forward_action=0.0"
  "env.preinsert_action_guard_force_reverse=true"
  "env.preinsert_action_guard_reverse_action=0.12"
  "env.preinsert_action_guard_steer_scale=1.0"
  "env.preinsert_action_guard_min_abs_steer=0.0"
  "env.preinsert_action_guard_once_per_episode=true"
  "env.preinsert_action_guard_max_steps=20"
)
EVAL_OVERRIDES_PULSE_STEER_ACTIONGUARD=(
  "env.preinsert_action_guard_enable=true"
  "env.preinsert_action_guard_stateful_enable=true"
  "env.preinsert_action_guard_initial_near_hard_only=true"
  "env.preinsert_action_guard_trigger_dist_m=0.40"
  "env.preinsert_action_guard_release_dist_m=0.62"
  "env.preinsert_action_guard_center_m=0.32"
  "env.preinsert_action_guard_tip_m=0.22"
  "env.preinsert_action_guard_yaw_deg=10.0"
  "env.preinsert_action_guard_insert_frac_max=0.16"
  "env.preinsert_action_guard_max_forward_action=0.0"
  "env.preinsert_action_guard_force_reverse=true"
  "env.preinsert_action_guard_reverse_action=0.10"
  "env.preinsert_action_guard_steer_scale=1.0"
  "env.preinsert_action_guard_min_abs_steer=0.0"
  "env.preinsert_action_guard_once_per_episode=true"
  "env.preinsert_action_guard_max_steps=20"
  "env.preinsert_action_guard_steer_to_reduce_error=true"
  "env.preinsert_action_guard_steer_action=0.35"
  "env.preinsert_action_guard_center_steer_weight=1.0"
  "env.preinsert_action_guard_yaw_steer_weight=0.6"
)

if [[ -z "${TERM:-}" || "$TERM" == "dumb" ]]; then
  export TERM=xterm
fi

if [[ -d "$CONDA_ENV_PATH" ]]; then
  export CONDA_PREFIX="$CONDA_ENV_PATH"
  export CONDA_DEFAULT_ENV="$(basename "$CONDA_ENV_PATH")"
  export CONDA_SHLVL=1
  export PATH="$CONDA_ENV_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
fi

mkdir -p "$OUTPUT_ROOT"

echo "[INFO] root: $ROOT"
echo "[INFO] IsaacLab: $ISAACLAB"
echo "[INFO] mode: $MODE"
echo "[INFO] base checkpoint: $BASE_CHECKPOINT"
echo "[INFO] output root: $OUTPUT_ROOT"

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "[FATAL] missing base checkpoint: $BASE_CHECKPOINT" >&2
  exit 2
fi

run_train() {
  local run_tag="$1"
  local max_iterations="$2"
  local clean_profile="$3"
  local resume_run="$4"
  local resume_checkpoint="$5"
  local run_name="$run_tag"

  echo "[INFO] train: run_name=$run_name max_iterations=$max_iterations clean_profile=$clean_profile"
  (
    cd "$ROOT"
    STAGE=insert \
    RESET_PROFILE=full \
    CLEAN_PROFILE="$clean_profile" \
    RUN_NAME="$run_name" \
    MAX_ITERATIONS="$max_iterations" \
    SEED="$SEED" \
    NUM_ENVS="$NUM_ENVS" \
    RESUME_RUN="$resume_run" \
    RESUME_CHECKPOINT="$resume_checkpoint" \
    bash scripts/run_geoedge_staged_train.sh
  )
}

eval_checkpoint() {
  local checkpoint="$1"
  local label="$2"
  local eval_seed="${3:-$SEED}"
  local output_dir="${4:-$OUTPUT_ROOT}"
  shift $(( $# < 4 ? $# : 4 ))
  local eval_overrides=("$@")

  mkdir -p "$output_dir"
  echo "[INFO] eval: label=$label seed=$eval_seed checkpoint=$checkpoint"
  (
    cd "$ISAACLAB"
    "$ISAACLAB/isaaclab.sh" -p "$ROOT/scripts/eval_geoedge_checkpoint.py" \
      --task "$TASK" \
      --headless \
      --stage1_eval \
      --reset_profile full \
      --checkpoint "$checkpoint" \
      --label "$label" \
      --num_envs "$EVAL_NUM_ENVS" \
      --rollouts "$EVAL_ROLLOUTS" \
      --seed "$eval_seed" \
      --output_dir "$output_dir" \
      "${eval_overrides[@]}"
  )

  local episodes_csv="$output_dir/${label}_episodes.csv"
  local bucket_md="$output_dir/${label}_bucket_summary.md"
  python3 "$ROOT/scripts/summarize_geoedge_eval.py" \
    --episodes-csv "$episodes_csv" \
    --output "$bucket_md" >/dev/null
  echo "[INFO] bucket summary: $bucket_md"
}

eval_run_models() {
  local run_name="$1"
  local output_dir="$2"
  shift 2
  local models=("$@")
  local eval_overrides_ref="${EVAL_OVERRIDES_REF:-EVAL_OVERRIDES_DEFAULT[@]}"
  local eval_overrides=("${!eval_overrides_ref}")
  local run_dir
  run_dir="$(run_dir_for "$run_name")"
  if [[ -z "$run_dir" ]]; then
    echo "[WARN] no run directory found for: $run_name" >&2
    return 0
  fi

  for model in "${models[@]}"; do
    local checkpoint="$run_dir/model_${model}.pt"
    if [[ -f "$checkpoint" ]]; then
      eval_checkpoint "$checkpoint" "${run_name}_model${model}_eval$((EVAL_NUM_ENVS * EVAL_ROLLOUTS))" "$SEED" "$output_dir" "${eval_overrides[@]}"
    else
      echo "[WARN] missing checkpoint, skipping eval: $checkpoint" >&2
    fi
  done
}

run_dir_for() {
  local run_name="$1"
  local base_dir="$ISAACLAB/logs/rsl_rl/forklift_pallet_insert_lift_geo_edge"
  if [[ -d "$base_dir/$run_name" ]]; then
    printf '%s\n' "$base_dir/$run_name"
    return 0
  fi
  find "$base_dir" -maxdepth 1 -type d -name "*${run_name}" -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

latest_model_number() {
  local run_name="$1"
  local run_dir
  run_dir="$(run_dir_for "$run_name")"
  if [[ -z "$run_dir" ]]; then
    return 0
  fi
  find "$run_dir" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
    | sed -E 's/^model_([0-9]+)\.pt$/\1/' \
    | sort -n \
    | tail -1
}

eval_latest_model() {
  local run_name="$1"
  local output_dir="$2"
  local model
  model="$(latest_model_number "$run_name")"
  if [[ -z "$model" ]]; then
    echo "[WARN] no checkpoints found for latest eval: $run_name" >&2
    return 0
  fi
  eval_run_models "$run_name" "$output_dir" "$model"
}

write_acceptance_summary() {
  local output_dir="$1"
  local out_md="$output_dir/hard95_eval_summary.md"
  mapfile -t summaries < <(find "$output_dir" -maxdepth 1 -type f -name '*_summary.json' | sort)
  if ((${#summaries[@]} == 0)); then
    echo "[WARN] no eval summaries found in $output_dir" >&2
    return 0
  fi
  python3 "$ROOT/scripts/summarize_geoedge_eval.py" \
    --min-avg-success 0.95 \
    --min-seed-success 0.93 \
    --max-avg-pallet-disp 1.20 \
    --output "$out_md" \
    "${summaries[@]}" >/dev/null
  echo "[INFO] acceptance summary: $out_md"
}

select_best_checkpoint() {
  local output_dir="$1"
  local best_json="$output_dir/best_checkpoint.json"
  local best_md="$output_dir/best_checkpoint.md"
  local best_checkpoint
  best_checkpoint="$(python3 "$ROOT/scripts/select_geoedge_best_checkpoint.py" \
    --eval-dir "$output_dir" \
    --output-json "$best_json" \
    --output-md "$best_md")"
  echo "[INFO] best checkpoint: $best_checkpoint"
  echo "[INFO] best report: $best_md"
}

probe_gate() {
  local output_dir="$1"
  local best_json="$output_dir/best_checkpoint.json"
  if [[ ! -f "$best_json" ]]; then
    echo "[FATAL] missing best checkpoint JSON for probe gate: $best_json" >&2
    return 2
  fi
  python3 - "$best_json" "$PROBE_MIN_STRICT" "$PROBE_MIN_NEAR_STRICT" "$PROBE_MAX_DIRTY" "$PROBE_MAX_TIMEOUT" <<'PY'
import json
import sys

path, min_strict, min_near, max_dirty, max_timeout = sys.argv[1:]
with open(path) as f:
    data = json.load(f)
best = data["ranking"][0] if "ranking" in data else data
strict = float(best["strict_success_rate"])
near = float(best["near_strict_success_rate"])
dirty = float(best["dirty_insert_rate"])
timeout = float(best["timeout_frac"])
passed = (
    strict >= float(min_strict)
    and near >= float(min_near)
    and dirty <= float(max_dirty)
    and timeout <= float(max_timeout)
)
print(
    "[INFO] probe gate: "
    f"strict={strict:.4f}/{float(min_strict):.4f} "
    f"near={near:.4f}/{float(min_near):.4f} "
    f"dirty={dirty:.4f}/{float(max_dirty):.4f} "
    f"timeout={timeout:.4f}/{float(max_timeout):.4f} "
    f"=> {'PASS' if passed else 'FAIL'}"
)
sys.exit(0 if passed else 1)
PY
}

run_smoke() {
  local run_name="geoedge_recovery_v3_hard95_smoke_seed${SEED}_iter2"
  local output_dir="$OUTPUT_ROOT/smoke"
  run_train "$run_name" 2 "$PROFILE_MAIN" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_smoke_v10() {
  local run_name="geoedge_recovery_v10_active_reverse_smoke_seed${SEED}_iter2"
  local output_dir="$OUTPUT_ROOT/smoke_v10"
  run_train "$run_name" 2 "$PROFILE_V10" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_smoke_v11() {
  local run_name="geoedge_recovery_v11_turnspace_reverse_smoke_seed${SEED}_iter2"
  local output_dir="$OUTPUT_ROOT/smoke_v11"
  run_train "$run_name" 2 "$PROFILE_V11" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_smoke_v12() {
  local run_name="geoedge_recovery_v12_narrow_turn_smoke_seed${SEED}_iter2"
  local output_dir="$OUTPUT_ROOT/smoke_v12"
  run_train "$run_name" 2 "$PROFILE_V12" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_smoke_v13() {
  local run_name="geoedge_recovery_v13_tipspace_turn_smoke_seed${SEED}_iter2"
  local output_dir="$OUTPUT_ROOT/smoke_v13"
  run_train "$run_name" 2 "$PROFILE_V13" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_smoke_v14() {
  local run_name="geoedge_recovery_v14_inithard_turn_smoke_seed${SEED}_iter2"
  local output_dir="$OUTPUT_ROOT/smoke_v14"
  run_train "$run_name" 2 "$PROFILE_V14" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_smoke_v15() {
  local run_name="geoedge_recovery_v15_guarded_turn_smoke_seed${SEED}_iter2"
  local output_dir="$OUTPUT_ROOT/smoke_v15"
  run_train "$run_name" 2 "$PROFILE_V15" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_smoke_v16() {
  local run_name="geoedge_recovery_v16_actionguard_turnroom_smoke_seed${SEED}_iter2"
  local output_dir="$OUTPUT_ROOT/smoke_v16"
  run_train "$run_name" 2 "$PROFILE_V16" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_ACTIONGUARD[@]" eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_smoke_v17() {
  local run_name="geoedge_recovery_v17_narrow_actionguard_smoke_seed${SEED}_iter2"
  local output_dir="$OUTPUT_ROOT/smoke_v17"
  run_train "$run_name" 2 "$PROFILE_V17" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_NARROW_ACTIONGUARD[@]" eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_smoke_v18() {
  local run_name="geoedge_recovery_v18_pulse_turnroom_smoke_seed${SEED}_iter2"
  local output_dir="$OUTPUT_ROOT/smoke_v18"
  run_train "$run_name" 2 "$PROFILE_V18" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_PULSE_ACTIONGUARD[@]" eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_smoke_v19() {
  local run_name="geoedge_recovery_v19_pulse_steer_smoke_seed${SEED}_iter2"
  local output_dir="$OUTPUT_ROOT/smoke_v19"
  run_train "$run_name" 2 "$PROFILE_V19" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_PULSE_STEER_ACTIONGUARD[@]" eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_smoke_v20() {
  local run_name="geoedge_recovery_v20_tipdist_turnroom_smoke_seed${SEED}_iter2"
  local output_dir="$OUTPUT_ROOT/smoke_v20"
  run_train "$run_name" 2 "$PROFILE_V20" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe() {
  local run_name="geoedge_recovery_v3_hard95_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe"
  run_train "$run_name" 75 "$PROFILE_MAIN" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe45() {
  local run_name="geoedge_recovery_v3_hard95_frac45_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe45"
  run_train "$run_name" 75 "$PROFILE_FRAC45" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v4() {
  local run_name="geoedge_recovery_v4_turnroom_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v4"
  run_train "$run_name" 75 "$PROFILE_V4" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v5() {
  local run_name="geoedge_recovery_v5_turnroom_shortfall_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v5"
  run_train "$run_name" 75 "$PROFILE_V5" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v6() {
  local run_name="geoedge_recovery_v6_centerdist_gate_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v6"
  run_train "$run_name" 75 "$PROFILE_V6" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v7() {
  local run_name="geoedge_recovery_v7_centerdist_harddirty_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v7"
  run_train "$run_name" 75 "$PROFILE_V7" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v8() {
  local run_name="geoedge_recovery_v8_rootdist_successgate_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v8"
  run_train "$run_name" 75 "$PROFILE_V8" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v9() {
  local run_name="geoedge_recovery_v9_rootdist_reverse_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v9"
  run_train "$run_name" 75 "$PROFILE_V9" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v10() {
  local run_name="geoedge_recovery_v10_active_reverse_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v10"
  run_train "$run_name" 75 "$PROFILE_V10" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v11() {
  local run_name="geoedge_recovery_v11_turnspace_reverse_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v11"
  run_train "$run_name" 75 "$PROFILE_V11" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v12() {
  local run_name="geoedge_recovery_v12_narrow_turn_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v12"
  run_train "$run_name" 75 "$PROFILE_V12" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v13() {
  local run_name="geoedge_recovery_v13_tipspace_turn_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v13"
  run_train "$run_name" 75 "$PROFILE_V13" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v14() {
  local run_name="geoedge_recovery_v14_inithard_turn_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v14"
  run_train "$run_name" 75 "$PROFILE_V14" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v15() {
  local run_name="geoedge_recovery_v15_guarded_turn_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v15"
  run_train "$run_name" 75 "$PROFILE_V15" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v16() {
  local run_name="geoedge_recovery_v16_actionguard_turnroom_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v16"
  run_train "$run_name" 75 "$PROFILE_V16" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_ACTIONGUARD[@]" eval_run_models "$run_name" "$output_dir" 75 100 125 141
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_ACTIONGUARD[@]" eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v17() {
  local run_name="geoedge_recovery_v17_narrow_actionguard_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v17"
  run_train "$run_name" 75 "$PROFILE_V17" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_NARROW_ACTIONGUARD[@]" eval_run_models "$run_name" "$output_dir" 75 100 125 141
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_NARROW_ACTIONGUARD[@]" eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v18() {
  local run_name="geoedge_recovery_v18_pulse_turnroom_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v18"
  run_train "$run_name" 75 "$PROFILE_V18" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_PULSE_ACTIONGUARD[@]" eval_run_models "$run_name" "$output_dir" 75 100 125 141
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_PULSE_ACTIONGUARD[@]" eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v19() {
  local run_name="geoedge_recovery_v19_pulse_steer_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v19"
  run_train "$run_name" 75 "$PROFILE_V19" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_PULSE_STEER_ACTIONGUARD[@]" eval_run_models "$run_name" "$output_dir" 75 100 125 141
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_PULSE_STEER_ACTIONGUARD[@]" eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_probe_v20() {
  local run_name="geoedge_recovery_v20_tipdist_turnroom_probe_seed${SEED}_iter75"
  local output_dir="$OUTPUT_ROOT/probe_v20"
  run_train "$run_name" 75 "$PROFILE_V20" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 141
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main() {
  local run_name="geoedge_recovery_v3_hard95_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main"
  run_train "$run_name" 300 "$PROFILE_MAIN" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v4() {
  local run_name="geoedge_recovery_v4_turnroom_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v4"
  run_train "$run_name" 300 "$PROFILE_V4" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v5() {
  local run_name="geoedge_recovery_v5_turnroom_shortfall_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v5"
  run_train "$run_name" 300 "$PROFILE_V5" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v6() {
  local run_name="geoedge_recovery_v6_centerdist_gate_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v6"
  run_train "$run_name" 300 "$PROFILE_V6" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v7() {
  local run_name="geoedge_recovery_v7_centerdist_harddirty_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v7"
  run_train "$run_name" 300 "$PROFILE_V7" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v8() {
  local run_name="geoedge_recovery_v8_rootdist_successgate_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v8"
  run_train "$run_name" 300 "$PROFILE_V8" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v9() {
  local run_name="geoedge_recovery_v9_rootdist_reverse_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v9"
  run_train "$run_name" 300 "$PROFILE_V9" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v10() {
  local run_name="geoedge_recovery_v10_active_reverse_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v10"
  run_train "$run_name" 300 "$PROFILE_V10" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v11() {
  local run_name="geoedge_recovery_v11_turnspace_reverse_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v11"
  run_train "$run_name" 300 "$PROFILE_V11" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v12() {
  local run_name="geoedge_recovery_v12_narrow_turn_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v12"
  run_train "$run_name" 300 "$PROFILE_V12" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v13() {
  local run_name="geoedge_recovery_v13_tipspace_turn_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v13"
  run_train "$run_name" 300 "$PROFILE_V13" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v14() {
  local run_name="geoedge_recovery_v14_inithard_turn_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v14"
  run_train "$run_name" 300 "$PROFILE_V14" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v15() {
  local run_name="geoedge_recovery_v15_guarded_turn_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v15"
  run_train "$run_name" 300 "$PROFILE_V15" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v16() {
  local run_name="geoedge_recovery_v16_actionguard_turnroom_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v16"
  run_train "$run_name" 300 "$PROFILE_V16" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_ACTIONGUARD[@]" eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_ACTIONGUARD[@]" eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v17() {
  local run_name="geoedge_recovery_v17_narrow_actionguard_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v17"
  run_train "$run_name" 300 "$PROFILE_V17" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_NARROW_ACTIONGUARD[@]" eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_NARROW_ACTIONGUARD[@]" eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v18() {
  local run_name="geoedge_recovery_v18_pulse_turnroom_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v18"
  run_train "$run_name" 300 "$PROFILE_V18" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_PULSE_ACTIONGUARD[@]" eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_PULSE_ACTIONGUARD[@]" eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v19() {
  local run_name="geoedge_recovery_v19_pulse_steer_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v19"
  run_train "$run_name" 300 "$PROFILE_V19" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_PULSE_STEER_ACTIONGUARD[@]" eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  EVAL_OVERRIDES_REF="EVAL_OVERRIDES_PULSE_STEER_ACTIONGUARD[@]" eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_main_v20() {
  local run_name="geoedge_recovery_v20_tipdist_turnroom_main_seed${SEED}_iter300"
  local output_dir="$OUTPUT_ROOT/main_v20"
  run_train "$run_name" 300 "$PROFILE_V20" "$BASE_RESUME_RUN" "$BASE_RESUME_CHECKPOINT"
  eval_run_models "$run_name" "$output_dir" 75 100 125 150 175 200 225 250 275 300 325 350 366
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_confirm() {
  local checkpoint="${BEST_CHECKPOINT:-}"
  if [[ -z "$checkpoint" && -f "$OUTPUT_ROOT/main/best_checkpoint.json" ]]; then
    checkpoint="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$OUTPUT_ROOT/main/best_checkpoint.json")"
  fi
  if [[ -z "$checkpoint" ]]; then
    echo "[FATAL] MODE=confirm requires BEST_CHECKPOINT=/abs/path/model_XXX.pt or $OUTPUT_ROOT/main/best_checkpoint.json" >&2
    exit 2
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "[FATAL] missing BEST_CHECKPOINT: $checkpoint" >&2
    exit 2
  fi

  local output_dir="$OUTPUT_ROOT/confirm"
  local stem
  stem="$(basename "$(dirname "$checkpoint")")_$(basename "$checkpoint" .pt)"
  for eval_seed in 42 43 44; do
    eval_checkpoint "$checkpoint" "${stem}_seed${eval_seed}_eval1024" "$eval_seed" "$output_dir"
  done
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

run_frac45_continuation() {
  local checkpoint="${BEST_CHECKPOINT:-}"
  if [[ -z "$checkpoint" && -f "$OUTPUT_ROOT/main/best_checkpoint.json" ]]; then
    checkpoint="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$OUTPUT_ROOT/main/best_checkpoint.json")"
  fi
  if [[ -z "$checkpoint" ]]; then
    echo "[FATAL] MODE=frac45 requires BEST_CHECKPOINT=/abs/path/model_XXX.pt or $OUTPUT_ROOT/main/best_checkpoint.json" >&2
    exit 2
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "[FATAL] missing BEST_CHECKPOINT: $checkpoint" >&2
    exit 2
  fi

  local resume_run resume_checkpoint run_name output_dir
  resume_run="$(basename "$(dirname "$checkpoint")")"
  resume_checkpoint="$(basename "$checkpoint")"
  run_name="geoedge_recovery_v3_hard95_frac45_seed${SEED}_iter200_from_${resume_checkpoint%.pt}"
  output_dir="$OUTPUT_ROOT/frac45"
  run_train "$run_name" 200 "$PROFILE_FRAC45" "$resume_run" "$resume_checkpoint"
  eval_run_models "$run_name" "$output_dir" 25 50 75 100 125 150 175 200
  eval_latest_model "$run_name" "$output_dir"
  write_acceptance_summary "$output_dir"
  select_best_checkpoint "$output_dir"
}

case "$MODE" in
  smoke)
    run_smoke
    ;;
  smoke_v10)
    run_smoke_v10
    ;;
  smoke_v11)
    run_smoke_v11
    ;;
  smoke_v12)
    run_smoke_v12
    ;;
  smoke_v13)
    run_smoke_v13
    ;;
  smoke_v14)
    run_smoke_v14
    ;;
  smoke_v15)
    run_smoke_v15
    ;;
  smoke_v16)
    run_smoke_v16
    ;;
  smoke_v17)
    run_smoke_v17
    ;;
  smoke_v18)
    run_smoke_v18
    ;;
  smoke_v19)
    run_smoke_v19
    ;;
  smoke_v20)
    run_smoke_v20
    ;;
  probe)
    run_probe
    ;;
  probe45)
    run_probe45
    ;;
  probe_v4)
    run_probe_v4
    ;;
  probe_v5)
    run_probe_v5
    ;;
  probe_v6)
    run_probe_v6
    ;;
  probe_v7)
    run_probe_v7
    ;;
  probe_v8)
    run_probe_v8
    ;;
  probe_v9)
    run_probe_v9
    ;;
  probe_v10)
    run_probe_v10
    ;;
  probe_v11)
    run_probe_v11
    ;;
  probe_v12)
    run_probe_v12
    ;;
  probe_v13)
    run_probe_v13
    ;;
  probe_v14)
    run_probe_v14
    ;;
  probe_v15)
    run_probe_v15
    ;;
  probe_v16)
    run_probe_v16
    ;;
  probe_v17)
    run_probe_v17
    ;;
  probe_v18)
    run_probe_v18
    ;;
  probe_v19)
    run_probe_v19
    ;;
  probe_v20)
    run_probe_v20
    ;;
  main)
    run_main
    ;;
  main_v4)
    run_main_v4
    ;;
  main_v5)
    run_main_v5
    ;;
  main_v6)
    run_main_v6
    ;;
  main_v7)
    run_main_v7
    ;;
  main_v8)
    run_main_v8
    ;;
  main_v9)
    run_main_v9
    ;;
  main_v10)
    run_main_v10
    ;;
  main_v11)
    run_main_v11
    ;;
  main_v12)
    run_main_v12
    ;;
  main_v13)
    run_main_v13
    ;;
  main_v14)
    run_main_v14
    ;;
  main_v15)
    run_main_v15
    ;;
  main_v16)
    run_main_v16
    ;;
  main_v17)
    run_main_v17
    ;;
  main_v18)
    run_main_v18
    ;;
  main_v19)
    run_main_v19
    ;;
  main_v20)
    run_main_v20
    ;;
  confirm)
    run_confirm
    ;;
  frac45)
    run_frac45_continuation
    ;;
  all)
    run_smoke
    run_probe
    probe_gate "$OUTPUT_ROOT/probe" || {
      echo "[FATAL] probe gate failed; not starting main. Try MODE=probe45 or adjust profile before a 300-iter run." >&2
      exit 1
    }
    run_main
    ;;
  *)
    echo "[FATAL] MODE must be smoke, smoke_v10, smoke_v11, smoke_v12, smoke_v13, smoke_v14, smoke_v15, smoke_v16, smoke_v17, smoke_v18, smoke_v19, probe, probe45, probe_v4, probe_v5, probe_v6, probe_v7, probe_v8, probe_v9, probe_v10, probe_v11, probe_v12, probe_v13, probe_v14, probe_v15, probe_v16, probe_v17, probe_v18, probe_v19, main, main_v4, main_v5, main_v6, main_v7, main_v8, main_v9, main_v10, main_v11, main_v12, main_v13, main_v14, main_v15, main_v16, main_v17, main_v18, main_v19, confirm, frac45, or all; got: $MODE" >&2
    exit 2
    ;;
esac
