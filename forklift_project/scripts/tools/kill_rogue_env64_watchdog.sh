#!/usr/bin/env bash
set -u

log_path="${1:-/tmp/kill_rogue_env64_watchdog.log}"
target_envs="${2:-64}"

while true; do
  pgrep -f "tr[a]in.py.*--num_envs ${target_envs}" | while read -r pid; do
    [ -z "$pid" ] && continue
    printf '%s killing rogue env%s pid=%s\n' "$(date '+%F %T')" "${target_envs}" "$pid" >> "$log_path"
    kill "$pid" 2>/dev/null || true
  done
  sleep 3
done
