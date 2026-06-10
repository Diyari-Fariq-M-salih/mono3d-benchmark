#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="$repo_root/outputs/logs/svo_batch"
mkdir -p "$log_dir"
run_log="$log_dir/${timestamp}_tum.log"
svo_report_script="$repo_root/evaluation/odometry/svo_sanity_report.py"
tum_report_script="$repo_root/scripts/plot_svo_tum_sanity.py"

sequences=()
calib_mode="official-rgb"
variants=()
all_variants=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sequence)
      sequences+=("$2")
      shift 2
      ;;
    --calib-mode)
      calib_mode="$2"
      shift 2
      ;;
    --variant)
      variants+=("$2")
      shift 2
      ;;
    --all-variants)
      all_variants=1
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 [--sequence NAME] [--calib-mode official-rgb|ros-default] [--variant NAME] [--all-variants]" >&2
      exit 1
      ;;
  esac
done

cd "$repo_root"

export ROS_DISTRO="${ROS_DISTRO:-noetic}"
set +u
source /opt/ros/noetic/setup.bash
source "$repo_root/environment/ros_ws/svo/devel/setup.bash"
set -u

svo_benchmark_script="$repo_root/methods/svo/upstream/svo_benchmarking/scripts/benchmark.py"

latest_run_dir() {
  local experiment_name="$1"
  local trace_base="$repo_root/outputs/logs/svo_benchmarks/$experiment_name"
  find "$trace_base" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

emit_sanity_reports() {
  local experiment_name="$1"
  local latest_run
  latest_run="$(latest_run_dir "$experiment_name")"
  if [[ -z "${latest_run:-}" ]]; then
    echo "No trace directory found for $experiment_name" >&2
    return 0
  fi

  while IFS= read -r seq_dir; do
    local report_path="$seq_dir/sanity_report.json"
    echo "Sanity report: ${seq_dir#$repo_root/}"
    PYTHONDONTWRITEBYTECODE=1 python3 "$svo_report_script" "$seq_dir" --json-out "$report_path"
  done < <(find "$latest_run" -mindepth 1 -maxdepth 1 -type d | sort)
}

latest_report_dir() {
  local prefix="$1"
  local root="$repo_root/reports/evaluation"
  find "$root" -mindepth 1 -maxdepth 1 -type d -name "${prefix}_*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

prepare_cmd=("python3" "$repo_root/scripts/prepare_svo_tum.py" "--write-configs" "--calib-mode" "$calib_mode")
for seq in "${sequences[@]}"; do
  prepare_cmd+=("--sequence" "$seq")
done
if [[ $all_variants -eq 1 ]]; then
  prepare_cmd+=("--all-variants")
fi
for variant in "${variants[@]}"; do
  prepare_cmd+=("--variant" "$variant")
done
prepare_json="$log_dir/${timestamp}_tum_prepare.json"
"${prepare_cmd[@]}" | tee "$prepare_json"

roscore_started=0
if ! timeout 2 rosparam list >/dev/null 2>&1; then
  roscore >"$log_dir/${timestamp}_tum_roscore.log" 2>&1 &
  roscore_pid=$!
  roscore_started=1
  trap 'if [[ ${roscore_started:-0} -eq 1 ]]; then kill ${roscore_pid:-0} >/dev/null 2>&1 || true; fi' EXIT
  sleep 3
fi

{
  echo "Running $timestamp"
  mapfile -t experiment_specs < <(
    python3 - "$prepare_json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
meta = payload.get("generated_experiment_meta", [])
if meta:
    for item in meta:
        print(f"{item['variant']}|{item['experiment_name']}|{Path(item['path']).name}")
else:
    for path in payload.get("generated_experiments", []):
        p = Path(path)
        stem = p.stem
        print(f"{stem}|{stem}|{p.name}")
PY
  )

  for spec in "${experiment_specs[@]}"; do
    IFS='|' read -r variant_name experiment_name yaml_name <<<"$spec"
    echo "Experiment: $experiment_name ($variant_name)"
    benchmark_rc=0
    python3 "$svo_benchmark_script" "$yaml_name" || benchmark_rc=$?
    echo "Benchmark exit code for $experiment_name: $benchmark_rc"

    latest_run="$(latest_run_dir "$experiment_name")"
    if [[ -n "${latest_run:-}" ]]; then
      emit_sanity_reports "$experiment_name" || true
      report_output="$repo_root/reports/evaluation/svo_tum_${variant_name}_${experiment_name}_$(basename "$latest_run")"
      python3 "$tum_report_script" --mono-dir "$latest_run" --output-dir "$report_output" || true
      compare_output="$repo_root/reports/evaluation/tum_svo_vs_da3_${variant_name}_$(basename "$latest_run")"
      python3 "$repo_root/scripts/plot_tum_svo_vs_da3.py" --svo-dir "$latest_run" --output-dir "$compare_output" || true
    else
      echo "No trace directory found for $experiment_name after benchmark exit code $benchmark_rc"
    fi
  done
} 2>&1 | tee "$run_log"
