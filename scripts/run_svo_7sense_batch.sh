#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="$repo_root/outputs/logs/svo_batch"
mkdir -p "$log_dir"
run_log="$log_dir/${timestamp}_7sense.log"
svo_report_script="$repo_root/evaluation/odometry/svo_sanity_report.py"

scenes=()
sequences=()
split="test"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)
      scenes+=("$2")
      shift 2
      ;;
    --sequence)
      sequences+=("$2")
      shift 2
      ;;
    --split)
      split="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 [--scene NAME] [--sequence scene/seq-01] [--split train|test|all]" >&2
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

prepare_cmd=("python3" "$repo_root/scripts/prepare_7sense_svo.py" "--write-configs" "--split" "$split")
for scene in "${scenes[@]}"; do
  prepare_cmd+=("--scene" "$scene")
done
for sequence in "${sequences[@]}"; do
  prepare_cmd+=("--sequence" "$sequence")
done
prepare_json="$log_dir/${timestamp}_7sense_prepare.json"
"${prepare_cmd[@]}" | tee "$prepare_json"

roscore_started=0
if ! timeout 2 rosparam list >/dev/null 2>&1; then
  roscore >"$log_dir/${timestamp}_7sense_roscore.log" 2>&1 &
  roscore_pid=$!
  roscore_started=1
  trap 'if [[ ${roscore_started:-0} -eq 1 ]]; then kill ${roscore_pid:-0} >/dev/null 2>&1 || true; fi' EXIT
  sleep 3
fi

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
    return 0
  fi
  while IFS= read -r seq_dir; do
    local report_path="$seq_dir/sanity_report.json"
    echo "Sanity report: ${seq_dir#$repo_root/}"
    PYTHONDONTWRITEBYTECODE=1 python3 "$svo_report_script" "$seq_dir" --json-out "$report_path"
  done < <(find "$latest_run" -mindepth 1 -maxdepth 1 -type d | sort)
}

{
  echo "Running $timestamp"
  echo "Experiment: mono3d_7sense_mono"
  python3 "$svo_benchmark_script" mono3d_7sense_mono.yaml || true
  emit_sanity_reports "mono3d_7sense_mono" || true
} 2>&1 | tee "$run_log"
