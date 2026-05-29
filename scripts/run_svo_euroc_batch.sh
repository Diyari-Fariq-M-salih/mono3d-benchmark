#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="$repo_root/outputs/logs/svo_batch"
mkdir -p "$log_dir"
run_log="$log_dir/${timestamp}.log"
svo_report_script="$repo_root/evaluation/odometry/svo_sanity_report.py"

mode="both"
sequences=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="$2"
      shift 2
      ;;
    --sequence)
      sequences+=("$2")
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 [--mode mono|mono-imu|both] [--sequence SEQ]" >&2
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

emit_sanity_reports() {
  local experiment_name="$1"
  local trace_base="$repo_root/outputs/logs/svo_benchmarks/$experiment_name"
  local latest_run_dir
  latest_run_dir="$(find "$trace_base" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${latest_run_dir:-}" ]]; then
    echo "No trace directory found for $experiment_name" >&2
    return 0
  fi

  local target_dirs=()
  if [[ ${#sequences[@]} -gt 0 ]]; then
    local seq
    for seq in "${sequences[@]}"; do
      if [[ -d "$latest_run_dir/$seq" ]]; then
        target_dirs+=("$latest_run_dir/$seq")
      fi
    done
  else
    while IFS= read -r seq_dir; do
      target_dirs+=("$seq_dir")
    done < <(find "$latest_run_dir" -mindepth 1 -maxdepth 1 -type d | sort)
  fi

  local trace_dir
  for trace_dir in "${target_dirs[@]}"; do
    local report_path="$trace_dir/sanity_report.json"
    echo "Sanity report: ${trace_dir#$repo_root/}"
    PYTHONDONTWRITEBYTECODE=1 python3 "$svo_report_script" "$trace_dir" --json-out "$report_path"
  done
}

prepare_cmd=("python3" "$repo_root/scripts/prepare_svo_euroc.py" "--write-configs")
for seq in "${sequences[@]}"; do
  prepare_cmd+=("--sequence" "$seq")
done
"${prepare_cmd[@]}" | tee "$log_dir/${timestamp}_prepare.json"

roscore_started=0
if ! timeout 2 rosparam list >/dev/null 2>&1; then
  roscore >"$log_dir/${timestamp}_roscore.log" 2>&1 &
  roscore_pid=$!
  roscore_started=1
  trap 'if [[ ${roscore_started:-0} -eq 1 ]]; then kill ${roscore_pid:-0} >/dev/null 2>&1 || true; fi' EXIT
  sleep 3
fi

{
  echo "Running $timestamp"
  case "$mode" in
    mono)
      echo "Experiment: mono"
      python3 "$svo_benchmark_script" mono3d_euroc_mono.yaml
      emit_sanity_reports "mono3d_euroc_mono"
      ;;
    mono-imu)
      echo "Experiment: mono+imu"
      python3 "$svo_benchmark_script" mono3d_euroc_mono_imu.yaml
      emit_sanity_reports "mono3d_euroc_mono_imu"
      ;;
    both)
      echo "Experiment: mono"
      python3 "$svo_benchmark_script" mono3d_euroc_mono.yaml
      emit_sanity_reports "mono3d_euroc_mono"
      echo "Experiment: mono+imu"
      python3 "$svo_benchmark_script" mono3d_euroc_mono_imu.yaml
      emit_sanity_reports "mono3d_euroc_mono_imu"
      ;;
    *)
      echo "unsupported mode: $mode" >&2
      exit 1
      ;;
  esac
} 2>&1 | tee "$run_log"
