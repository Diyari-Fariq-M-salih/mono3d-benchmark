#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sequence="V1_01_easy"
mode="mono-imu"
tag="isolated_v1_01"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sequence)
      sequence="$2"
      shift 2
      ;;
    --mode)
      mode="$2"
      shift 2
      ;;
    --tag)
      tag="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 [--sequence SEQ] [--mode mono|mono-imu] [--tag NAME]" >&2
      exit 1
      ;;
  esac
done

case "$mode" in
  mono)
    exp_name="mono3d_euroc_${tag}_mono"
    ;;
  mono-imu)
    exp_name="mono3d_euroc_${tag}_mono_imu"
    ;;
  *)
    echo "unsupported mode: $mode" >&2
    exit 1
    ;;
esac

trace_root="$repo_root/outputs/logs/svo_benchmarks_isolated"
log_root="$repo_root/outputs/logs/svo_batch_isolated"
mkdir -p "$trace_root" "$log_root"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_log="$log_root/${timestamp}_${sequence}_${mode}.log"

cd "$repo_root"

export ROS_DISTRO="${ROS_DISTRO:-noetic}"
set +u
source /opt/ros/noetic/setup.bash
source "$repo_root/environment/ros_ws/svo/devel/setup.bash"
set -u

python3 "$repo_root/scripts/prepare_svo_euroc.py" \
  --sequence "$sequence" \
  --write-configs \
  --mono-exp-name "mono3d_euroc_${tag}_mono" \
  --mono-imu-exp-name "mono3d_euroc_${tag}_mono_imu" \
  --trace-root "$trace_root"

svo_benchmark_script="$repo_root/methods/svo/upstream/svo_benchmarking/scripts/benchmark.py"

roscore_started=0
if ! timeout 2 rosparam list >/dev/null 2>&1; then
  roscore >"$log_root/${timestamp}_roscore.log" 2>&1 &
  roscore_pid=$!
  roscore_started=1
  trap 'if [[ ${roscore_started:-0} -eq 1 ]]; then kill ${roscore_pid:-0} >/dev/null 2>&1 || true; fi' EXIT
  sleep 3
fi

{
  echo "Running isolated SVO at $timestamp"
  echo "Sequence: $sequence"
  echo "Mode: $mode"
  echo "Experiment: $exp_name"
  python3 "$svo_benchmark_script" "${exp_name}.yaml"
} 2>&1 | tee "$run_log"

latest_run_dir="$(find "$trace_root/$exp_name" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -n "${latest_run_dir:-}" ]]; then
  echo
  echo "Latest isolated trace:"
  echo "$latest_run_dir/$sequence"
fi
