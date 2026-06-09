#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="$repo_root/methods/depth_anything_3/upstream"
venv_name="${DA3_VENV_NAME:-depth_anything_3}"
venv_python="$repo_root/environment/venv/.venvs/$venv_name/bin/python"
conda_env="${DA3_CONDA_ENV:-}"

runner=()

if [[ -n "$conda_env" ]]; then
  runner=(conda run -n "$conda_env" python)
elif [[ -x "$venv_python" ]]; then
  runner=("$venv_python")
else
  echo "missing DA3 runtime" >&2
  echo "either:" >&2
  echo "  - run: bash scripts/setup_da3.sh" >&2
  echo "  - or set DA3_CONDA_ENV=da3stream and install with: bash scripts/install_da3_into_conda.sh da3stream" >&2
  exit 1
fi

image_dir=""
config_path="$upstream_dir/da3_streaming/configs/base_config.yaml"
output_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-dir)
      image_dir="$2"
      shift 2
      ;;
    --config)
      config_path="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 --image-dir DIR [--config FILE] [--output-dir DIR]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$image_dir" ]]; then
  echo "missing --image-dir" >&2
  exit 1
fi

if [[ -z "$output_dir" ]]; then
  run_name="$(basename "$image_dir")"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  output_dir="$repo_root/outputs/reconstructions/depth_anything_3/${run_name}/${timestamp}_da3_streaming"
fi

mkdir -p "$output_dir"

cd "$upstream_dir/da3_streaming"
PYTHONPATH="$upstream_dir/src:${PYTHONPATH:-}" "${runner[@]}" da3_streaming.py \
  --image_dir "$image_dir" \
  --config "$config_path" \
  --output_dir "$output_dir"

echo "DA3-Streaming output: $output_dir"
