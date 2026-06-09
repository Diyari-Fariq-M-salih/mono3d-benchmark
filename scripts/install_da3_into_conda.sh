#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="$repo_root/methods/depth_anything_3/upstream"
conda_env="${1:-da3stream}"

if [[ ! -d "$upstream_dir" ]]; then
  echo "missing upstream repo: $upstream_dir" >&2
  exit 1
fi

conda run -n "$conda_env" pip install -e "$upstream_dir"
echo "installed Depth Anything 3 into conda env: $conda_env"
