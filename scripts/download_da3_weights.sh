#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="$repo_root/methods/depth_anything_3/upstream"

if [[ ! -d "$upstream_dir/da3_streaming" ]]; then
  echo "missing DA3 streaming directory" >&2
  exit 1
fi

cd "$upstream_dir/da3_streaming"
bash ./scripts/download_weights.sh

echo "weights downloaded under: $upstream_dir/da3_streaming/weights"
