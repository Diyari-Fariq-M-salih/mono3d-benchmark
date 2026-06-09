#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_name="${1:-depth_anything_3}"
upstream_dir="$repo_root/methods/depth_anything_3/upstream"

if [[ ! -d "$upstream_dir" ]]; then
  echo "missing upstream repo: $upstream_dir" >&2
  echo "run: python3 scripts/clone_method.py depth_anything_3" >&2
  exit 1
fi

python3 "$repo_root/scripts/bootstrap_venv.py" "$venv_name"

venv_python="$repo_root/environment/venv/.venvs/$venv_name/bin/python"
venv_pip="$repo_root/environment/venv/.venvs/$venv_name/bin/pip"

"$venv_pip" install -r "$upstream_dir/requirements.txt"
"$venv_pip" install -r "$upstream_dir/da3_streaming/requirements.txt"
"$venv_pip" install -e "$upstream_dir"

echo "DA3 environment ready: environment/venv/.venvs/$venv_name"
echo "activate with: source environment/venv/.venvs/$venv_name/bin/activate"
