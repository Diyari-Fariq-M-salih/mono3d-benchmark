#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
sudo docker run --rm -it -v "$repo_root":/workspace mono3d-benchmark/svo:noetic bash
