#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
sudo docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -w /workspace -v "$repo_root":/workspace mono3d-benchmark/svo:noetic bash
