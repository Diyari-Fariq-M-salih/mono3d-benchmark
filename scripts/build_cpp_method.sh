#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {svo}" >&2
  exit 1
fi

method="$1"

case "$method" in
  svo)
    docker build -f environment/docker/svo.Dockerfile -t mono3d-benchmark/svo:noetic .
    ;;
  *)
    echo "unknown method: $method" >&2
    exit 1
    ;;
esac
