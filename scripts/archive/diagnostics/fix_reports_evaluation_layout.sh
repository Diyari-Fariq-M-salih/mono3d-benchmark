#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$REPO_ROOT/reports/evaluation"
ARCHIVE_TUM_DIR="$EVAL_DIR/archive/tum"

NONCANONICAL_TUM_DIRS=(
  "svo_tum_mono_20260610_075102_mono3d_tum_mono"
  "svo_tum_mono_20260610_080001_mono3d_tum_mono"
  "svo_tum_mono_20260610_080434_mono3d_tum_mono"
  "svo_tum_ceres_mono3d_tum_mono_ceres_20260610_091912_mono3d_tum_mono_ceres"
  "svo_tum_loop_mono3d_tum_mono_loop_20260610_092332_mono3d_tum_mono_loop"
  "svo_tum_conservative_mono3d_tum_mono_conservative_20260610_092441_mono3d_tum_mono_conservative"
  "svo_tum_conservative_loop_mono3d_tum_mono_conservative_loop_20260610_092550_mono3d_tum_mono_conservative_loop"
  "tum_svo_vs_da3_ceres_20260610_091912_mono3d_tum_mono_ceres"
  "tum_svo_vs_da3_loop_20260610_092332_mono3d_tum_mono_loop"
  "tum_svo_vs_da3_conservative_20260610_092441_mono3d_tum_mono_conservative"
  "tum_svo_vs_da3_conservative_loop_20260610_092550_mono3d_tum_mono_conservative_loop"
)

CANONICAL_TUM_DIRS=(
  "svo_tum_baseline_mono3d_tum_mono_20260610_091041_mono3d_tum_mono"
  "tum_svo_vs_da3_baseline_20260610_091041_mono3d_tum_mono"
  "tum_svo_variants_summary_20260610_091759"
)

mkdir -p "$ARCHIVE_TUM_DIR"

if [[ "${1:-}" == "--print-only" ]]; then
  echo "sudo chown -R qcar:qcar \"$EVAL_DIR\""
  for dir in "${NONCANONICAL_TUM_DIRS[@]}"; do
    echo "mv \"$EVAL_DIR/$dir\" \"$ARCHIVE_TUM_DIR/\""
  done
  exit 0
fi

echo "[info] Evaluation dir: $EVAL_DIR"
echo "[info] Archiving non-canonical TUM runs into: $ARCHIVE_TUM_DIR"

for dir in "${CANONICAL_TUM_DIRS[@]}"; do
  if [[ -d "$EVAL_DIR/$dir" ]]; then
    echo "[keep] $dir"
  fi
done

for dir in "${NONCANONICAL_TUM_DIRS[@]}"; do
  src="$EVAL_DIR/$dir"
  if [[ ! -e "$src" ]]; then
    echo "[skip] Missing $dir"
    continue
  fi
  echo "[move] $dir"
  mv "$src" "$ARCHIVE_TUM_DIR/"
done

echo "[done] Evaluation layout cleaned."
