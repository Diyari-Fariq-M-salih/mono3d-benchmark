#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/datasets/eth3d/high_res_multi_view"

mkdir -p "${DATA_DIR}/courtyard" "${DATA_DIR}/delivery_area" "${DATA_DIR}/electro"

download() {
  local url="$1"
  local out="$2"
  echo "Downloading ${url}"
  wget -c -O "${out}" "${url}"
}

download "https://www.eth3d.net/data/courtyard_dslr_jpg.7z" \
  "${DATA_DIR}/courtyard/courtyard_dslr_jpg.7z"
download "https://www.eth3d.net/data/courtyard_dslr_scan_eval.7z" \
  "${DATA_DIR}/courtyard/courtyard_dslr_scan_eval.7z"

download "https://www.eth3d.net/data/delivery_area_dslr_jpg.7z" \
  "${DATA_DIR}/delivery_area/delivery_area_dslr_jpg.7z"
download "https://www.eth3d.net/data/delivery_area_dslr_scan_eval.7z" \
  "${DATA_DIR}/delivery_area/delivery_area_dslr_scan_eval.7z"

download "https://www.eth3d.net/data/electro_dslr_jpg.7z" \
  "${DATA_DIR}/electro/electro_dslr_jpg.7z"
download "https://www.eth3d.net/data/electro_dslr_scan_eval.7z" \
  "${DATA_DIR}/electro/electro_dslr_scan_eval.7z"

if command -v 7z >/dev/null 2>&1; then
  echo "Extracting ETH3D archives with 7z"
  (
    cd "${DATA_DIR}/courtyard"
    7z x -y courtyard_dslr_jpg.7z
    7z x -y courtyard_dslr_scan_eval.7z
  )
  (
    cd "${DATA_DIR}/delivery_area"
    7z x -y delivery_area_dslr_jpg.7z
    7z x -y delivery_area_dslr_scan_eval.7z
  )
  (
    cd "${DATA_DIR}/electro"
    7z x -y electro_dslr_jpg.7z
    7z x -y electro_dslr_scan_eval.7z
  )
else
  echo "7z is not installed; archives were downloaded but not extracted."
fi

echo "ETH3D first-pass download complete."
