#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
euroc_root="$repo_root/datasets/euroc"
staging_dir="$euroc_root/.unpack_staging"

mkdir -p "$staging_dir"

extract_grouped_archive() {
  local archive_path="$1"
  local group_name="$2"
  local group_dir="$staging_dir/$group_name"

  rm -rf "$group_dir"
  mkdir -p "$group_dir"
  unzip -oq "$archive_path" -d "$group_dir"
}

extract_sequence_zip() {
  local sequence_name="$1"
  local nested_zip="$2"
  local target_dir="$euroc_root/$sequence_name"

  if [[ -d "$target_dir/mav0" ]]; then
    echo "skip existing $sequence_name"
    return
  fi

  rm -rf "$target_dir"
  mkdir -p "$target_dir"
  unzip -oq "$nested_zip" -d "$target_dir"
  find "$target_dir" -name '__MACOSX' -type d -prune -exec rm -rf {} +
  find "$target_dir" \( -name '.DS_Store' -o -name '._*' \) -type f -delete
  echo "extracted $sequence_name"
}

extract_grouped_archive "$euroc_root/machine_hall.zip" "machine_hall"
extract_grouped_archive "$euroc_root/vicon_room1.zip" "vicon_room1"
extract_grouped_archive "$euroc_root/vicon_room2.zip" "vicon_room2"

while IFS= read -r nested_zip; do
  sequence_name="$(basename "$(dirname "$nested_zip")")"
  extract_sequence_zip "$sequence_name" "$nested_zip"
done < <(find "$staging_dir" -type f -name '*.zip' | sort)

find "$euroc_root" -maxdepth 1 -type d | sort
