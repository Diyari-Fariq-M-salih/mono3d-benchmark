#!/usr/bin/env python3
"""Prepare EuRoC dataset metadata for benchmark wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def count_files(directory: Path, suffix: str | None = None) -> int:
    files = [path for path in directory.iterdir() if path.is_file()]
    if suffix is None:
        return len(files)
    return sum(1 for path in files if path.suffix.lower() == suffix.lower())


GROUNDTRUTH_CANDIDATE_NAMES = (
    "state_groundtruth_estimate0",
    "vicon0",
    "leica0",
    "pointcloud0",
)


def relative_or_none(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(root))


def find_groundtruth_dirs(mav0_dir: Path) -> list[Path]:
    return [mav0_dir / name for name in GROUNDTRUTH_CANDIDATE_NAMES if (mav0_dir / name).exists()]


def validate_sequence_layout(sequence_dir: Path) -> dict[str, Any]:
    mav0_dir = sequence_dir / "mav0"
    required_paths = {
        "cam0_csv": mav0_dir / "cam0" / "data.csv",
        "cam0_images": mav0_dir / "cam0" / "data",
        "cam1_csv": mav0_dir / "cam1" / "data.csv",
        "cam1_images": mav0_dir / "cam1" / "data",
        "imu0_csv": mav0_dir / "imu0" / "data.csv",
    }

    missing = [name for name, path in required_paths.items() if not path.exists()]
    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence directory does not exist: {sequence_dir}")
    if not mav0_dir.exists():
        raise FileNotFoundError(f"Missing mav0 directory under: {sequence_dir}")
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing required EuRoC paths: {missing_text}")

    groundtruth_dirs = find_groundtruth_dirs(mav0_dir)
    preferred_groundtruth_dir = groundtruth_dirs[0] if groundtruth_dirs else None
    cam0_images = count_files(required_paths["cam0_images"], ".png")
    cam1_images = count_files(required_paths["cam1_images"], ".png")
    cam0_rows = count_csv_rows(required_paths["cam0_csv"])
    cam1_rows = count_csv_rows(required_paths["cam1_csv"])
    imu_rows = count_csv_rows(required_paths["imu0_csv"])
    warnings: list[str] = []

    if cam0_images != cam1_images:
        warnings.append(
            f"Stereo image count mismatch: cam0 has {cam0_images} images, cam1 has {cam1_images}."
        )
    if cam0_images != cam0_rows:
        warnings.append(
            f"cam0 image/file count mismatch: directory has {cam0_images} PNGs, CSV has {cam0_rows} rows."
        )
    if cam1_images != cam1_rows:
        warnings.append(
            f"cam1 image/file count mismatch: directory has {cam1_images} PNGs, CSV has {cam1_rows} rows."
        )

    metadata: dict[str, Any] = {
        "sequence_name": sequence_dir.name,
        "sequence_dir": str(sequence_dir.resolve()),
        "layout_status": "ok",
        "warnings": warnings,
        "paths": {
            "mav0_dir": str(mav0_dir.resolve()),
            "cam0_csv": str(required_paths["cam0_csv"].resolve()),
            "cam0_images_dir": str(required_paths["cam0_images"].resolve()),
            "cam1_csv": str(required_paths["cam1_csv"].resolve()),
            "cam1_images_dir": str(required_paths["cam1_images"].resolve()),
            "imu0_csv": str(required_paths["imu0_csv"].resolve()),
            "groundtruth_dir": str(preferred_groundtruth_dir.resolve()) if preferred_groundtruth_dir else None,
        },
        "counts": {
            "cam0_images": cam0_images,
            "cam1_images": cam1_images,
            "cam0_csv_rows": cam0_rows,
            "cam1_csv_rows": cam1_rows,
            "imu0_csv_rows": imu_rows,
        },
        "available_modalities": sorted(
            child.name
            for child in mav0_dir.iterdir()
            if child.is_dir() and not child.name.startswith("__")
        ),
        "optional_groundtruth": {
            "preferred_kind": preferred_groundtruth_dir.name if preferred_groundtruth_dir else None,
            "preferred_relative_path": relative_or_none(preferred_groundtruth_dir, sequence_dir),
            "available": [
                {
                    "kind": path.name,
                    "relative_path": relative_or_none(path, sequence_dir),
                }
                for path in groundtruth_dirs
            ],
        },
    }
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EuRoC sequence metadata.")
    parser.add_argument("sequence_dir", type=Path, help="Path to a EuRoC sequence")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = validate_sequence_layout(args.sequence_dir.resolve())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
