#!/usr/bin/env python3
"""Prepare ETH3D dataset metadata for benchmark wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def count_files(directory: Path, suffix: str | None = None) -> int:
    files = [path for path in directory.iterdir() if path.is_file()]
    if suffix is None:
        return len(files)
    return sum(1 for path in files if path.suffix.lower() == suffix.lower())


def validate_scene_layout(scene_dir: Path) -> dict[str, Any]:
    required_paths = {
        "images_dir": scene_dir / "images" / "dslr_images",
        "cameras_txt": scene_dir / "dslr_calibration_jpg" / "cameras.txt",
        "images_txt": scene_dir / "dslr_calibration_jpg" / "images.txt",
        "points3d_txt": scene_dir / "dslr_calibration_jpg" / "points3D.txt",
        "scan_eval_dir": scene_dir / "dslr_scan_eval",
    }

    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")

    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing required ETH3D paths: {missing_text}")

    scan_eval_dir = required_paths["scan_eval_dir"]
    scan_plys = sorted(path.name for path in scan_eval_dir.glob("*.ply"))

    metadata: dict[str, Any] = {
        "scene_name": scene_dir.name,
        "scene_dir": str(scene_dir.resolve()),
        "layout_status": "ok",
        "paths": {
            "images_dir": str(required_paths["images_dir"].resolve()),
            "dslr_calibration_dir": str((scene_dir / "dslr_calibration_jpg").resolve()),
            "scan_eval_dir": str(scan_eval_dir.resolve()),
        },
        "counts": {
            "images": count_files(required_paths["images_dir"], ".jpg"),
            "scan_eval_ply_files": len(scan_plys),
            "scan_eval_total_files": count_files(scan_eval_dir),
        },
        "calibration_files": [
            "cameras.txt",
            "images.txt",
            "points3D.txt",
        ],
        "scan_eval_files": sorted(path.name for path in scan_eval_dir.iterdir() if path.is_file()),
        "scan_eval_ply_files": scan_plys,
    }
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ETH3D scene metadata.")
    parser.add_argument("scene_dir", type=Path, help="Path to an ETH3D scene")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = validate_scene_layout(args.scene_dir.resolve())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
