#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets" / "custom"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a custom video into DA3-Streaming frame folders."
    )
    parser.add_argument("video", type=Path, help="Input video file.")
    parser.add_argument(
        "--fps",
        nargs="+",
        default=["2", "3", "5"],
        help="Output FPS values, e.g. 1 2 3 5.",
    )
    parser.add_argument(
        "--scale-width",
        default="640",
        help="Resize width passed to ffmpeg scale=<width>:-1.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where extracted frame folders will be created.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracted frame folders.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    video_path = resolve_path(args.video)
    output_root = resolve_path(args.output_root)

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    stem = video_path.stem
    for fps_text in args.fps:
        fps_label = fps_text.replace(".", "p")
        out_dir = output_root / f"{stem}_fps{fps_label}"
        if out_dir.exists():
            existing = list(out_dir.glob("*.png"))
            if existing and not args.force:
                print(f"[skip] {out_dir} ({len(existing)} PNGs)")
                continue
            if args.force:
                shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pattern = out_dir / "frame_%06d.png"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps_text},scale={args.scale_width}:-1",
            str(out_pattern),
        ]
        print(f"[extract] {' '.join(command)}")
        subprocess.run(command, check=True)
        count = len(list(out_dir.glob("*.png")))
        print(f"[write] {out_dir} ({count} PNGs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
