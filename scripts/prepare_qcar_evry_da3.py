#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets" / "Qcar_evry"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets" / "qcar_evry_prepared"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare QCar Evry RGB sequences into DA3-Streaming frame folders."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sequence", action="append", default=[], help="Scene name, e.g. warehouse1")
    parser.add_argument("--fps", nargs="+", default=["2", "3", "5"], help="Target output FPS values.")
    parser.add_argument(
        "--force-output",
        action="store_true",
        help="Rewrite output frame folders even if they already contain PNGs.",
    )
    return parser.parse_args()


def selected_sequences(data_root: Path, names: list[str]) -> list[Path]:
    candidates = sorted(
        path for path in data_root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and (path / "rgb.txt").exists() and (path / "rgb").exists()
    )
    if not names:
        if not candidates:
            raise FileNotFoundError(f"No QCar Evry sequences found in {data_root}")
        return candidates

    wanted = []
    for name in names:
        path = data_root / name
        if not (path / "rgb.txt").exists() or not (path / "rgb").exists():
            raise FileNotFoundError(f"Missing QCar Evry RGB sequence: {path}")
        wanted.append(path)
    return wanted


def load_rgb_frames(sequence_dir: Path) -> list[tuple[float, Path]]:
    rows: list[tuple[float, Path]] = []
    with (sequence_dir / "rgb.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), sequence_dir / parts[1]))
    if not rows:
        raise FileNotFoundError(f"No RGB frames found in {sequence_dir / 'rgb.txt'}")
    return rows


def select_frames_by_fps(frames: list[tuple[float, Path]], fps: float) -> list[Path]:
    interval = 1.0 / fps
    selected: list[Path] = []
    next_time = None
    for timestamp, image_path in frames:
        if next_time is None or timestamp + 1e-9 >= next_time:
            selected.append(image_path)
            next_time = timestamp + interval
    return selected


def write_frame_links(output_dir: Path, selected_frames: list[Path], force: bool) -> None:
    existing = list(output_dir.glob("*.png")) if output_dir.exists() else []
    if existing and not force and len(existing) == len(selected_frames):
        print(f"[skip] {output_dir} ({len(existing)} PNGs)")
        return
    if output_dir.exists():
        for old_file in output_dir.glob("*.png"):
            old_file.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, source_path in enumerate(selected_frames, start=1):
        target_path = output_dir / f"frame_{index:06d}.png"
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        os.symlink(source_path.resolve(), target_path)
    print(f"[write] {output_dir} ({len(selected_frames)} PNG symlinks)")


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for sequence_dir in selected_sequences(data_root, args.sequence):
        frames = load_rgb_frames(sequence_dir)
        for fps_text in args.fps:
            fps_value = float(fps_text)
            selected = select_frames_by_fps(frames, fps_value)
            output_dir = output_root / f"{sequence_dir.name}_fps{fps_text.replace('.', 'p')}"
            write_frame_links(output_dir, selected, args.force_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
