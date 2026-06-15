#!/usr/bin/env python3
"""Prepare combined 7sense RGB sequences for DA3-Streaming."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "datasets" / "7sense"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets" / "7sense_prepared_combined"
ASSUMED_SOURCE_FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare combined 7sense RGB sequences for DA3-Streaming.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene", action="append", default=[], help="Scene name, e.g. chess")
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Specific sequence identifier, e.g. chess/seq-01 or redkitchen_seq-14",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help="Dataset split to prepare when --sequence is not provided.",
    )
    parser.add_argument("--fps", nargs="+", default=["2", "3", "5"], help="Target output FPS values.")
    parser.add_argument(
        "--group-mode",
        choices=["scene", "all", "both"],
        default="both",
        help="Whether to create one combined folder per scene, one global folder, or both.",
    )
    parser.add_argument("--force-output", action="store_true")
    return parser.parse_args()


def split_file_for(scene_dir: Path, split: str) -> Path:
    if split == "train":
        return scene_dir / "TrainSplit.txt"
    if split == "test":
        return scene_dir / "TestSplit.txt"
    raise ValueError(f"unsupported split: {split}")


def split_entries(scene_dir: Path, split: str) -> list[str]:
    if split == "all":
        return sorted(path.name for path in scene_dir.iterdir() if path.is_dir() and path.name.startswith("seq-"))
    path = split_file_for(scene_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    selected: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            match = re.search(r"(\d+)$", stripped)
            if not match:
                continue
            selected.append(f"seq-{int(match.group(1)):02d}")
    return selected


def parse_sequence_filter(value: str) -> tuple[str, str]:
    if "/" in value:
        scene, seq_name = value.split("/", 1)
        return scene, seq_name
    match = re.match(r"(.+)_seq-(\d+)$", value)
    if match:
        return match.group(1), f"seq-{int(match.group(2)):02d}"
    raise ValueError(f"Sequence filter must look like scene/seq-01 or scene_seq-01, got: {value}")


def selected_sequences(root: Path, scenes: list[str], sequence_filters: list[str], split: str) -> list[tuple[str, Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Missing 7sense root: {root}")
    requested_scenes = set(scenes)
    selected: list[tuple[str, Path]] = []
    if sequence_filters:
        for item in sequence_filters:
            scene_name, seq_name = parse_sequence_filter(item)
            seq_dir = root / scene_name / seq_name
            if not seq_dir.exists():
                raise FileNotFoundError(f"Missing extracted sequence: {seq_dir}")
            selected.append((scene_name, seq_dir))
        return sorted(selected)

    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if requested_scenes and scene_dir.name not in requested_scenes:
            continue
        for seq_name in split_entries(scene_dir, split):
            seq_dir = scene_dir / seq_name
            if not seq_dir.exists():
                raise FileNotFoundError(f"Missing extracted sequence directory: {seq_dir}")
            selected.append((scene_dir.name, seq_dir))
    if not selected:
        raise FileNotFoundError("No 7sense sequences matched the requested filters.")
    return selected


def ordered_color_frames(seq_dir: Path) -> list[tuple[float, Path]]:
    frames: list[tuple[float, Path]] = []
    for frame_path in sorted(seq_dir.glob("frame-*.color.png")):
        match = re.search(r"frame-(\d+)\.color\.png$", frame_path.name)
        if not match:
            continue
        timestamp = int(match.group(1)) / ASSUMED_SOURCE_FPS
        frames.append((timestamp, frame_path))
    if not frames:
        raise FileNotFoundError(f"No color frames found in {seq_dir}")
    return frames


def select_frames_by_fps(frames: list[tuple[float, Path]], fps: float) -> list[tuple[float, Path]]:
    interval = 1.0 / fps
    selected: list[tuple[float, Path]] = []
    next_time = None
    for timestamp, image_path in frames:
        if next_time is None or timestamp + 1e-9 >= next_time:
            selected.append((timestamp, image_path))
            next_time = timestamp + interval
    return selected


def grouped_sequences(
    selected: list[tuple[str, Path]], group_mode: str, split: str
) -> dict[str, list[tuple[str, Path]]]:
    groups: dict[str, list[tuple[str, Path]]] = {}
    by_scene: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for scene_name, seq_dir in selected:
        by_scene[scene_name].append((scene_name, seq_dir))
    if group_mode in {"scene", "both"}:
        for scene_name, items in sorted(by_scene.items()):
            groups[f"{scene_name}_{split}_allseq"] = items
    if group_mode in {"all", "both"}:
        groups[f"all_{split}_allseq"] = selected
    return groups


def write_group(
    output_dir: Path,
    manifest_path: Path,
    selected_frames: list[tuple[str, str, float, Path]],
    force: bool,
) -> None:
    existing = list(output_dir.glob("*.png")) if output_dir.exists() else []
    if existing and not force and len(existing) == len(selected_frames):
        print(f"[skip] {output_dir} ({len(existing)} PNGs)")
    else:
        if output_dir.exists():
            for old_file in output_dir.glob("*.png"):
                old_file.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, (_, _, _, source_path) in enumerate(selected_frames, start=1):
            target_path = output_dir / f"frame_{index:06d}.png"
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()
            os.symlink(source_path.resolve(), target_path)
        print(f"[write] {output_dir} ({len(selected_frames)} PNG symlinks)")

    manifest_rows = [
        {
            "combined_index": index,
            "scene": scene_name,
            "sequence": sequence_name,
            "source_timestamp_s": round(timestamp, 6),
            "source_path": str(source_path.resolve()),
        }
        for index, (scene_name, sequence_name, timestamp, source_path) in enumerate(selected_frames, start=1)
    ]
    manifest_path.write_text(json.dumps(manifest_rows, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected = selected_sequences(root, args.scene, args.sequence, args.split)
    groups = grouped_sequences(selected, args.group_mode, args.split)

    for group_name, items in groups.items():
        all_frames: list[tuple[str, str, float, Path]] = []
        time_offset = 0.0
        for scene_name, seq_dir in items:
            sequence_name = seq_dir.name
            sequence_frames = ordered_color_frames(seq_dir)
            for timestamp, frame_path in sequence_frames:
                all_frames.append((scene_name, sequence_name, time_offset + timestamp, frame_path))
            if sequence_frames:
                time_offset = all_frames[-1][2] + (1.0 / ASSUMED_SOURCE_FPS)
        for fps_text in args.fps:
            fps_value = float(fps_text)
            reduced = select_frames_by_fps([(ts, path) for _, _, ts, path in all_frames], fps_value)
            source_lookup = {(path.resolve(), round(ts, 9)): (scene, seq, ts, path) for scene, seq, ts, path in all_frames}
            selected_frames: list[tuple[str, str, float, Path]] = []
            for timestamp, path in reduced:
                selected_frames.append(source_lookup[(path.resolve(), round(timestamp, 9))])
            output_dir = output_root / f"{group_name}_fps{fps_text.replace('.', 'p')}"
            manifest_path = output_dir.with_name(output_dir.name + "_manifest.json")
            write_group(output_dir, manifest_path, selected_frames, args.force_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
