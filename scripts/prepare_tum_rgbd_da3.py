#!/usr/bin/env python3
import argparse
import os
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets" / "tum_rgbd"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets" / "tum_rgbd_prepared"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare TUM RGB-D archives into DA3-Streaming image folders."
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Directory containing TUM .tgz archives.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where DA3-ready frame folders will be written.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        required=True,
        help=(
            "TUM sequence names without extension, e.g. "
            "rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg3_teddy"
        ),
    )
    parser.add_argument(
        "--fps",
        nargs="+",
        default=["2", "3", "5"],
        help="Target output FPS values.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract .tgz archives before preparing RGB frame folders.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract an archive even if its extracted folder already exists.",
    )
    parser.add_argument(
        "--force-output",
        action="store_true",
        help="Rewrite output frame folders even if they already contain PNGs.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def extract_archive(archive_path: Path, extract_root: Path, force: bool) -> Path:
    sequence_dir = extract_root / archive_path.stem
    if sequence_dir.exists() and not force:
        return sequence_dir
    if sequence_dir.exists() and force:
        raise RuntimeError(
            f"Refusing to overwrite existing extracted directory manually: {sequence_dir}"
        )
    with tarfile.open(archive_path, "r:gz") as handle:
        handle.extractall(extract_root)
    return sequence_dir


def load_rgb_frames(rgb_dir: Path) -> list[tuple[float, Path]]:
    frames = []
    for image_path in sorted(rgb_dir.glob("*.png")):
        try:
            timestamp = float(image_path.stem)
        except ValueError:
            continue
        frames.append((timestamp, image_path))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {rgb_dir}")
    return frames


def select_frames_by_fps(frames: list[tuple[float, Path]], fps: float) -> list[Path]:
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    interval = 1.0 / fps
    selected = []
    next_time = None
    for timestamp, image_path in frames:
        if next_time is None or timestamp + 1e-9 >= next_time:
            selected.append(image_path)
            next_time = timestamp + interval
    return selected


def write_frame_links(output_dir: Path, selected_frames: list[Path], force: bool) -> None:
    existing = list(output_dir.glob("*.png")) if output_dir.exists() else []
    if existing and not force:
        print(f"[skip] {output_dir} ({len(existing)} PNGs)")
        return
    if output_dir.exists() and force:
        for old_file in output_dir.glob("*.png"):
            old_file.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, source_path in enumerate(selected_frames, start=1):
        target_path = output_dir / f"frame_{index:06d}.png"
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        os.symlink(source_path, target_path)
    print(f"[write] {output_dir} ({len(selected_frames)} PNG symlinks)")


def main() -> int:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_root)

    for sequence in args.sequences:
        archive_path = data_root / f"{sequence}.tgz"
        extracted_dir = data_root / sequence

        if args.extract:
            if not archive_path.exists():
                raise FileNotFoundError(f"Missing archive: {archive_path}")
            extracted_dir = extract_archive(archive_path, data_root, args.force_extract)

        if not extracted_dir.exists():
            raise FileNotFoundError(
                f"Missing extracted sequence directory: {extracted_dir}. "
                "Run with --extract first."
            )

        rgb_dir = extracted_dir / "rgb"
        frames = load_rgb_frames(rgb_dir)
        stem = sequence.replace("rgbd_dataset_", "")

        for fps_text in args.fps:
            fps_value = float(fps_text)
            selected = select_frames_by_fps(frames, fps_value)
            fps_label = fps_text.replace(".", "p")
            output_dir = output_root / f"{stem}_fps{fps_label}"
            write_frame_links(output_dir, selected, args.force_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
