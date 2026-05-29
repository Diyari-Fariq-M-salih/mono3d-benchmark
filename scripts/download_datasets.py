#!/usr/bin/env python3
"""Download benchmark datasets from a local manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


MANIFEST_PATH = Path("datasets/download_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets from the local manifest.")
    parser.add_argument(
        "selection",
        help="Dataset key or preset name from datasets/download_manifest.json",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract supported archives after download",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist locally",
    )
    return parser.parse_args()


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def resolve_selection(manifest: dict, selection: str) -> list[str]:
    if selection in manifest["datasets"]:
        return [selection]
    if selection in manifest["presets"]:
        return manifest["presets"][selection]
    raise SystemExit(f"unknown dataset or preset: {selection}")


def extract_archive(archive_path: Path, kind: str) -> None:
    if kind == "zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(archive_path.parent)
        return
    if kind == "7z":
        seven_zip = shutil.which("7z")
        if not seven_zip:
            print(f"7z not installed, keeping archive only: {archive_path}")
            return
        subprocess.run([seven_zip, "x", str(archive_path)], cwd=archive_path.parent, check=True)
        return
    raise SystemExit(f"unsupported archive type: {kind}")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(url, destination)
        return
    except Exception as exc:
        wget = shutil.which("wget")
        if not wget:
            raise
        print(f"python download failed ({exc}), retrying with wget")
        subprocess.run([wget, "-c", "-O", str(destination), url], check=True)


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    dataset_keys = resolve_selection(manifest, args.selection)

    for key in dataset_keys:
        spec = manifest["datasets"][key]
        target_dir = Path(spec["target_dir"])
        for archive in spec["archives"]:
            archive_path = target_dir / archive["filename"]
            if args.skip_existing and archive_path.exists():
                print(f"skip existing: {archive_path}")
            else:
                print(f"download: {archive['url']} -> {archive_path}")
                download_file(archive["url"], archive_path)
            if args.extract:
                print(f"extract: {archive_path}")
                extract_archive(archive_path, archive["extract"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
