#!/usr/bin/env python3
"""Create a local virtual environment for benchmark utilities or methods."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_VENV_ROOT = Path("environment/venv/.venvs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a project-local Python virtual environment.")
    parser.add_argument("name", help="Environment name, e.g. benchmark or vggt")
    parser.add_argument(
        "--requirements",
        type=Path,
        help="Optional requirements file to install after creating the venv",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    venv_dir = DEFAULT_VENV_ROOT / args.name
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "venv", str(venv_dir)])
    python_bin = venv_dir / "bin" / "python"
    run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    if args.requirements:
        run([str(python_bin), "-m", "pip", "install", "-r", str(args.requirements)])
    print(venv_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
