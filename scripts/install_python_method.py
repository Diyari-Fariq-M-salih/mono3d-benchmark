#!/usr/bin/env python3
"""Create a per-method virtual environment and install a local Python repo."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a method venv and install a local repository into it."
    )
    parser.add_argument("venv_name", help="Environment name under environment/venv/.venvs/")
    parser.add_argument("repo_dir", type=Path, help="Path to the local repo directory")
    parser.add_argument(
        "--extras",
        default="",
        help='Optional extras to install, e.g. "app" or "colmap,gradio"',
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    args = parse_args()
    run(["python3", "scripts/bootstrap_venv.py", args.venv_name])

    venv_python = Path("environment/venv/.venvs") / args.venv_name / "bin" / "python"
    install_target = args.repo_dir.resolve()
    if args.extras:
        editable_spec = f"{install_target}[{args.extras}]"
    else:
        editable_spec = str(install_target)

    run([str(venv_python), "-m", "pip", "install", "-e", editable_spec])
    print(f"installed {install_target} into {args.venv_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
