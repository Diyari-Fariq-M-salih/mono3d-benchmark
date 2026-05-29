#!/usr/bin/env python3
"""Clone a benchmark method from the local method registry."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REGISTRY_PATH = Path("methods/method_registry.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone a method repository from the local registry.")
    parser.add_argument("method_name", help="Registry key, e.g. orb_slam3")
    return parser.parse_args()


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    registry = load_registry()
    methods = {item["name"]: item for item in registry["methods"]}
    if args.method_name not in methods:
        raise SystemExit(f"unknown method: {args.method_name}")

    method = methods[args.method_name]
    clone_url = method.get("clone_url", "")
    if not clone_url:
        raise SystemExit(f"clone URL is not yet configured for {args.method_name}")

    target_dir = Path(method["target_dir"])
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        print(f"already exists: {target_dir}")
        return 0

    subprocess.run(["git", "clone", clone_url, str(target_dir)], check=True)
    print(target_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
