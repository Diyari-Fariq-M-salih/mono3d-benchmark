#!/usr/bin/env python3
"""Print the configured dataset and method install targets."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    datasets = load_json(Path("datasets/dataset_registry.json"))["datasets"]
    methods = load_json(Path("methods/method_registry.json"))["methods"]

    print("Datasets:")
    for item in datasets:
        print(f"- {item['display_name']}: {item['target_dir']} <- {item['landing_page']}")

    print("Methods:")
    for item in methods:
        clone_url = item["clone_url"] or "TBD"
        print(f"- {item['display_name']}: {item['target_dir']} <- {clone_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
