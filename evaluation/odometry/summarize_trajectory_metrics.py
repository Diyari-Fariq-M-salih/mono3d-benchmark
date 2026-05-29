#!/usr/bin/env python3
"""Summarize trajectory benchmark rows from CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize odometry metrics CSV.")
    parser.add_argument("csv_path", type=Path, help="Path to an odometry results CSV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print(f"rows: {len(rows)}")
    methods = sorted({row.get("method", "").strip() for row in rows if row.get("method")})
    print(f"methods: {', '.join(methods) if methods else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
