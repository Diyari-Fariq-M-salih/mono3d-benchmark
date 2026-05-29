#!/usr/bin/env python3
"""Build simple markdown tables from benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a markdown summary table from CSV.")
    parser.add_argument("csv_path", type=Path, help="Input CSV file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"Loaded {len(rows)} rows from {args.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
