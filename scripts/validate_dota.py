"""Validate a canonical DOTA dataset without changing it."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ship_detection.data import validate_dota_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    report = validate_dota_dataset(args.dataset_root)
    if args.report:
        report.save_json(args.report)
    for split, stats in report.splits.items():
        print(
            f"{split}: images={stats['images']} labels={stats['labels']} "
            f"objects={stats['objects']} missing_labels={stats['missing_labels']}"
        )
    if report.issues:
        print(f"issues={len(report.issues)}")
        return 1
    print("dataset validation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
