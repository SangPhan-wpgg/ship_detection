"""Validate and prepare the canonical DOTA dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ship_detection.data import prepare_dota_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--class-name",
        default="ship",
        help="Keep only this DOTA class; use an empty value to keep all classes.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON path for the preparation report.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    allowed_classes = {args.class_name} if args.class_name else None
    report = prepare_dota_dataset(
        args.source_root,
        args.output_root,
        allowed_classes=allowed_classes,
    )
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
    print("dataset preparation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
