"""Export the canonical DOTA dataset in Ultralytics YOLO-OBB format."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ship_detection.data.yolo import export_yolo_obb_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--class-name",
        action="append",
        dest="class_names",
        help="Class to export. Repeat to export multiple classes. Defaults to ship.",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = export_yolo_obb_dataset(
        args.source_root,
        args.output_root,
        class_names=tuple(args.class_names or ["ship"]),
    )
    if args.report:
        result.dataset.save_json(args.report)

    for split, stats in result.dataset.splits.items():
        print(
            f"{split}: images={stats['images']} labels={stats['labels']} "
            f"objects={stats['objects']} missing_labels={stats['missing_labels']}"
        )
    print(f"data_yaml={result.data_yaml}")
    if result.dataset.issues:
        print(f"issues={len(result.dataset.issues)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
