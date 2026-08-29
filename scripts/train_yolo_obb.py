"""Train an Ultralytics YOLO OBB model from an exported data.yaml file."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="YOLO-OBB data.yaml file.")
    parser.add_argument("--weights", default="yolov8n-obb.pt")
    parser.add_argument("--output-root", default=Path("outputs/yolov8_obb"), type=Path)
    parser.add_argument("--run-name", help="Defaults to a timestamped run name.")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--image-size", default=1024, type=int)
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", help="Ultralytics device value, for example 0 or cpu.")
    parser.add_argument(
        "--test-after-train",
        action="store_true",
        help="Evaluate the selected best checkpoint once on the test split.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"data configuration not found: {args.data}")
    if args.epochs <= 0 or args.image_size <= 0 or args.batch <= 0:
        raise ValueError("epochs, image-size and batch must be positive")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required. Install requirements/yolo.txt before training."
        ) from error

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    train_options = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.image_size,
        "batch": args.batch,
        "seed": args.seed,
        "project": str(args.output_root),
        "name": run_name,
        "exist_ok": False,
    }
    if args.device:
        train_options["device"] = args.device

    model = YOLO(args.weights)
    model.train(**train_options)
    best_checkpoint = args.output_root / run_name / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"best checkpoint was not created: {best_checkpoint}")
    print(f"best_checkpoint={best_checkpoint}")

    if args.test_after_train:
        evaluation_options = {"data": str(args.data.resolve()), "split": "test"}
        if args.device:
            evaluation_options["device"] = args.device
        YOLO(str(best_checkpoint)).val(**evaluation_options)
        print("test evaluation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
