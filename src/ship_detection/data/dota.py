"""Utilities for the canonical DOTA-style dataset used by all models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Iterable, Sequence


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DotaObject:
    """One DOTA annotation with four ordered polygon vertices."""

    class_name: str
    polygon: tuple[tuple[float, float], ...]
    difficulty: int = 0

    @property
    def horizontal_bbox(self) -> tuple[float, float, float, float]:
        """Return the enclosing axis-aligned box as xmin, ymin, xmax, ymax."""

        x_values = [point[0] for point in self.polygon]
        y_values = [point[1] for point in self.polygon]
        return min(x_values), min(y_values), max(x_values), max(y_values)


@dataclass(frozen=True)
class DatasetIssue:
    """A non-fatal issue found while scanning one dataset split."""

    split: str
    image: str
    message: str


@dataclass(frozen=True)
class DatasetReport:
    """Summary produced by dataset validation or preparation."""

    root: str
    splits: dict[str, dict[str, int]]
    issues: tuple[DatasetIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        result = asdict(self)
        result["issues"] = [asdict(issue) for issue in self.issues]
        return result

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _parse_difficulty(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def _polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
        / 2
    )


def parse_dota_lines(lines: Iterable[str]) -> list[DotaObject]:
    """Parse DOTA lines and skip headers or malformed annotations.

    A supported annotation has the standard form:
    ``x1 y1 x2 y2 x3 y3 x4 y4 class [difficulty]``.
    """

    objects = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue

        try:
            coordinates = [float(value) for value in parts[:8]]
        except ValueError:
            continue

        if not all(math.isfinite(value) for value in coordinates):
            continue

        polygon = tuple(
            (coordinates[index], coordinates[index + 1])
            for index in range(0, 8, 2)
        )
        if _polygon_area(polygon) <= 0:
            continue

        objects.append(
            DotaObject(
                class_name=parts[8],
                polygon=polygon,
                difficulty=_parse_difficulty(parts[9] if len(parts) > 9 else None),
            )
        )
    return objects


def parse_dota_file(path: str | Path) -> list[DotaObject]:
    """Read and parse one DOTA annotation file."""

    label_path = Path(path)
    if not label_path.exists():
        return []
    return parse_dota_lines(label_path.read_text(encoding="utf-8").splitlines())


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _scan_split(root: Path, split: str) -> tuple[dict[str, int], list[DatasetIssue]]:
    split_root = root / split
    image_dir = split_root / "images"
    label_dir = split_root / "labelTxt"
    issues: list[DatasetIssue] = []
    stats = {"images": 0, "labels": 0, "objects": 0, "missing_labels": 0}

    if not image_dir.is_dir():
        issues.append(DatasetIssue(split, "", f"missing directory: {image_dir}"))
        return stats, issues
    if not label_dir.is_dir():
        issues.append(DatasetIssue(split, "", f"missing directory: {label_dir}"))
        return stats, issues

    images = _image_files(image_dir)
    stats["images"] = len(images)
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            stats["missing_labels"] += 1
            issues.append(DatasetIssue(split, image_path.name, "missing label file"))
            continue

        stats["labels"] += 1
        objects = parse_dota_file(label_path)
        stats["objects"] += len(objects)

    return stats, issues


def validate_dota_dataset(root: str | Path) -> DatasetReport:
    """Validate the canonical ``train/val/test`` DOTA directory layout."""

    dataset_root = Path(root)
    split_stats: dict[str, dict[str, int]] = {}
    issues: list[DatasetIssue] = []
    for split in SPLITS:
        stats, split_issues = _scan_split(dataset_root, split)
        split_stats[split] = stats
        issues.extend(split_issues)
    return DatasetReport(str(dataset_root), split_stats, tuple(issues))


def _format_dota_object(obj: DotaObject) -> str:
    coordinates = " ".join(
        f"{coordinate:g}" for point in obj.polygon for coordinate in point
    )
    return f"{coordinates} {obj.class_name} {obj.difficulty}"


def prepare_dota_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    allowed_classes: set[str] | None = None,
) -> DatasetReport:
    """Copy images and canonicalize DOTA labels into a new dataset root.

    The source directory is read only. Invalid annotation lines are omitted,
    while each source image receives a matching label file in the output.
    """

    source = Path(source_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    issues: list[DatasetIssue] = []
    split_stats: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        source_images = source / split / "images"
        source_labels = source / split / "labelTxt"
        output_images = output / split / "images"
        output_labels = output / split / "labelTxt"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        stats = {"images": 0, "labels": 0, "objects": 0, "missing_labels": 0}
        if not source_images.is_dir():
            issues.append(DatasetIssue(split, "", f"missing directory: {source_images}"))
            split_stats[split] = stats
            continue

        for image_path in _image_files(source_images):
            stats["images"] += 1
            shutil.copy2(image_path, output_images / image_path.name)
            source_label = source_labels / f"{image_path.stem}.txt"
            target_label = output_labels / f"{image_path.stem}.txt"
            objects = parse_dota_file(source_label)
            if not source_label.exists():
                stats["missing_labels"] += 1
                issues.append(DatasetIssue(split, image_path.name, "missing label file"))
            filtered_objects = [
                obj for obj in objects if allowed_classes is None or obj.class_name in allowed_classes
            ]
            if source_label.exists():
                stats["labels"] += 1
            stats["objects"] += len(filtered_objects)
            target_label.write_text(
                "\n".join(_format_dota_object(obj) for obj in filtered_objects),
                encoding="utf-8",
            )

        split_stats[split] = stats

    return DatasetReport(str(output), split_stats, tuple(issues))
