"""Convert COCO detection datasets to Ultralytics YOLO format."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def coco_bbox_to_yolo(
    bbox: list[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Convert a COCO ``[x, y, width, height]`` box to normalized YOLO form."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four values")

    x, y, width, height = bbox
    if width < 0 or height < 0:
        raise ValueError("bbox width and height must be non-negative")

    return (
        min(max((x + width / 2.0) / image_width, 0.0), 1.0),
        min(max((y + height / 2.0) / image_height, 0.0), 1.0),
        min(max(width / image_width, 0.0), 1.0),
        min(max(height / image_height, 0.0), 1.0),
    )


def convert_coco_to_yolo(
    input_base_dir: str | Path, output_dir: str | Path, split: str = "train"
) -> None:
    """Convert one COCO split and copy its images into a YOLO directory tree."""
    input_split_dir = Path(input_base_dir) / split
    output_split_dir = Path(output_dir) / split
    output_images_dir = output_split_dir / "images"
    output_labels_dir = output_split_dir / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = input_split_dir / "_annotations.coco.json"
    with annotation_path.open(encoding="utf-8") as file:
        coco: dict[str, Any] = json.load(file)

    images_by_id = {image["id"]: image for image in coco["images"]}
    category_ids = sorted(category["id"] for category in coco["categories"])
    category_to_yolo = {
        category_id: index for index, category_id in enumerate(category_ids)
    }

    yolo_lines: defaultdict[int, list[str]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0) == 1:
            continue
        image = images_by_id.get(annotation["image_id"])
        if image is None:
            continue
        category_id = annotation["category_id"]
        if category_id not in category_to_yolo:
            raise ValueError(f"unknown COCO category id: {category_id}")

        x_center, y_center, width, height = coco_bbox_to_yolo(
            annotation["bbox"], image["width"], image["height"]
        )
        class_id = category_to_yolo[category_id]
        yolo_lines[annotation["image_id"]].append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    for image in coco["images"]:
        source_image = input_split_dir / image["file_name"]
        if not source_image.is_file():
            raise FileNotFoundError(f"COCO image does not exist: {source_image}")

        image_name = Path(image["file_name"]).name
        label_path = output_labels_dir / Path(image_name).with_suffix(".txt")
        lines = yolo_lines.get(image["id"], [])
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        shutil.copy2(source_image, output_images_dir / image_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_coco_to_yolo(args.input_base_dir, args.output_dir, args.split)


if __name__ == "__main__":
    main()
