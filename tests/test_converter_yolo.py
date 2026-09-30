import json
from pathlib import Path

import pytest

from tree_detection.converter_yolo import coco_bbox_to_yolo, convert_coco_to_yolo


def test_coco_bbox_to_yolo() -> None:
    assert coco_bbox_to_yolo([10, 20, 40, 20], 100, 100) == (0.3, 0.3, 0.4, 0.2)


@pytest.mark.parametrize(
    ("bbox", "width", "height"),
    [([0, 0, 1, 1], 0, 10), ([0, 0, -1, 1], 10, 10), ([0, 0, 1], 10, 10)],
)
def test_coco_bbox_to_yolo_rejects_invalid_input(
    bbox: list[float], width: int, height: int
) -> None:
    with pytest.raises(ValueError):
        coco_bbox_to_yolo(bbox, width, height)


def test_convert_coco_to_yolo(tmp_path: Path) -> None:
    input_dir = tmp_path / "coco" / "train"
    output_dir = tmp_path / "yolo"
    input_dir.mkdir(parents=True)
    (input_dir / "tile.jpg").write_bytes(b"image data")
    (input_dir / "empty.jpg").write_bytes(b"image data")
    coco = {
        "images": [
            {"id": 1, "file_name": "tile.jpg", "width": 100, "height": 50},
            {"id": 2, "file_name": "empty.jpg", "width": 100, "height": 50},
        ],
        "categories": [{"id": 7, "name": "tree"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 5, 20, 10]}
        ],
    }
    (input_dir / "_annotations.coco.json").write_text(
        json.dumps(coco), encoding="utf-8"
    )

    convert_coco_to_yolo(tmp_path / "coco", output_dir, "train")

    assert (output_dir / "train/images/tile.jpg").read_bytes() == b"image data"
    assert (output_dir / "train/labels/tile.txt").read_text(encoding="utf-8") == (
        "0 0.200000 0.200000 0.200000 0.200000\n"
    )
    assert (output_dir / "train/labels/empty.txt").read_text(encoding="utf-8") == ""
