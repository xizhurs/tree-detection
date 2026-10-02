"""Build bbox-only COCO datasets from GeoTIFF and GeoJSON scene pairs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import rowcol
from rasterio.windows import Window
from shapely.geometry import box

from tree_detection.dataset import iter_windows


def geometry_bbox(
    geometry: Any,
    transform: Any,
    width: int,
    height: int,
) -> list[int] | None:
    """Convert world-coordinate geometry bounds to a clipped COCO pixel bbox."""
    if geometry.is_empty:
        return None
    minimum_x, minimum_y, maximum_x, maximum_y = geometry.bounds
    row_min, column_min = rowcol(transform, minimum_x, maximum_y)
    row_max, column_max = rowcol(transform, maximum_x, minimum_y)
    x_min = max(0, min(column_min, column_max))
    x_max = min(width - 1, max(column_min, column_max))
    y_min = max(0, min(row_min, row_max))
    y_max = min(height - 1, max(row_min, row_max))
    if x_max < x_min or y_max < y_min:
        return None
    return [
        int(x_min),
        int(y_min),
        int(x_max - x_min + 1),
        int(y_max - y_min + 1),
    ]


def _to_rgb(tile: np.ndarray) -> np.ndarray:
    if tile.shape[0] < 3:
        tile = np.repeat(tile[:1], 3, axis=0)
    tile = tile[:3]
    if tile.dtype != np.uint8:
        values = tile.astype(np.float32)
        minimum = values.min(axis=(1, 2), keepdims=True)
        span = values.max(axis=(1, 2), keepdims=True) - minimum
        span[span == 0] = 1
        tile = ((values - minimum) / span * 255).astype(np.uint8)
    return np.transpose(tile, (1, 2, 0))


def build_coco_dataset(
    pairs: Sequence[tuple[Path, Path]],
    output_dir: str | Path,
    tile_size: tuple[int, int] = (512, 512),
    overlap: int = 128,
    *,
    class_name_attribute: str = "category",
    keep_empty: bool = False,
) -> Path:
    """Create tiled JPEG images and a COCO annotation file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    category_ids: dict[str, int] = {}
    next_image_id = 1
    next_annotation_id = 1

    for raster_path, geojson_path in pairs:
        with rasterio.open(raster_path) as source:
            features = gpd.read_file(geojson_path)
            if source.crs is None or features.crs is None:
                raise ValueError("raster and GeoJSON must both define a CRS")
            if features.crs != source.crs:
                features = features.to_crs(source.crs)
            if class_name_attribute not in features.columns:
                features[class_name_attribute] = "tree"

            class_names = sorted(
                str(name) for name in features[class_name_attribute].dropna().unique()
            )
            for class_name in class_names:
                category_ids.setdefault(class_name, len(category_ids) + 1)

            for pixel_window in iter_windows(
                source.width,
                source.height,
                tile_size[0],
                tile_size[1],
                overlap,
            ):
                window = Window(
                    pixel_window.column,
                    pixel_window.row,
                    pixel_window.width,
                    pixel_window.height,
                )
                transform = source.window_transform(window)
                bounds = rasterio.transform.array_bounds(
                    pixel_window.height, pixel_window.width, transform
                )
                window_polygon = box(*bounds)
                selected = features[features.intersects(window_polygon)]
                if selected.empty and not keep_empty:
                    continue

                filename = (
                    f"{raster_path.stem}_{pixel_window.column}_{pixel_window.row}.jpg"
                )
                tile = source.read(window=window)
                Image.fromarray(_to_rgb(tile)).save(output_dir / filename, quality=95)
                images.append(
                    {
                        "id": next_image_id,
                        "file_name": filename,
                        "width": pixel_window.width,
                        "height": pixel_window.height,
                    }
                )

                for feature in selected.itertuples():
                    clipped = feature.geometry.intersection(window_polygon)
                    bbox = geometry_bbox(
                        clipped,
                        transform,
                        pixel_window.width,
                        pixel_window.height,
                    )
                    if bbox is None:
                        continue
                    class_name = str(getattr(feature, class_name_attribute))
                    annotations.append(
                        {
                            "id": next_annotation_id,
                            "image_id": next_image_id,
                            "category_id": category_ids[class_name],
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                        }
                    )
                    next_annotation_id += 1
                next_image_id += 1

    categories = [
        {"id": category_id, "name": name, "supercategory": ""}
        for name, category_id in sorted(category_ids.items(), key=lambda item: item[1])
    ]
    output_path = output_dir / "_annotations.coco.json"
    output_path.write_text(
        json.dumps(
            {
                "info": {"description": "Tree detection dataset", "version": "1.0"},
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path
