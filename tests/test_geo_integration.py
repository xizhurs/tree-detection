import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from tree_detection.coco import build_coco_dataset
from tree_detection.geo import tile_sources


def create_scene(tmp_path: Path) -> tuple[Path, Path]:
    raster_path = tmp_path / "scene.tif"
    geojson_path = tmp_path / "scene.geojson"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 8, 1, 1),
    ) as destination:
        destination.write(np.full((3, 8, 8), 100, dtype=np.uint8))

    gpd.GeoDataFrame(
        {"category": ["tree"]},
        geometry=[box(1, 5, 3, 7)],
        crs="EPSG:3857",
    ).to_file(geojson_path, driver="GeoJSON")
    return raster_path, geojson_path


def test_tile_sources_preserves_georeferencing(tmp_path: Path) -> None:
    pair = create_scene(tmp_path)
    created = tile_sources([pair], tmp_path / "tiles", tile_size=(4, 4))

    assert len(created) == 1
    with rasterio.open(created[0][0]) as tile:
        assert (tile.width, tile.height) == (4, 4)
        assert tile.crs.to_epsg() == 3857
    clipped = gpd.read_file(created[0][1])
    assert clipped.total_bounds.tolist() == [1.0, 5.0, 3.0, 7.0]


def test_build_coco_dataset_from_synthetic_scene(tmp_path: Path) -> None:
    pair = create_scene(tmp_path)
    annotation_path = build_coco_dataset(
        [pair], tmp_path / "coco", tile_size=(4, 4), overlap=0
    )

    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert coco["categories"] == [{"id": 1, "name": "tree", "supercategory": ""}]
    assert len(coco["images"]) == 1
    assert coco["annotations"][0]["category_id"] == 1
    assert (tmp_path / "coco" / coco["images"][0]["file_name"]).is_file()
