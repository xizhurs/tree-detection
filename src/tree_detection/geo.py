"""Geospatial tiling utilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.io import DatasetReader
from rasterio.transform import array_bounds
from rasterio.windows import Window
from shapely import make_valid
from shapely.geometry import box

from tree_detection.dataset import iter_windows


def clip_to_window(
    features: gpd.GeoDataFrame, source: DatasetReader, window: Window
) -> gpd.GeoDataFrame:
    """Clip features to a raster window, preserving the raster CRS."""
    transform = source.window_transform(window)
    bounds = array_bounds(int(window.height), int(window.width), transform)
    valid_features = features.copy()
    valid_features["geometry"] = valid_features.geometry.apply(
        lambda geometry: make_valid(geometry) if geometry is not None else None
    )
    valid_features = valid_features[valid_features.geometry.notnull()]
    clipped = gpd.clip(valid_features, box(*bounds))
    clipped = clipped[clipped.geometry.is_valid & ~clipped.geometry.is_empty]
    return clipped.set_crs(source.crs, allow_override=True)


def write_raster_tile(source: DatasetReader, window: Window, output_path: Path) -> None:
    """Write a raster window as a georeferenced GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = source.profile.copy()
    profile.update(
        width=int(window.width),
        height=int(window.height),
        transform=source.window_transform(window),
    )
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(source.read(window=window))


def tile_sources(
    pairs: Sequence[tuple[Path, Path]],
    output_root: str | Path,
    tile_size: tuple[int, int] = (1024, 1024),
    overlap: int = 0,
    *,
    keep_empty: bool = False,
    keep_columns: Sequence[str] | None = None,
) -> list[tuple[Path, Path]]:
    """Tile raster/GeoJSON scene pairs and return their generated file pairs."""
    output_root = Path(output_root)
    created: list[tuple[Path, Path]] = []

    for raster_path, geojson_path in pairs:
        if not raster_path.is_file():
            raise FileNotFoundError(f"raster does not exist: {raster_path}")
        if not geojson_path.is_file():
            raise FileNotFoundError(f"GeoJSON does not exist: {geojson_path}")

        scene_output = output_root / raster_path.stem
        with rasterio.open(raster_path) as source:
            if source.crs is None:
                raise ValueError(f"raster has no CRS: {raster_path}")
            features = gpd.read_file(geojson_path)
            if features.crs is None:
                raise ValueError(f"GeoJSON has no CRS: {geojson_path}")
            if features.crs != source.crs:
                features = features.to_crs(source.crs)

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
                clipped = clip_to_window(features, source, window)
                if clipped.empty and not keep_empty:
                    continue

                tile_name = (
                    f"{raster_path.stem}_r{pixel_window.row}_c{pixel_window.column}"
                    f"_w{pixel_window.width}_h{pixel_window.height}"
                )
                output_raster = scene_output / f"{tile_name}.tif"
                output_geojson = scene_output / f"{tile_name}.geojson"
                write_raster_tile(source, window, output_raster)

                if keep_columns is not None:
                    columns = [
                        column for column in keep_columns if column in clipped.columns
                    ]
                    clipped = clipped[[*columns, "geometry"]]
                output_geojson.parent.mkdir(parents=True, exist_ok=True)
                clipped.to_file(output_geojson, driver="GeoJSON")
                created.append((output_raster, output_geojson))

    return created
