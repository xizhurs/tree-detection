import random
import rasterio
import geopandas as gpd
from rasterio.windows import Window
from rasterio.transform import array_bounds
from pathlib import Path
from typing import Iterable, Tuple, List, Optional
from shapely.geometry import box
from shapely import make_valid  # for shapely >= 2.0; else use geom.buffer(0)


def _windows_grid(
    width: int, height: int, tile_w: int, tile_h: int, overlap: int
) -> Iterable[Window]:
    """Generate raster windows that cover the image with given tile size and overlap."""
    assert (
        0 <= overlap < min(tile_w, tile_h)
    ), "overlap must be >=0 and smaller than tile size"
    step_w = tile_w - overlap
    step_h = tile_h - overlap
    for r0 in range(0, height, step_h):
        for c0 in range(0, width, step_w):
            w = min(tile_w, width - c0)
            h = min(tile_h, height - r0)
            if w > 0 and h > 0:
                yield Window(col_off=c0, row_off=r0, width=w, height=h)


def _clip_gdf_to_window(
    gdf: gpd.GeoDataFrame, src: rasterio.DatasetReader, win: Window
) -> gpd.GeoDataFrame:
    """Clip GeoDataFrame to window bounds in the raster CRS."""
    tfm = src.window_transform(win)
    bounds = array_bounds(int(win.height), int(win.width), tfm)
    wpoly = box(*bounds)
    # Fix invalid geoms if any
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: make_valid(geom) if geom is not None else None
    )
    gdf = gdf[gdf.geometry.notnull()]
    gdf_clip = gpd.clip(gdf, wpoly)
    # Keep valid, non-empty
    gdf_clip = gdf_clip[gdf_clip.geometry.is_valid & ~gdf_clip.geometry.is_empty]
    gdf_clip = gdf_clip.set_crs(src.crs, allow_override=True)
    return gdf_clip


def _write_tile_tif(src: rasterio.DatasetReader, win: Window, out_path: Path):
    """Write a window of the source raster to a new GeoTIFF with correct transform/profile."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = src.profile.copy()
    profile.update(
        width=int(win.width),
        height=int(win.height),
        transform=src.window_transform(win),
    )
    # Most drivers keep dtype/count/CRS automatically from profile
    with rasterio.open(out_path, "w", **profile) as dst:
        for b in range(1, src.count + 1):
            dst.write(src.read(b, window=win, boundless=False), b)


def tile_tiffs_with_geojson(
    pairs: List[Tuple[Path, Path]],
    out_root: Path,
    tile_size: Tuple[int, int] = (1024, 1024),
    overlap: int = 0,
    keep_empty_tiles: bool = False,
    keep_columns: Optional[List[str]] = None,
) -> List[Tuple[Path, Path]]:
    """
    Split many (TIFF, GeoJSON) sources into smaller tiles with matched clipped GeoJSONs.

    Args:
        pairs: list of (tif_path, geojson_path).
        out_root: root output directory.
        tile_size: (tile_width, tile_height) in pixels.
        overlap: overlap (pixels) between adjacent tiles.
        keep_empty_tiles: if False, skip tiles with no intersecting features.
        keep_columns: optional list of attribute columns to keep in output GeoJSON.

    Returns:
        List of (tile_tif_path, tile_geojson_path) created.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    created: List[Tuple[Path, Path]] = []

    tw, th = int(tile_size[0]), int(tile_size[1])

    for tif_path, gj_path in pairs:
        tif_path, gj_path = Path(tif_path), Path(gj_path)
        assert tif_path.exists(), f"Missing raster: {tif_path}"
        assert gj_path.exists(), f"Missing GeoJSON: {gj_path}"

        stem = tif_path.stem
        scene_out_dir = out_root  # / stem
        scene_out_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(tif_path) as src:
            gdf = gpd.read_file(gj_path)
            # Match CRS
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            W, H = src.width, src.height
            for win in _windows_grid(W, H, tw, th, overlap):
                r, c = int(win.row_off), int(win.col_off)
                # Clip vector to this window
                gdf_clip = _clip_gdf_to_window(gdf, src, win)

                if not keep_empty_tiles and len(gdf_clip) == 0:
                    continue

                # Write tile raster
                tile_name = f"{stem}_r{r}_c{c}_w{int(win.width)}_h{int(win.height)}"
                out_tif = scene_out_dir / f"{tile_name}.tif"
                _write_tile_tif(src, win, out_tif)

                # Write clipped GeoJSON
                if keep_columns:
                    cols = [c for c in keep_columns if c in gdf_clip.columns]
                    gdf_out = (
                        gdf_clip[cols + ["geometry"]]
                        if cols
                        else gdf_clip[["geometry"]]
                    )
                else:
                    gdf_out = gdf_clip
                out_gj = scene_out_dir / f"{tile_name}.geojson"
                gdf_out.to_file(out_gj, driver="GeoJSON")

                created.append((out_tif, out_gj))

        print(
            f"[OK] {stem}: wrote {len([p for p in created if p[0].parent==scene_out_dir])} tiles to {scene_out_dir}"
        )

    return created


# Optional: compute a weight per scene to stratify by 'tree' count rather than file count
def count_labels(gj: Path) -> int:
    try:
        gdf = gpd.read_file(gj)
        return len(gdf)
    except Exception:
        return 0


def get_split_pairs(
    root_dir=Path("data/raw"),
    train_ratio=0.7,
    val_ratio=0.15,
    tile_size=(1024, 1024),
    split_tiffs_first=True,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    # find all (tif, geojson) pairs
    if split_tiffs_first:
        tifs = sorted(root_dir.rglob("*.tif"))
        pairs = []
        for t in tifs:
            gj = t.with_suffix(".geojson")
            if gj.exists():
                pairs.append((t, gj))

        # Run tiling
        created = tile_tiffs_with_geojson(
            pairs=pairs,
            out_root=Path("data/tiles_scenes"),
            tile_size=tile_size,  # e.g., 1024px tiles
            overlap=0,  # optional overlap if you need it
            keep_empty_tiles=False,  # skip tiles without labels
            keep_columns=["category"],  # keep specific attributes if you want
        )
        print(f"Created {len(created)} tiled (tif, geojson) pairs.")
        root_dir = Path("data/tiles_scenes")
    tifs = list(root_dir.rglob("*.tif"))
    pairs = [
        (t, t.with_suffix(".geojson"))
        for t in tifs
        if t.with_suffix(".geojson").exists()
    ]
    assert pairs, "No (tif, geojson) pairs found."

    pairs_with_counts = [(t, g, count_labels(g)) for t, g in pairs]
    total_labels = sum(c for _, _, c in pairs_with_counts) or len(pairs_with_counts)

    # ------------- 2) deterministic split by scene -------------
    rng = random.Random(42)
    pairs_with_counts.sort(key=lambda x: x[0].name)  # stable order
    rng.shuffle(pairs_with_counts)

    # Simple 70/15/15 by cumulative label counts (fallbacks to file-count if counts are 0)
    train, val, test = [], [], []
    acc_train = acc_val = acc_test = 0

    for tif, gj, c in pairs_with_counts:
        w = c if total_labels > 0 else 1
        # Greedy fill keeping roughly 70/15/15 of total weight
        if acc_train / (total_labels or len(pairs_with_counts)) < train_ratio:
            train.append((tif, gj))
            acc_train += w
        elif acc_val / (total_labels or len(pairs_with_counts)) < val_ratio:
            val.append((tif, gj))
            acc_val += w
        else:
            test.append((tif, gj))
            acc_test += w

    print(f"Scenes -> train {len(train)}, val {len(val)}, test {len(test)}")
    return train, val, test


if __name__ == "__main__":

    # Collect all (tif, geojson) pairs by shared stem
    raw_dir = Path("data/raw")
    tifs = sorted(raw_dir.rglob("*.tif"))
    pairs = []
    for t in tifs:
        gj = t.with_suffix(".geojson")
        if gj.exists():
            pairs.append((t, gj))

    # Run tiling
    created = tile_tiffs_with_geojson(
        pairs=pairs,
        out_root=Path("data/tiles_scenes"),
        tile_size=(1024, 1024),  # e.g., 1024px tiles
        overlap=0,  # optional overlap if you need it
        keep_empty_tiles=False,  # skip tiles without labels
        keep_columns=["category"],  # keep specific attributes if you want
    )

    print(f"Total tiles written: {len(created)}")
