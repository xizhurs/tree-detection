import os, json, math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import geopandas as gpd
from shapely.geometry import box, MultiPolygon
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.transform import array_bounds, rowcol
import cv2
from shapely.geometry import mapping
from dataclasses import dataclass
import random


@dataclass
class SourcePair:
    raster_path: str
    geojson_path: str


class GeoCocoBuilder:
    """
    Multi-source TIFF + GeoJSON -> COCO (tiled) for DETECTION (bbox-only).
    Saves images and annotations JSON in the SAME folder: out_dir/.
    """

    def __init__(
        self,
        out_dir: str,
        tile_size: Tuple[int, int] = (512, 512),
        overlap: int = 128,
        class_id_attr: Optional[str] = None,  # e.g. "class_id"
        class_name_attr: Optional[str] = None,  # e.g. "category"
        jpeg_quality: int = 95,
        keep_empty_tiles: bool = False,
        seed: int = 42,
    ):
        self.out_dir = Path(out_dir)
        self.images_dir = self.out_dir  # <-- same directory
        self.ann_dir = self.out_dir  # <-- same directory
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.ann_dir.mkdir(parents=True, exist_ok=True)

        self.tw, self.th = tile_size
        self.overlap = overlap
        self.class_id_attr = class_id_attr
        self.class_name_attr = class_name_attr
        self.jpeg_quality = jpeg_quality
        self.keep_empty_tiles = keep_empty_tiles
        self.rng = random.Random(seed)

        # COCO accumulators
        self.images: List[Dict] = []
        self.annotations: List[Dict] = []
        self.categories: List[Dict] = []
        self._next_image_id = 1
        self._next_ann_id = 1

        # category bookkeeping
        self._cat_name_to_id: Dict[str, int] = {}  # name -> contiguous id
        self._rawid_to_contig: Dict[int, int] = {}  # original id -> contiguous id

    def _ensure_categories(self, gdf: gpd.GeoDataFrame):
        if self.class_id_attr and self.class_id_attr in gdf.columns:
            raw_ids = [
                int(x) for x in gdf[self.class_id_attr].dropna().unique().tolist()
            ]
            for rid in sorted(raw_ids):
                if rid not in self._rawid_to_contig:
                    cid = len(self._rawid_to_contig)
                    self._rawid_to_contig[rid] = cid
                    name = str(rid)
                    if self.class_name_attr and self.class_name_attr in gdf.columns:
                        names = gdf.loc[
                            gdf[self.class_id_attr] == rid, self.class_name_attr
                        ].dropna()
                        if len(names):
                            name = str(names.iloc[0])
                    self.categories.append(
                        {"id": cid, "name": name, "supercategory": ""}
                    )
        else:
            if self.class_name_attr and self.class_name_attr in gdf.columns:
                names = sorted(
                    [
                        str(x)
                        for x in gdf[self.class_name_attr].dropna().unique().tolist()
                    ]
                )
            else:
                names = ["tree"]
            for n in names:
                if n not in self._cat_name_to_id:
                    cid = len(self._cat_name_to_id)
                    self._cat_name_to_id[n] = cid
                    self.categories.append({"id": cid, "name": n, "supercategory": ""})

    def _tile_offsets(self, parent: Window) -> List[Tuple[int, int]]:
        step_w = max(1, self.tw - self.overlap)
        step_h = max(1, self.th - self.overlap)
        cols = [
            max(0, int(parent.col_off)) + i * step_w
            for i in range(max(1, math.ceil((parent.width - self.overlap) / step_w)))
        ]
        rows = [
            max(0, int(parent.row_off)) + j * step_h
            for j in range(max(1, math.ceil((parent.height - self.overlap) / step_h)))
        ]
        return [(c, r) for r in rows for c in cols]

    @staticmethod
    def _bbox_from_geom_bounds(inter_geom, w_transform, W, H) -> Optional[List[int]]:
        if inter_geom.is_empty:
            return None
        minx, miny, maxx, maxy = inter_geom.bounds
        r0, c0 = rowcol(w_transform, minx, maxy)  # top-left (y=maxy)
        r1, c1 = rowcol(w_transform, maxx, miny)  # bottom-right (y=miny)
        xmin = max(0, min(c0, c1))
        xmax = min(W - 1, max(c0, c1))
        ymin = max(0, min(r0, r1))
        ymax = min(H - 1, max(r0, r1))
        if xmax < xmin or ymax < ymin:
            return None
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        if w <= 0 or h <= 0:
            return None
        return [int(xmin), int(ymin), int(w), int(h)]

    def add_source(self, raster_path: Path, geojson_path: Path):
        with rasterio.open(raster_path) as src:
            gdf = gpd.read_file(geojson_path)
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            self._ensure_categories(gdf)

            # map to contiguous category ids
            if self.class_id_attr and self.class_id_attr in gdf.columns:
                gdf["_cat"] = gdf[self.class_id_attr].map(self._rawid_to_contig)
            else:
                if self.class_name_attr and self.class_name_attr in gdf.columns:
                    gdf["_cat"] = gdf[self.class_name_attr].map(self._cat_name_to_id)
                else:
                    gdf["_cat"] = 0  # single class

            # bounds intersection
            rb = src.bounds
            raster_poly = box(rb.left, rb.bottom, rb.right, rb.top)
            vec_poly = gdf.unary_union.envelope
            if not raster_poly.intersects(vec_poly):
                print(f"[skip] No overlap: {raster_path} vs {geojson_path}")
                return
            inter = raster_poly.intersection(vec_poly).envelope
            parent = from_bounds(*inter.bounds, transform=src.transform)

            tiles_total, tiles_kept, anns_added = 0, 0, 0
            for c_off, r_off in self._tile_offsets(parent):
                tiles_total += 1
                child = Window(
                    col_off=int(c_off),
                    row_off=int(r_off),
                    width=self.tw,
                    height=self.th,
                )
                w_transform = src.window_transform(child)
                w_bounds = array_bounds(self.th, self.tw, w_transform)
                w_poly = box(*w_bounds)

                # intersect labels
                mask_idx = gdf.intersects(w_poly)
                if not mask_idx.any() and not self.keep_empty_tiles:
                    continue

                # read tile
                img = src.read(window=child, boundless=True, fill_value=src.nodata or 0)
                if img.dtype != np.uint8:
                    img = cv2.normalize(
                        img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
                    )
                if img.shape[0] >= 3:
                    rgb = np.stack([img[0], img[1], img[2]], axis=2)
                else:
                    rgb = np.repeat(img[0:1].transpose(1, 2, 0), 3, axis=2)

                fname = f"{self._next_image_id}_{int(c_off)}_{int(r_off)}_{self.tw}_{self.th}.jpg"
                fpath = self.images_dir / fname
                cv2.imwrite(
                    str(fpath),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )

                # image entry (basename only, since same folder)
                self.images.append(
                    {
                        "id": self._next_image_id,
                        "file_name": str(fname),
                        "width": self.tw,
                        "height": self.th,
                    }
                )
                tiles_kept += 1

                # bbox-only annotations
                if mask_idx.any():
                    for _, row in gdf[mask_idx].sort_values("_cat").iterrows():
                        inter_geom = row.geometry.intersection(w_poly)
                        if inter_geom.is_empty:
                            continue
                        bbox = self._bbox_from_geom_bounds(
                            inter_geom, w_transform, self.tw, self.th
                        )
                        if bbox is None:
                            continue
                        area = float(bbox[2] * bbox[3])
                        self.annotations.append(
                            {
                                "id": self._next_ann_id,
                                "image_id": self._next_image_id,
                                "category_id": int(row["_cat"]),
                                "bbox": bbox,
                                "area": area,
                                "iscrowd": (
                                    1 if isinstance(row.geometry, MultiPolygon) else 0
                                ),
                            }
                        )
                        self._next_ann_id += 1
                        anns_added += 1

                self._next_image_id += 1

            print(
                f"[add_source] {Path(raster_path).name}: tiles_total={tiles_total}, "
                f"tiles_kept={tiles_kept}, anns_added={anns_added}"
            )

    def _coco_dict(self) -> Dict:
        return {
            "info": {
                "description": "GeoTIFF+GeoJSON to COCO (bbox-only)",
                "version": "1.0",
            },
            "licenses": [],
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def save_coco(self, filename: str = "_annotations.coco.json"):
        out_json = self.ann_dir / filename
        with open(out_json, "w") as f:
            json.dump(self._coco_dict(), f)
        print(f"COCO saved -> {out_json}")


def convert_to_coco(pairs: List[Tuple[Path, Path]], split: str = "train"):
    builder = GeoCocoBuilder(
        out_dir=f"data/tiles/coco/{split}",
        tile_size=(512, 512),
        overlap=128,
        class_id_attr=None,
        class_name_attr="category",
        jpeg_quality=95,
        keep_empty_tiles=False,
    )
    for tif, gj in pairs:
        builder.add_source(tif, gj)

    # # Save one COCO, or split into train/val/test JSONs
    builder.save_coco("_annotations.coco.json")


if __name__ == "__main__":

    paths_tif = list(Path("data/raw").rglob("*.tif"))
    paths_gj = [p.with_suffix(".geojson") for p in paths_tif]
    # Add many sources
    pairs = [(tif, gj) for tif, gj in zip(paths_tif, paths_gj) if os.path.exists(gj)]
    train_pairs = [
        (x, y) for i, (x, y) in enumerate(pairs) if "1" in x.stem or "4" in x.stem
    ]
    val_pairs = [(x, y) for i, (x, y) in enumerate(pairs) if "5" in x.stem]
    test_pairs = [(x, y) for i, (x, y) in enumerate(pairs) if "2" in x.stem]
    convert_to_coco(train_pairs, split="train")
    convert_to_coco(val_pairs, split="valid")
    convert_to_coco(test_pairs, split="test")
    print("done")
