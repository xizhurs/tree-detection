import math
import folium
import geopandas as gpd
import numpy as np
import rasterio
from pathlib import Path
from typing import List, Tuple
from shapely.geometry import Polygon, box as shp_box
from shapely.ops import unary_union
from rasterio.windows import Window
from rasterio.transform import Affine
from ultralytics import YOLO
from rfdetr import RFDETRMedium


def _windows_grid(W: int, H: int, tile_w: int, tile_h: int, overlap: int):
    step_w = tile_w - overlap
    step_h = tile_h - overlap
    for r0 in range(0, H, step_h):
        for c0 in range(0, W, step_w):
            w = min(tile_w, W - c0)
            h = min(tile_h, H - r0)
            if w > 0 and h > 0:
                yield Window(c0, r0, w, h)


def _xyxy_pix_to_poly_world(xmin, ymin, xmax, ymax, win_transform: Affine):
    """
    Convert pixel bbox (xyxy in *window-local* pixel coordinates) to a Polygon in the
    raster's CRS using the window transform.
    """
    # corners in pixel coords (col=x, row=y)
    pts_pix = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    # transform to world coords
    pts_world = [~(~win_transform) * (0, 0)]  # dummy to keep Affine around in some envs
    pts_world = [win_transform * (x, y) for (x, y) in pts_pix]
    return Polygon(pts_world)


def _collect_detections_from_yolo(
    result, win_transform: Affine, conf_thr: float = 0.25
):
    geoms, scores, classes = [], [], []
    if result.boxes is None or len(result.boxes) == 0:
        return geoms, scores, classes
    xyxy = result.boxes.xyxy.cpu().numpy()  # (N,4)
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    for (xmin, ymin, xmax, ymax), s, c in zip(xyxy, conf, cls):
        if s < conf_thr:
            continue
        geoms.append(_xyxy_pix_to_poly_world(xmin, ymin, xmax, ymax, win_transform))
        scores.append(float(s))
        classes.append(int(c))
    return geoms, scores, classes


def _collect_detections_from_rfdetr(
    model, tile_rgb, win_transform: Affine, conf_thr: float = 0.25
):
    geoms, scores, classes = [], [], []
    preds = model.predict(tile_rgb, threshold=conf_thr)
    # preds.xyxy: (N,4) in pixel coords relative to tile
    if preds.xyxy is None or len(preds.xyxy) == 0:
        return geoms, scores, classes
    for (xmin, ymin, xmax, ymax), s, c in zip(
        preds.xyxy, preds.confidence, preds.class_id
    ):
        geoms.append(
            _xyxy_pix_to_poly_world(
                float(xmin), float(ymin), float(xmax), float(ymax), win_transform
            )
        )
        scores.append(float(s))
        classes.append(int(c if c is not None else 0))
    return geoms, scores, classes


def predict_tiff_to_geojson(
    tiff_path: str,
    out_dir: str,
    yolo_weights: str,
    rfdetr_weights: str,
    tile_size: Tuple[int, int] = (1024, 1024),
    overlap: int = 128,
    conf_yolo: float = 0.25,
    conf_rfdetr: float = 0.25,
    keep_bands: Tuple[int, int, int] = (1, 2, 3),  # RGB-like bands
):
    """
    Run YOLO and RF-DETR over a GeoTIFF in sliding windows, export detections as GeoJSON (in the TIFF CRS).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    yolo = YOLO(yolo_weights)
    rfd = RFDETRMedium(
        pretrain_weights=rfdetr_weights,
        num_classes=1,
        class_names=["Tree"],
    )
    rfd.optimize_for_inference()

    yolo_rows = []
    rfd_rows = []

    with rasterio.open(tiff_path) as src:
        W, H = src.width, src.height
        crs = src.crs

        for win in _windows_grid(W, H, tile_size[0], tile_size[1], overlap):
            # Read tile as RGB np.uint8
            tile = src.read(
                indexes=keep_bands, window=win, boundless=True, fill_value=0
            )  # [C,H,W]
            tile = np.transpose(tile, (1, 2, 0))  # HWC
            if tile.dtype != np.uint8:
                # normalize each band to 0-255
                t = tile.astype(np.float32)
                t -= t.min(axis=(0, 1), keepdims=True)
                denom = t.max(axis=(0, 1), keepdims=True) + 1e-6
                tile = np.clip((t / denom) * 255.0, 0, 255).astype(np.uint8)

            # Window transform
            win_transform = src.window_transform(win)

            # YOLO predict
            yres = yolo.predict(
                source=tile, imgsz=tile.shape[0], conf=conf_yolo, verbose=False
            )[0]
            y_geoms, y_scores, y_classes = _collect_detections_from_yolo(
                yres, win_transform, conf_yolo
            )
            for g, s, c in zip(y_geoms, y_scores, y_classes):
                yolo_rows.append({"geometry": g, "score": s, "class_id": c})

            # RF-DETR predict (PIL image expected)
            y_geoms2, y_scores2, y_classes2 = _collect_detections_from_rfdetr(
                rfd, tile, win_transform, conf_rfdetr
            )
            for g, s, c in zip(y_geoms2, y_scores2, y_classes2):
                rfd_rows.append({"geometry": g, "score": s, "class_id": c})

    # Build GeoDataFrames in source CRS
    gdf_yolo = (
        gpd.GeoDataFrame(yolo_rows, geometry="geometry", crs=crs)
        if yolo_rows
        else gpd.GeoDataFrame(geometry=[], crs=crs)
    )
    gdf_rfd = (
        gpd.GeoDataFrame(rfd_rows, geometry="geometry", crs=crs)
        if rfd_rows
        else gpd.GeoDataFrame(geometry=[], crs=crs)
    )

    # Save to GeoJSON (still in the TIFF CRS to preserve precision)
    stem = Path(tiff_path).stem
    yolo_gj = out_dir / f"{stem}_yolo.geojson"
    rfd_gj = out_dir / f"{stem}_rfdetr.geojson"
    gdf_yolo.to_file(yolo_gj, driver="GeoJSON")
    gdf_rfd.to_file(rfd_gj, driver="GeoJSON")
    print(f"[OK] Wrote:\n  {yolo_gj}\n  {rfd_gj}")
    return str(yolo_gj), str(rfd_gj)


def make_folium_map(
    gt_geojson_path: str,  # your existing ground-truth GeoJSON (full-scene)
    yolo_geojson_path: str,
    rfd_geojson_path: str,
    out_html: str = "map.html",
):
    # Load vectors, reproject to EPSG:4326 for Folium
    gt = gpd.read_file(gt_geojson_path)
    yolo = gpd.read_file(yolo_geojson_path)
    rfd = gpd.read_file(rfd_geojson_path)

    # Reproject all to WGS84
    gt = gt.to_crs(4326) if gt.crs and gt.crs.to_epsg() != 4326 else gt
    yolo = yolo.to_crs(4326) if yolo.crs and yolo.crs.to_epsg() != 4326 else yolo
    rfd = rfd.to_crs(4326) if rfd.crs and rfd.crs.to_epsg() != 4326 else rfd

    # Map center from GT or predictions
    if len(gt):
        center = [
            gt.geometry.total_bounds[1]
            + (gt.geometry.total_bounds[3] - gt.geometry.total_bounds[1]) / 2,
            gt.geometry.total_bounds[0]
            + (gt.geometry.total_bounds[2] - gt.geometry.total_bounds[0]) / 2,
        ]
    elif len(yolo):
        b = yolo.geometry.total_bounds
        center = [b[1] + (b[3] - b[1]) / 2, b[0] + (b[2] - b[0]) / 2]
    elif len(rfd):
        b = rfd.geometry.total_bounds
        center = [b[1] + (b[3] - b[1]) / 2, b[0] + (b[2] - b[0]) / 2]
    else:
        center = [0, 0]

    m = folium.Map(location=center, zoom_start=18, tiles=None)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=False,
        control=True,
    ).add_to(m)
    # Ground truth layer (green)
    if len(gt):
        folium.GeoJson(
            gt,
            name="Ground Truth",
            style_function=lambda f: {
                "color": "#2ecc71",
                "weight": 2,
                "fillOpacity": 0.1,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in gt.columns if c != "geometry"]
            ),
        ).add_to(m)

    # YOLO predictions (blue)
    if len(yolo):
        folium.GeoJson(
            yolo,
            name="YOLO",
            style_function=lambda f: {
                "color": "#3498db",
                "weight": 2,
                "fillOpacity": 0.05,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in yolo.columns if c != "geometry"]
            ),
        ).add_to(m)

    # RF-DETR predictions (red)
    if len(rfd):
        folium.GeoJson(
            rfd,
            name="RF-DETR",
            style_function=lambda f: {
                "color": "#e74c3c",
                "weight": 2,
                "fillOpacity": 0.05,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in rfd.columns if c != "geometry"]
            ),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(out_html)
    print(f"[OK] Folium map saved to {out_html}")


if __name__ == "__main__":

    # 1) Run inference over a GeoTIFF and create GeoJSONs of predictions
    yolo_gj, rfd_gj = predict_tiff_to_geojson(
        tiff_path="data/tiles_scenes/clip1_colombia_r2048_c1024_w1024_h185.tif",
        out_dir="data/preds",
        yolo_weights="experiments/best.pt",
        rfdetr_weights="experiments/checkpoint_best_ema.pth",
        tile_size=(1024, 1024),
        overlap=128,
        conf_yolo=0.25,
        conf_rfdetr=0.25,
        keep_bands=(1, 2, 3),  # adjust if your RGB bands are different
    )

    # 2) Build a Folium map with GT and predictions (GT is your original scene-level GeoJSON)
    make_folium_map(
        gt_geojson_path="data/tiles_scenes/clip1_colombia_r2048_c1024_w1024_h185.geojson",
        yolo_geojson_path=yolo_gj,
        rfd_geojson_path=rfd_gj,
        out_html="scene_A_map.html",
    )
