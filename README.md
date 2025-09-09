# tree-detection

Single-tree detection in high‑resolution aerial/satellite imagery. This repo compares **Ultralytics YOLOv12** and a **transformer-based detector (RF‑DETR)**, and includes **geo‑aware inference** utilities so you can put detections back on a map (Folium/Leaflet) with the correct CRS.

> **Status:** Work in progress. Please open issues/PRs for bugs or ideas!

---

## 🔍 What’s inside

- **Detectors**
  - **YOLOv12**: fast to train and deploy.
  - **RF‑DETR**: DETR‑family transformer, good for small objects with proper training.
- **Geo‑aware inference**
  - Convert pixel-space detections → **CRS coordinates** using the source GeoTIFF transform.
  - Create **interactive Folium maps** for QA.
- **Data & tiling**
  - Train with **YOLO** or **COCO** format.
  - Tiling helpers for big GeoTIFFs (chip size + overlap).
- **Notebooks**
  - End‑to‑end examples: training → inference → mapping.


```
configs/          # model & data configs
notebooks/        # training / inference / mapping notebooks
src/              # helpers: tiling, geo utils, converters
README.md
LICENSE
```

---

## 🚀 Quick start

### 1) Environment

```
pip install -r requirements.txt
```

### 2) Data formats

You can use **YOLO** or **COCO**. Keep a **separate, non‑overlapping test region** when possible.

**YOLO format**

```
<dataset_root>/
  images/
    train/ *.jpg|*.png
    val/   *.jpg|*.png
    test/  *.jpg|*.png
  labels/
    train/ *.txt  # class x_center y_center width height  (normalized)
    val/   *.txt
    test/  *.txt
  data.yaml        # names, nc, and split paths
```

**COCO format**

```
<dataset_root>/
  annotations/
    instances_train.json
    instances_val.json
  images/
    train/
    val/
```

If you start from **GeoTIFF + vector labels (GeoJSON/SHP/COCO)**, first **tile** them into chips (e.g., `imgsz=1024`, overlap 200–400 px). See `notebooks/` and `src/` helpers.

### 3) Train — YOLOv11

**Python**

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # or your checkpoint
model.train(
    data="/path/to/data.yaml",
    imgsz=1024,
    epochs=100,
    batch=16,
    project="runs/tree_yolo",
    name="exp",
)
```

### 4) Train — RF‑DETR (COCO)

```python
from rfdetr import RFDETRMedium

# NOTE: set num_classes to YOUR dataset (e.g., 1 for 'tree')
model = RFDETRMedium(pretrain_weights="/path/to/checkpoint.pth")

# Pseudocode:
# model.fit(train_loader, val_loader, epochs=100, ...)
```

**Common pitfall:** pretrain head vs dataset classes. If you see a warning like *“num_classes mismatch… reinitializing detection head”*, ensure your model config uses the correct `num_classes` (e.g., 1). Loading only backbone weights is also fine.

---

## 🗺️ Geo‑aware inference & Folium

Convert pixel boxes to CRS coordinates using the GeoTIFF affine transform.

```python
from pathlib import Path
import folium, rasterio, geopandas as gpd
from shapely.geometry import box, mapping

src_path = "path/to/ortho.tif"
with rasterio.open(src_path) as src:
    transform = src.transform
    crs = src.crs

# example predictions (x1,y1,x2,y2 in pixels)
preds = [{"score": 0.91, "bbox": [120, 340, 170, 395]}]

def pixpoly_to_geo(poly, transform):
    xs, ys = poly.exterior.xy
    xs_t, ys_t = [], []
    for cx, cy in zip(xs, ys):
        X, Y = rasterio.transform.xy(transform, cy, cx)
        xs_t.append(X); ys_t.append(Y)
    return box(min(xs_t), min(ys_t), max(xs_t), max(ys_t))

geoms = []
for p in preds:
    poly_pix = box(*p["bbox"])
    poly_geo = pixpoly_to_geo(poly_pix, transform)
    geoms.append({"geometry": poly_geo, "score": p["score"]})

gdf = gpd.GeoDataFrame(geoms, geometry="geometry", crs=crs)

center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
m = folium.Map(location=center, zoom_start=18, tiles="OpenStreetMap")
for _, r in gdf.iterrows():
    folium.GeoJson(mapping(r.geometry), name=f"score={r['score']:.2f}").add_to(m)
m.save("detections_map.html")
print("Saved → detections_map.html")
```

> **Google/ESRI basemaps:** you can add third‑party tile endpoints (may require API keys/licenses).

---

## ✅ Evaluation tips

- **YOLO:** use built‑in `val/test` metrics (mAP@50/95, Precision/Recall).
- **RF‑DETR:** evaluate with COCO metrics (e.g., pycocotools or `supervision.MeanAveragePrecision`).
- Report size‑stratified AP for **small trees**.
- Prefer larger `imgsz` (e.g., **1024**) for 10–30 cm GSD imagery.
- Tile with **overlap** (200–400 px) to avoid edge misses.

---

```markdown
## 📊 Results

YOLOv11 vs RF‑DETR

![detections](experiments/results/comparison.png)
```

```html
<p float="left">
  <img src="experiments/results/comparison.png" width="49%" />
</p>
```
---

## 🧰 Utilities (planned/available)

- COCO ↔ YOLO converters.
- Tiling from GeoTIFF + vector labels; inverse mapping of detections.
- Post‑processing (NMS, score thresholds, center‑point export).

---

## 📒 Notebooks

- `notebooks/train_yolo11.ipynb` — YOLO training.
- `notebooks/train_rfdetr.ipynb` — RF‑DETR training.
- `notebooks/infer_and_map.ipynb` — inference + Folium map.

---

## 📦 Repro & configs

- Keep experiment configs in `configs/` (dataset paths, chip size, overlap, model hyperparams).
- Track runs with Ultralytics logs or MLflow; export ONNX if needed.

---

## 🧭 Roadmap

- [ ] End‑to‑end **tiling + label‑export** scripts in `src/`.
- [ ] **COCO↔YOLO** converters and tests.
- [ ] Minimal **pretrained checkpoint** for demo.
- [ ] **Google/ESRI basemap** examples for Folium.
- [ ] Benchmark **YOLO vs RF‑DETR** across chip sizes & size bins.

---

## 📜 License

MIT — see `LICENSE`.

---

## 🙏 Acknowledgements

- Ultralytics YOLO
- RF‑DETR
- Rasterio, GeoPandas, Shapely, Folium, Supervision
