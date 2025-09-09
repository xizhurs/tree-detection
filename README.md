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

> **Google/ESRI basemaps:** you can add third‑party tile endpoints (may require API keys/licenses).

---

## ✅ Evaluation tips

- **YOLO:** use built‑in `val/test` metrics (mAP@50/95, Precision/Recall).
- **RF‑DETR:** evaluate with COCO metrics (e.g., pycocotools or `supervision.MeanAveragePrecision`).
- Report size‑stratified AP for **small trees**.
- Prefer larger `imgsz` (e.g., **1024**) for 10–30 cm GSD imagery.
- Tile with **overlap** (200–400 px) to avoid edge misses.

---

## 📊 Example Results

YOLOv11 vs RF‑DETR

![detections](experiments/results/comparison.png)

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
