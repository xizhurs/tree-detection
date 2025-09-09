import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import supervision as sv
from ultralytics import YOLO
from rfdetr import RFDETRMedium


def _coco_xywh_to_xyxy(b):
    # COCO bbox is [x, y, w, h] → xyxy
    x, y, w, h = b
    return [x, y, x + w, y + h]


def _gt_detections_from_coco(coco_json_path: str, image_path: str):
    """
    Returns (detections: sv.Detections, class_id_to_name: dict, image_id: int)
    for a single image.
    """
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # map categories
    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

    # find image entry by basename match
    fname = Path(image_path).name
    images = coco["images"]
    img_entry = next((im for im in images if Path(im["file_name"]).name == fname), None)
    if img_entry is None:
        raise ValueError(f"Image {fname} not found in COCO JSON.")

    img_id = img_entry["id"]
    anns = [a for a in coco["annotations"] if a["image_id"] == img_id]

    if len(anns) == 0:
        # empty GT
        return sv.Detections.empty(), cat_id_to_name, img_id

    xyxy = np.array([_coco_xywh_to_xyxy(a["bbox"]) for a in anns], dtype=float)
    class_id = np.array([a["category_id"] for a in anns], dtype=int)

    det = sv.Detections(xyxy=xyxy, class_id=class_id)  # no confidence for GT
    return det, cat_id_to_name, img_id


def _labels_for_detections(
    detections: sv.Detections, class_id_to_name: dict, show_conf: bool = True
):
    labels = []
    for i in range(len(detections)):
        cid = int(detections.class_id[i]) if detections.class_id is not None else 0
        name = class_id_to_name.get(cid, str(cid))
        if show_conf and detections.confidence is not None:
            conf = float(detections.confidence[i])
            labels.append(f"{name} {conf:.2f}")
        else:
            labels.append(name)
    return labels


def compare_models_on_image(
    image_path: Path,
    coco_json_path: str,
    yolo_weights: str,
    rfdetr_weights: str,
    conf_yolo: float = 0.25,
    conf_rfdetr: float = 0.5,
):
    # Load image (np array, RGB)
    image = np.array(Image.open(image_path).convert("RGB"))

    # -------- GT from COCO --------
    gt_det, cat_id_to_name, _ = _gt_detections_from_coco(coco_json_path, image_path)

    # -------- YOLO inference ------
    yolo = YOLO(yolo_weights)
    yres = yolo.predict(source=image_path, conf=conf_yolo, verbose=False)[0]
    yolo_det = sv.Detections.from_ultralytics(yres)

    # Ensure class_id_to_name has YOLO classes if needed
    if yres.names:
        for k, v in yres.names.items():
            if k not in cat_id_to_name:
                cat_id_to_name[int(k)] = str(v)

    # -------- RF-DETR inference ---
    model_rfdetr = RFDETRMedium(pretrain_weights=rfdetr_weights, device="cpu")
    model_rfdetr.optimize_for_inference()
    rfd_det = model_rfdetr.predict(Image.fromarray(image), threshold=conf_rfdetr)

    # -------- Annotate 3 panels ----
    green = sv.Color(46, 204, 113)  # GT
    blue = sv.Color(52, 152, 219)  # YOLO
    red = sv.Color(231, 76, 60)  # RF-DETR

    box_gt = sv.BoxAnnotator(thickness=2, color=green)
    box_yolo = sv.BoxAnnotator(thickness=2, color=blue)
    box_rfd = sv.BoxAnnotator(thickness=2, color=red)

    gt_labels = _labels_for_detections(gt_det, cat_id_to_name, show_conf=False)
    yolo_labels = _labels_for_detections(yolo_det, cat_id_to_name, show_conf=True)
    rfd_labels = _labels_for_detections(rfd_det, cat_id_to_name, show_conf=True)

    img_gt = box_gt.annotate(image.copy(), gt_det, gt_labels)
    img_yolo = box_yolo.annotate(image.copy(), yolo_det, yolo_labels)
    img_rfd = box_rfd.annotate(image.copy(), rfd_det, rfd_labels)

    # -------- Plot in one figure ----
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    axs[0].imshow(img_gt)
    axs[0].set_title("Ground Truth")
    axs[0].axis("off")
    axs[1].imshow(img_yolo)
    axs[1].set_title("YOLO Predictions")
    axs[1].axis("off")
    axs[2].imshow(img_rfd)
    axs[2].set_title("RF-DETR Predictions")
    axs[2].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_paths = Path("data/processed_data/coco/test").glob("*.jpg")
    image_path = next(image_paths)  # take the first image for demo
    compare_models_on_image(
        image_path=image_path,
        coco_json_path="data/processed_data/coco/test/_annotations.coco.json",
        yolo_weights="experiments/best.pt",
        rfdetr_weights="experiments/checkpoint_best_ema.pth",
        conf_yolo=0.5,
        conf_rfdetr=0.5,
    )
    print("Done.")
