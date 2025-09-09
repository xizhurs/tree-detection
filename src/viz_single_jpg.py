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


def _labels_with_conf(
    dets: sv.Detections,
    cat_id_to_name: dict,
    show_conf: bool = True,
    default_name: str = "tree",
):
    """
    Build a label per detection like 'tree 0.87'. Falls back gracefully if
    class ids or confidences are missing (e.g., GT has no conf).
    """
    n = len(dets)
    # Supervision stores arrays: dets.class_id (ints) and dets.confidence (floats)
    class_ids = dets.class_id if dets.class_id is not None else np.zeros(n, dtype=int)
    confs = dets.confidence if dets.confidence is not None else np.full(n, np.nan)

    labels = []
    for cid, conf in zip(class_ids, confs):
        name = cat_id_to_name.get(int(cid), default_name)
        if show_conf and not np.isnan(conf):
            labels.append(f"{name} {conf:.2f}")
        else:
            labels.append(f"{name}")
    return labels


def get_models(
    yolo_weights: str,
    rfdetr_weights: str,
):
    yolo = YOLO(yolo_weights)
    model_rfdetr = RFDETRMedium(pretrain_weights=rfdetr_weights, device="cpu")
    model_rfdetr.optimize_for_inference()
    return yolo, model_rfdetr


def create_annotated_image(
    image, imagesize, detection, cat_id_to_name, color, thickness
):
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=imagesize)

    bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color, text_color=sv.Color.BLACK, text_scale=text_scale
    )
    labels = _labels_with_conf(detection, cat_id_to_name, show_conf=True)
    img_annotated = bbox_annotator.annotate(np.array(image).copy(), detection)
    img_annotated = label_annotator.annotate(img_annotated, detection, labels)
    return img_annotated


def compare_models_on_image(
    image_paths: list,
    coco_json_path: str,
    yolo,
    model_rfdetr,
    conf_yolo: float = 0.25,
    conf_rfdetr: float = 0.5,
):
    # Load image (np array, RGB)
    fig, axs = plt.subplots(
        3, 3, figsize=(21, 21), gridspec_kw={"wspace": 0.05, "hspace": 0.1}
    )
    for row, image_path in enumerate(image_paths):
        image = np.array(Image.open(image_path).convert("RGB"))

        # -------- GT from COCO --------
        gt_det, cat_id_to_name, _ = _gt_detections_from_coco(coco_json_path, image_path)

        # -------- YOLO inference ------
        yres = yolo.predict(source=image_path, conf=conf_yolo, verbose=False)[0]
        yolo_det = sv.Detections.from_ultralytics(yres)

        # Ensure class_id_to_name has YOLO classes if needed
        if yres.names:
            for k, v in yres.names.items():
                if k not in cat_id_to_name:
                    cat_id_to_name[int(k)] = str(v)

        # -------- RF-DETR inference ---
        rfd_det = model_rfdetr.predict(Image.fromarray(image), threshold=conf_rfdetr)

        # -------- Annotate 3 panels ----
        green = sv.Color(46, 204, 113)  # GT
        blue = sv.Color(52, 152, 219)  # YOLO
        red = sv.Color(231, 76, 60)  # RF-DETR
        imagesize = Image.open(image_path).convert("RGB").size
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=imagesize)

        img_gt = create_annotated_image(
            image, imagesize, gt_det, cat_id_to_name, green, thickness
        )
        img_yolo = create_annotated_image(
            image, imagesize, yolo_det, cat_id_to_name, blue, thickness
        )
        img_rfd = create_annotated_image(
            image, imagesize, rfd_det, cat_id_to_name, red, thickness
        )
        # -------- Plot in one figure ----

        axs[row][0].imshow(img_gt)
        axs[row][0].set_title("Ground Truth")
        axs[row][0].axis("off")
        axs[row][1].imshow(img_yolo)
        axs[row][1].set_title("YOLO Predictions (thres=%.2f)" % conf_yolo)
        axs[row][1].axis("off")
        axs[row][2].imshow(img_rfd)
        axs[row][2].set_title("RF-DETR Predictions (thres=%.2f)" % conf_rfdetr)
        axs[row][2].axis("off")
    plt.tight_layout()
    plt.savefig("experiments/results/comparison.png", dpi=500)
    plt.show()


if __name__ == "__main__":
    yolo_weights = "experiments/weights/best.pt"
    rfdetr_weights = "experiments/weights/checkpoint_best_ema.pth"
    yolo, model_rfdetr = get_models(yolo_weights, rfdetr_weights)
    image_paths = list(Path("data/processed_data/coco/test").glob("*.jpg"))
    image_paths = [image_paths[i] for i in [2, 3, 7]]  # select a few images

    compare_models_on_image(
        image_paths=image_paths,
        coco_json_path="data/processed_data/coco/test/_annotations.coco.json",
        yolo=yolo,
        model_rfdetr=model_rfdetr,
        conf_yolo=0.25,
        conf_rfdetr=0.25,
    )
    print("Done.")
