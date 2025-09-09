import argparse
import json
import os
import shutil
from collections import defaultdict
from tqdm import tqdm


def convert_coco_to_yolo(input_base_dir, output_dir, split="train"):
    json_path = os.path.join(input_base_dir, f"{split}/_annotations.coco.json")
    images_dir = os.path.join(input_base_dir, f"{split}")

    out_img_dir = os.path.join(output_dir, f"{split}", "images")
    out_lbl_dir = os.path.join(output_dir, f"{split}", "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    with open(json_path, "r") as f:
        coco = json.load(f)
    image_by_id = {img["id"]: img for img in coco["images"]}

    # COCO categories can be non-contiguous; build a 0..K-1 mapping
    cat_ids = [c["id"] for c in coco["categories"]]
    cat_ids_sorted = sorted(cat_ids)  # or keep input order if you prefer
    catid_to_yolo = {cid: i for i, cid in enumerate(cat_ids_sorted)}

    # 2) Collect YOLO lines per image
    yolo_lines = defaultdict(list)
    for ann in tqdm(coco["annotations"], desc="Converting"):
        if ann.get("iscrowd", 0) == 1:
            continue  # skip crowd if present

        img = image_by_id.get(ann["image_id"])
        if img is None:
            continue

        W, H = img["width"], img["height"]
        x, y, w, h = ann["bbox"]  # COCO xywh in pixels

        # Convert to YOLO (x_center, y_center, w, h) normalized to [0,1]
        x_c = (x + w / 2.0) / W
        y_c = (y + h / 2.0) / H
        nw = w / W
        nh = h / H

        # Clamp to be safe
        x_c = min(max(x_c, 0.0), 1.0)
        y_c = min(max(y_c, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)

        cls = catid_to_yolo[ann["category_id"]]
        yolo_lines[ann["image_id"]].append(
            f"{cls} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}"
        )

    for img in tqdm(coco["images"], desc="Writing labels + copying images"):
        img_id = img["id"]
        file_src = os.path.join(images_dir, img["file_name"])
        # mirror the exact filename; YOLO wants .txt with same stem
        stem, _ = os.path.splitext(os.path.basename(img["file_name"]))
        lbl_path = os.path.join(out_lbl_dir, stem + ".txt")

        # write all lines for this image (or create empty file)
        lines = yolo_lines.get(img_id, [])
        with open(lbl_path, "w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        # copy image
        file_dst = os.path.join(out_img_dir, os.path.basename(img["file_name"]))
        if not os.path.exists(file_dst):
            shutil.copyfile(file_src, file_dst)

    print("✅ COCO → YOLO conversion complete.")
    print(f"Images: {out_img_dir}")
    print(f"Labels: {out_lbl_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_base_dir",
        type=str,
        default="data/processed_data/coco",
        help="Base directory of COCO dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_data/yolo",
        help="Output directory for YOLO formatted data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to convert (train/valid/test)",
    )
    args = parser.parse_args()

    convert_coco_to_yolo(args.input_base_dir, args.output_dir, args.split)
