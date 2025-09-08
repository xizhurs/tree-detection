from rfdetr import RFDETRMedium
from ultralytics import YOLO


def parse_rfdetr_config(config_path):
    """
    Parse the RFDETR configuration file to extract training parameters.
    """
    import yaml

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def train_rfdetr():
    model = RFDETRMedium()
    config = parse_rfdetr_config("configs/rfdetr.yaml")
    model.train(
        dataset_dir=config["dataset"]["path"],
        num_classes=config["dataset"]["num_classes"],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        grad_accum_steps=config["training"]["grad_accum_steps"],
        weight_decay=config["training"]["weight_decay"],
        num_workers=config["training"]["num_workers"],
        resolution=config["training"]["resolution"],
        early_stopping=config["training"]["early_stopping"],
        output_dir=config["output"]["dir"],
        name="rfdetr-results",
        distributed=config["settings"]["distributed"],
        use_ema=config["settings"]["use_ema"],
        do_benchmark=config["settings"]["do_benchmark"],
    )


def parse_yolo_config(config_path):
    """
    Parse the YOLO configuration file to extract training parameters.
    """
    import yaml

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def train_yolo():
    """
    Train a YOLO model using the ultralytics library.
    """
    config = parse_yolo_config("configs/yolo.yaml")

    model = YOLO(config["Model configuration"]["model"])  # Using YOLOv12 nano model

    # Train the model with specified parameters
    model.train(
        data="configs/yolo.yaml",  # Path to dataset configuration file
        project=config["Model configuration"]["project"],
        exist_ok=True,  # Overwrite existing results
        name=config["Model configuration"]["name"],
        imgsz=config["Training parameters"]["imgsz"],
        epochs=config["Training parameters"]["epochs"],
        batch=config["Training parameters"]["batch_size"],
        device=config["Model configuration"]["device"],
        hsv_h=config["Augmentation"]["hsv_h"],
        hsv_s=config["Augmentation"]["hsv_s"],
        hsv_v=config["Augmentation"]["hsv_v"],
        degrees=config["Augmentation"]["degrees"],
        translate=config["Augmentation"]["translate"],
        scale=config["Augmentation"]["scale"],
        shear=config["Augmentation"]["shear"],
        perspective=config["Augmentation"]["perspective"],
        flipud=config["Augmentation"]["flipud"],
        fliplr=config["Augmentation"]["fliplr"],
        mosaic=config["Augmentation"]["mosaic"],
        mixup=config["Augmentation"]["mixup"],
        weight_decay=config["Training parameters"]["weight_decay"],
    )

    # Optionally, evaluate the model performance on the validation set
    model.val()


if __name__ == "__main__":
    train_yolo()
