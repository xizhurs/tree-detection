from rfdetr import RFDETRMedium
from ultralytics import YOLO


def train_rfdetr():
    model = RFDETRMedium()
    model.train(
        dataset_dir="data/tiles/coco",
        encoder="dinov2_windowed_base",
        epochs=10,
        batch_size=4,
        grad_accum_steps=4,
        num_classes=1,
        class_names=["Tree"],
        weight_decay=1e-1,
        num_workers=0,
        distributed=False,
        use_ema=True,
        do_benchmark=False,
        resolution=512,
        output_dir="experiments",
        # device="cuda",
        backbone_lora=False,
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
    # Load a pre-trained YOLO model (you can choose different sizes: 'n', 's', 'm', 'l', 'x')
    model = YOLO("yolo12n.pt")  # Using YOLOv12 nano model
    config = parse_yolo_config("configs/yolo.yaml")

    # Train the model with specified parameters
    model.train(
        data="configs/yolo.yaml",  # Path to dataset configuration file
        name="yolo12n-results",  # Name for the training results directory
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
    )

    # Optionally, evaluate the model performance on the validation set
    model.val()


if __name__ == "__main__":
    train_yolo()
