"""Optional local training entry points with lazy model imports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("training configuration must be a YAML mapping")
    return config


def _missing_extra(error: ImportError) -> RuntimeError:
    return RuntimeError(
        "local model dependencies are not installed; run `uv sync --extra ml`"
    )


def train_yolo(config_path: str | Path) -> None:
    """Train and validate YOLO from a repository configuration file."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise _missing_extra(error) from error

    config = load_config(config_path)
    training = config["training"]
    model_config = config["model"]
    output = config["output"]
    model = YOLO(model_config["weights"])
    model.train(
        data=config["dataset"],
        project=output["project"],
        name=output["name"],
        imgsz=training["image_size"],
        epochs=training["epochs"],
        batch=training["batch_size"],
        device=model_config.get("device"),
        weight_decay=training["weight_decay"],
        **config.get("augmentation", {}),
    )
    model.val()


def train_rfdetr(config_path: str | Path) -> None:
    """Train RF-DETR from a repository configuration file."""
    try:
        from rfdetr import RFDETRMedium
    except ImportError as error:
        raise _missing_extra(error) from error

    config = load_config(config_path)
    training = config["training"]
    model = RFDETRMedium()
    model.train(
        dataset_dir=config["dataset"]["path"],
        num_classes=config["dataset"]["num_classes"],
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        learning_rate=training["learning_rate"],
        grad_accum_steps=training["grad_accum_steps"],
        weight_decay=training["weight_decay"],
        num_workers=training["num_workers"],
        resolution=training["resolution"],
        early_stopping=training["early_stopping"],
        output_dir=config["output"]["dir"],
        distributed=config["settings"]["distributed"],
        use_ema=config["settings"]["use_ema"],
        do_benchmark=config["settings"]["do_benchmark"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("yolo", "rfdetr"))
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    if args.model == "yolo":
        train_yolo(args.config)
    else:
        train_rfdetr(args.config)


if __name__ == "__main__":
    main()
