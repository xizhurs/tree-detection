import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tree_detection.training import load_config, train_rfdetr, train_yolo


def test_load_config_requires_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)


def test_train_yolo_uses_lazy_model_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class FakeModel:
        def __init__(self, weights: str) -> None:
            calls["weights"] = weights

        def train(self, **kwargs: object) -> None:
            calls["train"] = kwargs

        def val(self) -> None:
            calls["validated"] = True

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    config = tmp_path / "yolo.yaml"
    config.write_text(
        """
dataset: data.yaml
training:
  epochs: 1
  batch_size: 2
  image_size: 64
  weight_decay: 0.01
model:
  weights: checkpoint.pt
output:
  project: runs
  name: test
augmentation:
  fliplr: 0.5
""",
        encoding="utf-8",
    )

    train_yolo(config)

    assert calls["weights"] == "checkpoint.pt"
    assert calls["validated"] is True
    assert calls["train"] == {
        "data": "data.yaml",
        "project": "runs",
        "name": "test",
        "imgsz": 64,
        "epochs": 1,
        "batch": 2,
        "device": None,
        "weight_decay": 0.01,
        "fliplr": 0.5,
    }


def test_train_rfdetr_uses_lazy_model_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class FakeModel:
        def train(self, **kwargs: object) -> None:
            calls["train"] = kwargs

    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRMedium=FakeModel))
    config = tmp_path / "rfdetr.yaml"
    config.write_text(
        """
dataset:
  path: data/coco
  num_classes: 1
training:
  epochs: 1
  batch_size: 2
  learning_rate: 0.0001
  grad_accum_steps: 1
  weight_decay: 0.01
  num_workers: 0
  resolution: 64
  early_stopping: true
output:
  dir: runs
settings:
  distributed: false
  use_ema: true
  do_benchmark: false
""",
        encoding="utf-8",
    )

    train_rfdetr(config)

    assert calls["train"] == {
        "dataset_dir": "data/coco",
        "num_classes": 1,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.0001,
        "grad_accum_steps": 1,
        "weight_decay": 0.01,
        "num_workers": 0,
        "resolution": 64,
        "early_stopping": True,
        "output_dir": "runs",
        "distributed": False,
        "use_ema": True,
        "do_benchmark": False,
    }
