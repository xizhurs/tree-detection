# tree-detection

[![CI](https://github.com/xizhurs/tree-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/xizhurs/tree-detection/actions/workflows/ci.yml)

Tools for preparing tree-detection datasets from geospatial imagery and calling remotely
hosted YOLO or RF-DETR models. Model checkpoints are not stored in this repository and
are not downloaded by the default development or CI environments.

![YOLO and RF-DETR detection comparison](experiments/results/comparison.png)

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

Create the core development environment:

```bash
uv sync
```

Install geospatial functionality without model frameworks:

```bash
uv sync --extra geo
```

The optional local model stack is deliberately separate because it is large:

```bash
uv sync --extra ml
```

`uv.lock` is committed so local and CI installations resolve the same versions.

## Dataset preparation

The Python package exposes helpers for deterministic scene-level splitting, GeoTIFF
tiling, and bbox-only COCO generation:

```python
from pathlib import Path

from tree_detection.coco import build_coco_dataset
from tree_detection.dataset import discover_source_pairs, split_pairs

pairs = discover_source_pairs(Path("data/raw"))
train, validation, test = split_pairs(pairs)
build_coco_dataset(train, Path("data/processed_data/coco/train"))
```

Splitting takes place before tiling so chips from one source scene cannot leak between
training and evaluation sets.

Convert a COCO split to YOLO format with the installed command:

```bash
uv run tree-detection-convert-yolo \
  --input-base-dir data/processed_data/coco \
  --output-dir data/processed_data/yolo \
  --split train
```

## Remote inference

Large custom checkpoints should run on a GPU service such as Modal, RunPod, SageMaker,
Vertex AI, or Azure ML. The repository contains a provider-neutral HTTP client:

```python
import os

from tree_detection.inference import InferenceClient

with InferenceClient(
    os.environ["TREE_DETECTION_ENDPOINT"],
    os.environ.get("TREE_DETECTION_TOKEN"),
) as client:
    predictions = client.predict_file("tile.jpg")
```

See [the inference API contract](docs/inference-api.md) for the request and response
schema. Keep endpoint tokens in environment variables or a secret manager.

## Quality checks

```bash
uv run ruff check src/tree_detection tests
uv run ruff format --check src/tree_detection tests
uv run pytest
uv build
```

Tests create small synthetic rasters and mock HTTP calls. They do not require data,
network access, model weights, PyTorch, or a GPU. GitHub Actions runs linting, tests, and
package builds on Python 3.11, 3.12, and 3.13.

## License

MIT. See [LICENSE](LICENSE).
