# Deploying the detector on Modal

`deploy/modal_app.py` serves the fine-tuned RF-DETR Medium checkpoint on a Modal GPU
behind the [inference API contract](inference-api.md), so the existing
`tree_detection.inference.InferenceClient` works against it unchanged.

## How the pieces fit

| Modal concept | Used for |
| --- | --- |
| `modal.Image` | Container with torch, rfdetr and FastAPI pinned to `uv.lock`, plus the local `tree_detection` package. Built once and cached. |
| `modal.Volume` (`tree-detection-weights`) | Holds the checkpoint, separate from code. Retraining only needs a new upload, no rebuild. |
| `modal.Secret` (`tree-detection-api-token`) | Injects `TREE_DETECTION_TOKEN`, the bearer token the endpoint checks. |
| `@app.cls(gpu="T4")` | The GPU container. `@modal.enter()` loads the model once per container, not per request. |
| `@modal.method()` | Python RPC entry point, used by `modal run` for smoke tests. |
| `@modal.asgi_app()` | FastAPI app with `GET /health` and `POST /predict`. |

Containers scale to zero when idle. `scaledown_window=300` keeps one warm for five
minutes after the last request, and `max_containers=2` caps spend.

## One-time setup

```bash
uv sync --group deploy             # installs the modal CLI into the project venv
uv run modal setup                 # browser login, stores a token in ~/.modal.toml
```

GPU functions need a payment method on the Modal account. The monthly free credits
still apply, but until a card is added Modal rejects `gpu=...` with
`Please add a payment method to use T4 GPU functions`. Set `TREE_DETECTION_GPU=cpu`
to run without a GPU in the meantime.

Upload the checkpoint, then create the token secret:

```bash
uv run modal volume put tree-detection-weights \
    experiments/weights/checkpoint_best_ema.pth /rfdetr/checkpoint_best_ema.pth
uv run modal volume ls tree-detection-weights /rfdetr

uv run modal secret create tree-detection-api-token \
    TREE_DETECTION_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

To ship a retrained model, run the same `volume put` again with `--force`. New
containers pick it up. Run `modal app stop tree-detection` to recycle warm ones.

## Run, serve, deploy

```bash
# 1. RPC smoke test: builds the image, starts a container, prints detections.
uv run modal run deploy/modal_app.py --image tile.jpg

# 2. Temporary HTTPS URL that lives while the command runs (*-dev.modal.run).
uv run modal serve deploy/modal_app.py

# 3. Persistent URL: https://<workspace>--tree-detection-treedetector-web.modal.run
uv run modal deploy deploy/modal_app.py
```

Pick the hardware at deploy time with `TREE_DETECTION_GPU` (`T4` by default; for
example `L4`, `A10G`, or `cpu`).

## Call the endpoint

```bash
curl -H "Authorization: Bearer $TREE_DETECTION_TOKEN" \
     -F image=@tile.jpg "$TREE_DETECTION_ENDPOINT/predict?threshold=0.25"
```

```python
import os

from tree_detection.inference import InferenceClient

with InferenceClient(
    os.environ["TREE_DETECTION_ENDPOINT"], os.environ["TREE_DETECTION_TOKEN"]
) as client:
    predictions = client.predict_file("tile.jpg")
```

The endpoint returns `401` for a missing or wrong token, `400` for an undecodable
image, and `422` for a `threshold` outside `[0, 1]`.

## Operating it

- **Logs:** `uv run modal app logs tree-detection`, or the app page on modal.com.
  When a container fails in `@modal.enter()`, the client usually sees only a timeout
  or 5xx, so check the logs first.
- **Cold starts:** the first request after idle pays for container start plus model
  load, which takes a few seconds. Warm requests reuse the loaded model.
- **Tile size:** the model was trained on 512 x 512 tiles. Send tiles of that size.
  Large rasters should be tiled client-side, for example with
  `tree_detection.dataset.iter_windows`.

## Gotchas met while building this

- **rfdetr version drift:** the checkpoint was trained with an older rfdetr whose
  head had no background logit. rfdetr 1.11 dropped the `class_names` constructor
  argument and `optimize_for_inference()`. `serving.load_rfdetr` therefore uses
  `RFDETRMedium.from_checkpoint`, which infers the head size from the weights.
- **FastAPI and `from __future__ import annotations`:** the handler's `UploadFile`
  annotation must resolve at runtime, so the Modal app does not use postponed
  annotations.
- **Windows shells:** Git Bash rewrites `/rfdetr/...` arguments into Windows paths
  (`C:/Program Files/Git/rfdetr/...`). Run `modal volume` commands from PowerShell or
  prefix them with `MSYS_NO_PATHCONV=1`. If Modal's output crashes with a `charmap`
  encoding error, set `PYTHONUTF8=1`.
- **Hot reload on OneDrive:** `modal serve` may not notice file changes in a
  OneDrive-synced folder. Restart it after editing.
