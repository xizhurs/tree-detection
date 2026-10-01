"""Serve the fine-tuned RF-DETR tree detector on a Modal GPU.

Usage (see docs/deploy-modal.md for the full walkthrough):

    modal run deploy/modal_app.py --image tile.jpg   # one-off remote call
    modal serve deploy/modal_app.py                  # temporary dev URL, hot reload
    modal deploy deploy/modal_app.py                 # persistent URL
"""

# No `from __future__ import annotations`: FastAPI must resolve the handler's
# annotations at runtime, and UploadFile is only imported inside the container.
import os
from pathlib import Path

import modal

APP_NAME = "tree-detection"
WEIGHTS_VOLUME = "tree-detection-weights"
WEIGHTS_DIR = "/weights"
DEFAULT_WEIGHTS = f"{WEIGHTS_DIR}/rfdetr/checkpoint_best_ema.pth"
TOKEN_SECRET = "tree-detection-api-token"
# Read locally at deploy time, e.g. TREE_DETECTION_GPU=L4, or =cpu for no GPU.
_gpu = os.environ.get("TREE_DETECTION_GPU", "T4")
GPU = None if _gpu.lower() in ("", "cpu", "none") else _gpu

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "tree_detection"

# The container image is built once and cached; only changed layers rebuild.
# Versions match uv.lock so the server runs the same code that was trained locally.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        "rfdetr==1.11.1",
        "supervision==0.30.6",
        "pillow==12.3.0",
        "fastapi[standard]",
    )
    # Ship the local package code; this layer is re-uploaded on every run/deploy.
    .add_local_dir(PACKAGE_DIR, "/root/tree_detection")
)

app = modal.App(APP_NAME, image=image)

# Weights live in a Volume, separate from the image, so retraining only needs
# `modal volume put ...` and no rebuild.
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)


@app.cls(
    gpu=GPU,
    volumes={WEIGHTS_DIR: weights_volume},
    secrets=[modal.Secret.from_name(TOKEN_SECRET)],
    scaledown_window=300,  # keep a warm container for 5 min after the last request
    max_containers=2,  # cost guard-rail
    timeout=600,
)
class TreeDetector:
    @modal.enter()
    def load_model(self) -> None:
        """Runs once per container start, so requests reuse the loaded model."""
        from tree_detection.serving import load_rfdetr

        weights = os.environ.get("TREE_DETECTION_WEIGHTS", DEFAULT_WEIGHTS)
        self.model = load_rfdetr(weights)  # cuda if a GPU is attached

    def _predict(self, data: bytes, threshold: float) -> dict:
        from tree_detection.serving import decode_image, detections_to_payload

        detections = self.model.predict(decode_image(data), threshold=threshold)
        return detections_to_payload(
            detections.xyxy, detections.confidence, detections.class_id
        )

    @modal.method()
    def predict(self, image_bytes: bytes, threshold: float = 0.25) -> dict:
        """Python RPC entry point: ``TreeDetector().predict.remote(...)``."""
        return self._predict(image_bytes, threshold)

    @modal.asgi_app()
    def web(self):
        """HTTP entry point implementing docs/inference-api.md."""
        from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile

        from tree_detection.serving import is_authorized

        api = FastAPI(title="Tree detection")
        token = os.environ["TREE_DETECTION_TOKEN"]

        @api.get("/health")
        def health() -> dict:
            return {"status": "ok"}

        # Sync handler: FastAPI runs it in a thread pool, so GPU work does not
        # block the event loop.
        @api.post("/predict")
        def predict(
            image: UploadFile = File(...),  # noqa: B008
            threshold: float = Query(0.25, ge=0.0, le=1.0),
            authorization: str | None = Header(None),
        ) -> dict:
            if not is_authorized(authorization, token):
                raise HTTPException(status_code=401, detail="invalid or missing token")
            try:
                return self._predict(image.file.read(), threshold)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return api


@app.local_entrypoint()
def main(image: str, threshold: float = 0.25) -> None:
    """Smoke test: send a local tile to the GPU container and print detections."""
    result = TreeDetector().predict.remote(Path(image).read_bytes(), threshold)
    predictions = result["predictions"]
    print(f"{len(predictions)} trees detected (threshold={threshold})")
    for prediction in predictions[:10]:
        print(prediction)
