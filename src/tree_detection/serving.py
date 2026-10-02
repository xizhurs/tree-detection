"""Framework-free helpers for serving the detector behind the ``/predict`` API."""

from __future__ import annotations

import hmac
import io
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def decode_image(data: bytes) -> Any:
    """Decode encoded image bytes into an RGB ``PIL.Image``."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("request body is not a decodable image") from exc


def detections_to_payload(
    xyxy: Iterable[Sequence[float]] | None,
    confidence: Iterable[float] | None,
    class_id: Iterable[int | None] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Format detections as the JSON body documented in ``docs/inference-api.md``."""
    if xyxy is None or confidence is None:
        return {"predictions": []}
    boxes = list(xyxy)
    scores = list(confidence)
    classes = list(class_id) if class_id is not None else [0] * len(boxes)
    predictions = [
        {
            "bbox": [float(value) for value in box],
            "score": min(max(float(score), 0.0), 1.0),
            "class_id": int(cls) if cls is not None else 0,
        }
        for box, score, cls in zip(boxes, scores, classes, strict=True)
    ]
    return {"predictions": predictions}


def is_authorized(header: str | None, expected_token: str) -> bool:
    """Return whether an ``Authorization`` header carries the expected bearer token."""
    if not header or not expected_token:
        return False
    return hmac.compare_digest(header.encode(), f"Bearer {expected_token}".encode())


def load_rfdetr(weights_path: str | Path, device: str | None = None) -> Any:
    """Load the fine-tuned RF-DETR Medium checkpoint for inference.

    ``from_checkpoint`` infers the head size from the weights, which keeps checkpoints
    trained with older rfdetr releases (no background logit) loadable.
    """
    from rfdetr import RFDETRMedium

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return RFDETRMedium.from_checkpoint(str(weights_path), device=device)
