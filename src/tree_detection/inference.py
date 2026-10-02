"""Client types for calling a remotely hosted detection model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class Prediction:
    """One detection returned by a remote inference endpoint."""

    bbox: tuple[float, float, float, float]
    score: float
    class_id: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Prediction:
        bbox = value.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("prediction bbox must be a four-item list")
        score = float(value["score"])
        if not 0 <= score <= 1:
            raise ValueError("prediction score must be between 0 and 1")
        return cls(tuple(float(item) for item in bbox), score, int(value["class_id"]))


class InferenceClient:
    """HTTPS client for a provider-neutral object-detection endpoint."""

    def __init__(
        self,
        endpoint: str,
        api_token: str | None = None,
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else None
        self._client = httpx.Client(
            base_url=endpoint,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> InferenceClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def predict_bytes(
        self, image: bytes, *, filename: str = "tile.jpg"
    ) -> list[Prediction]:
        """Send encoded image bytes and parse pixel-space ``xyxy`` predictions."""
        response = self._client.post(
            "/predict", files={"image": (filename, image, "image/jpeg")}
        )
        response.raise_for_status()
        payload = response.json()
        predictions = payload.get("predictions")
        if not isinstance(predictions, list):
            raise ValueError("response must contain a predictions list")
        return [Prediction.from_dict(item) for item in predictions]

    def predict_file(self, image_path: str | Path) -> list[Prediction]:
        path = Path(image_path)
        return self.predict_bytes(path.read_bytes(), filename=path.name)
