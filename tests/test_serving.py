import io
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tree_detection.serving import (
    decode_image,
    detections_to_payload,
    is_authorized,
    load_rfdetr,
)


def _jpeg_bytes(mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (16, 8)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_decode_image_returns_rgb() -> None:
    image = decode_image(_jpeg_bytes("L"))
    assert image.mode == "RGB"
    assert image.size == (16, 8)


def test_decode_image_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="decodable image"):
        decode_image(b"not an image")


def test_detections_to_payload_matches_api_contract() -> None:
    payload = detections_to_payload(
        np.array([[1, 2, 3, 4], [5.5, 6, 7, 8]], dtype=np.float32),
        np.array([0.9, 1.0000001]),
        [0, None],
    )
    assert payload == {
        "predictions": [
            {"bbox": [1.0, 2.0, 3.0, 4.0], "score": pytest.approx(0.9), "class_id": 0},
            {"bbox": [5.5, 6.0, 7.0, 8.0], "score": 1.0, "class_id": 0},
        ]
    }


def test_detections_to_payload_handles_empty_and_missing_classes() -> None:
    assert detections_to_payload(None, None, None) == {"predictions": []}
    payload = detections_to_payload([[0, 0, 1, 1]], [0.5], None)
    assert payload["predictions"][0]["class_id"] == 0


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer secret", True),
        ("Bearer wrong", False),
        ("secret", False),
        (None, False),
    ],
)
def test_is_authorized(header: str | None, expected: bool) -> None:
    assert is_authorized(header, "secret") is expected


def test_is_authorized_rejects_empty_expected_token() -> None:
    assert is_authorized("Bearer ", "") is False


def test_load_rfdetr_uses_lazy_model_import(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeModel:
        @classmethod
        def from_checkpoint(cls, path: str, **kwargs: object) -> "FakeModel":
            calls["path"] = path
            calls["kwargs"] = kwargs
            return cls()

    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRMedium=FakeModel))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    model = load_rfdetr("weights.pth")

    assert isinstance(model, FakeModel)
    assert calls == {"path": "weights.pth", "kwargs": {"device": "cpu"}}
