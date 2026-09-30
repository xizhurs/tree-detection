import httpx
import pytest

from tree_detection.inference import InferenceClient, Prediction


def test_inference_client_sends_token_and_parses_predictions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        assert request.headers["Authorization"] == "Bearer secret"
        assert b"image data" in request.read()
        return httpx.Response(
            200,
            json={
                "predictions": [{"bbox": [1, 2, 30, 40], "score": 0.9, "class_id": 0}]
            },
        )

    with InferenceClient(
        "https://models.example.test",
        "secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        predictions = client.predict_bytes(b"image data")

    assert predictions == [Prediction((1.0, 2.0, 30.0, 40.0), 0.9, 0)]


def test_inference_client_rejects_invalid_response() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, json={"predictions": "invalid"})
    )
    with (
        InferenceClient("https://models.example.test", transport=transport) as client,
        pytest.raises(ValueError, match="predictions list"),
    ):
        client.predict_bytes(b"image data")
