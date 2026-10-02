# Remote inference API

Large model weights are intentionally excluded from this repository and from CI. The
client in `tree_detection.inference` can call a custom model hosted on Modal, RunPod,
SageMaker, Vertex AI, Azure ML, or any service implementing the contract below.

## Request

```http
POST /predict
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

The multipart body must contain one `image` file containing an encoded JPEG tile.
An optional `threshold` query parameter (0–1, default `0.25`) sets the minimum
confidence. `GET /health` returns `{"status": "ok"}` without authentication.

A reference implementation on Modal lives in `deploy/modal_app.py`; see
[Deploying the detector on Modal](deploy-modal.md).

## Response

```json
{
  "predictions": [
    {
      "bbox": [12.5, 20.0, 80.0, 96.5],
      "score": 0.91,
      "class_id": 0
    }
  ]
}
```

Bounding boxes use pixel-space `[xmin, ymin, xmax, ymax]` coordinates relative to the
submitted tile. Scores must be between zero and one.

Keep credentials outside configuration files. Load the endpoint and token from
environment variables or a secret manager in the application that creates the client.
