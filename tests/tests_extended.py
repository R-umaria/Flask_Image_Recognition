"""
Extended test suite for Flask Image Recognition Project
Covers edge cases, integration, and robustness.
"""

import io
import pytest
from PIL import Image
import numpy as np
from app import app
from model import model


@pytest.fixture
def client():
    """Flask test client fixture"""
    app.config.update(TESTING=True)
    return app.test_client()


def test_predict_rejects_empty_file(client):
    """Rejects empty or missing file uploads"""
    data = {"file": ("", "")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code in (400, 422)


def test_predict_rejects_large_image(client):
    """Rejects excessively large files"""
    img_data = io.BytesIO(b"\x89PNG" + b"0" * 5_000_000)
    data = {"file": (img_data, "big.png")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code in (400, 413)


def test_full_prediction_flow(client, tmp_path):
    """Integration test – simulate end-to-end prediction"""
    img_path = tmp_path / "digit.png"
    Image.new("L", (28, 28)).save(img_path)
    with open(img_path, "rb") as f:
        resp = client.post("/predict", data={"file": (f, "digit.png")},
                           content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Predicted" in resp.data


def test_predict_multiple_requests(client):
    """Performance check – handles multiple sequential requests"""
    img = Image.new("L", (28, 28))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    for _ in range(5):
        buf.seek(0)
        resp = client.post("/predict", data={"file": (buf, "img.png")},
                           content_type="multipart/form-data")
        assert resp.status_code == 200


def test_model_output_is_digit():
    """Unit test – ensures model outputs valid probability vector"""
    dummy = np.zeros((1, 28, 28, 1))
    result = model.predict(dummy)
    assert result.shape == (1, 10)
    assert np.all(result >= 0)
    assert np.all(result <= 1)
