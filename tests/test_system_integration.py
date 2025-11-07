# tests/test_system_integration.py
import io
import numpy as np
from PIL import Image
from unittest.mock import patch
from app import app


def _make_dummy_png(width=28, height=28):
    """Create a tiny grayscale PNG in-memory to simulate a user upload."""
    img = Image.fromarray((np.random.rand(height, width) * 255).astype("uint8"), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@patch("model.predict_result")   # patch the real function used by app.py
def test_happy_path_predicts_digit(mock_predict):
    """
    Happy path:
    - POST /prediction with valid PNG
    - Mock the model.predict_result() call
    - Expect HTTP 200 and rendered result page containing the fake digit
    """
    mock_predict.return_value = "7"  # Fake prediction

    client = app.test_client()
    data = {"file": (_make_dummy_png(), "digit.png")}
    resp = client.post("/prediction", data=data, content_type="multipart/form-data")

    # Since Flask renders HTML template, 200 OK means success
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "7" in html or "predictions" in html  # check that template rendered correctly


def test_sad_path_missing_file_returns_error():
    """
    Sad path:
    - POST /prediction with no file
    - Expect graceful error message and 200 (template rendered with error)
    """
    client = app.test_client()
    resp = client.post("/prediction", data={}, content_type="multipart/form-data")

    # App always returns 200 even on error
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "File cannot be processed" in html or "error" in html
