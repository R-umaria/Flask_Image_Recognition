from io import BytesIO
import pytest


def test_app_error_handling(client):
    """Test error handling paths in app.py"""

    # Test missing file (should trigger error handling and return 200 with error message)
    response = client.post('/prediction')
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data

    # Test invalid file type (simulate non-image content)
    data = BytesIO(b"not an image")
    data.name = "test.txt"
    response = client.post(
        '/prediction',
        data={'file': (data, 'test.txt')},
        content_type='multipart/form-data'
    )
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_app_server_error(client, monkeypatch):
    """Test server error handling (should trigger except block)"""

    def mock_predict(*args):
        raise Exception("Simulated error")

    from model import predict_result
    monkeypatch.setattr("model.predict_result", mock_predict)

    data = BytesIO(b"fake image data")
    data.name = "test.jpg"
    response = client.post(
        '/prediction',
        data={'file': (data, 'test.jpg')},
        content_type='multipart/form-data'
    )
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data

def test_prediction_route_logic(client, monkeypatch):
    """Covers lines 12, 21 - 22, and 31 in app.py"""

    # Mock both preprocess_img and predict_result
    monkeypatch.setattr("app.preprocess_img", lambda x: "mocked_img")
    monkeypatch.setattr("app.predict_result", lambda x: "mocked_digit")

    from io import BytesIO
    dummy = BytesIO(b"fake image data")
    dummy.name = "test.jpg"

    response = client.post(
        "/prediction",
        data={"file": (dummy, "test.jpg")},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"mocked_digit" in response.data

def test_prediction_forces_exception(client, monkeypatch):
    """Force an exception to hit the except block (line 31)"""

    def raise_error(*args, **kwargs):
        raise Exception("forced error")

    monkeypatch.setattr("app.preprocess_img", raise_error)

    from io import BytesIO
    dummy = BytesIO(b"bad data")
    dummy.name = "test.jpg"

    response = client.post(
        "/prediction",
        data={"file": (dummy, "test.jpg")},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data

def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hand Sign Digit" in response.data  # or any unique text from index.html
