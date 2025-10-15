import io
from app import app

def test_home_ok():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        r = c.get("/")
        assert r.status_code == 200

def test_predict_rejects_non_image():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        data = {"file": (io.BytesIO(b"not an image"), "bad.txt")}
        r = c.post("/predict", data=data, content_type="multipart/form-data")
        assert r.status_code in (400, 422) or b"invalid" in r.data.lower()
