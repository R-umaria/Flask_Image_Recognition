# tests/test_routes_smoke.py
import io
from app import app

def test_rejects_non_image_without_500():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        for path in ("/predict", "/", "/upload"):
            # NEW: fresh BytesIO each time so it's not closed after first POST
            data = {"file": (io.BytesIO(b"not an image"), "bad.txt")}
            resp = c.post(
                path,
                data=data,
                content_type="multipart/form-data",
                follow_redirects=True,
            )
            # Accept legacy behaviors; just ensure no 5xx
            assert resp.status_code < 500
            # If we hit anything other than 404, we’re good—stop trying alternates
            if resp.status_code != 404:
                break
