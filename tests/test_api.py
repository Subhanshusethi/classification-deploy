from fastapi.testclient import TestClient
from app.main import app

def test_api_predicts():
    image_path = "tests/test_image.png"
    with TestClient(app) as client:
        with open(image_path, "rb") as img:
            response = client.post("/upload_image", files={"file": img})

        assert response.status_code == 200
        data = response.json()
        print(data)
        assert "pred" in data
        assert "prob_no_scratches" in data
        assert "prob_scratches" in data
