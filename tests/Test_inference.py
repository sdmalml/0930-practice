# tests/test_inference.py

from fastapi.testclient import TestClient
from inference import app

client = TestClient(app)

def test_prediction():
    sample_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
