from fastapi.testclient import TestClient
from inference import app  # FastAPI 애플리케이션 가져오기

client = TestClient(app)

def test_prediction():
    # 샘플 데이터 정의
    sample_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    # POST 요청을 통해 예측 결과 확인
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
