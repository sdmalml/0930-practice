from fastapi.testclient import TestClient
from inference import app  # app.py에서 FastAPI 인스턴스 가져오기

client = TestClient(app)

def test_prediction():
    # 샘플 데이터
    sample_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    # POST 요청을 통해 예측 결과 확인
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "iris_class" in response.json()  # 'prediction'이 아닌 'iris_class'로 수정
    assert isinstance(response.json()["iris_class"], int)  # 예측 결과가 int 타입인지 확인
