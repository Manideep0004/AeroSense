from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "active_models" in data
    assert "drift_status" in data


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_predict_invalid_payload():
    # Test with empty payload to verify Pydantic validation
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity
