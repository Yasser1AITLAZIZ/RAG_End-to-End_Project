from fastapi.testclient import TestClient
from src.api.main import app

# Create a test client
client = TestClient(app)


def test_health_check():
    """
    Test the health check endpoint.
    """
    response = client.get("/api/health")
    assert response.status_code == 200, "Health endpoint should return status code 200"
    assert response.json() == {"status": "OK"}, "Health endpoint should return {'status': 'OK'}"


def test_generate_response_success():
    """
    Test the generate-response endpoint with valid data.
    """
    payload = {
        "query": "What are the benefits of the OptimRiskMaximizer method?",
        "documents": [
            {"text": "The OptimRiskMaximizer is a method for optimizing risk management in trading."},
            {"text": "It balances risk and reward dynamically based on market conditions."},
        ],
    }
    response = client.post("/api/generate-response", json=payload)
    assert response.status_code == 200, "Generate-response endpoint should return status code 200"
    response_data = response.json()
    assert "response" in response_data, "Response should contain a 'response' field"
    assert len(response_data["response"]) > 0, "Response should not be empty"


def test_generate_response_no_documents():
    """
    Test the generate-response endpoint with no documents.
    """
    payload = {"query": "What are the benefits of the OptimRiskMaximizer method?", "documents": []}
    response = client.post("/api/generate-response", json=payload)
    assert response.status_code == 200, "Generate-response endpoint should handle empty documents gracefully"
    response_data = response.json()
    assert "response" in response_data, "Response should contain a 'response' field"
    assert len(response_data["response"]) > 0, "Response should not be empty even with no documents"


def test_generate_response_missing_fields():
    """
    Test the generate-response endpoint with missing required fields.
    """
    payload = {"query": "What are the benefits of the OptimRiskMaximizer method?"}
    response = client.post("/api/generate-response", json=payload)
    assert response.status_code == 422, "Missing fields should return status code 422 (Unprocessable Entity)"


def test_generate_response_invalid_data():
    """
    Test the generate-response endpoint with invalid data types.
    """
    payload = {"query": "What are the benefits of the OptimRiskMaximizer method?", "documents": "This is not a list"}
    response = client.post("/api/generate-response", json=payload)
    assert response.status_code == 422, "Invalid data types should return status code 422 (Unprocessable Entity)"
