"""
test_api.py – Minimal Unit Test for BNPL Credit Risk FastAPI Service (B5W5)
------------------------------------------------------------------------------

Ensures that the Pydantic input model and the FastAPI app respond correctly.

Author: Nabil Mohamed
"""

# -------------------------------------------------------------------------
# 📦 Standard Imports
# -------------------------------------------------------------------------

from fastapi.testclient import TestClient  # FastAPI test client
from src.api.main import app  # Import FastAPI app

# -------------------------------------------------------------------------
# 🚀 Initialize Test Client
# -------------------------------------------------------------------------

client = TestClient(app)  # Create test client for FastAPI

# -------------------------------------------------------------------------
# 🧪 Test Case: Health Check for /predict
# -------------------------------------------------------------------------


def test_predict_endpoint():
    """
    Tests the /predict endpoint with dummy valid input and checks for valid response.
    """

    # Define minimal valid input matching Pydantic schema
    payload = {"feature_1": 0.5, "feature_2": 1200.0, "feature_3": 1, "feature_4": 0.25}

    # Send POST request to /predict
    response = client.post("/predict", json=payload)

    # Assert that response is successful
    assert response.status_code == 200, "Expected status code 200 OK"

    # Assert that risk_probability is in response and is a float
    result = response.json()
    assert "risk_probability" in result, "Missing 'risk_probability' in response"
    assert isinstance(
        result["risk_probability"], float
    ), "'risk_probability' must be a float"
