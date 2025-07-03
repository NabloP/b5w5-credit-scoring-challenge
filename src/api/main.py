"""
main.py – FastAPI Inference Service for BNPL Credit Risk Model (B5W5)
------------------------------------------------------------------------------

Exposes a REST API for serving the trained credit risk model using FastAPI.
The API loads the best model from MLflow and provides a /predict endpoint
for real-time scoring of new applicants.

Author: Nabil Mohamed
"""

# -------------------------------------------------------------------------
# 📦 Standard Imports
# -------------------------------------------------------------------------

import uvicorn  # ASGI server to run FastAPI
import mlflow.sklearn  # For loading the model from MLflow
from fastapi import FastAPI, HTTPException  # FastAPI core components
from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
)  # Pydantic models for request/response validation

# -------------------------------------------------------------------------
# 🚀 Initialize FastAPI App and Load Model
# -------------------------------------------------------------------------

app = FastAPI(
    title="BNPL Credit Risk Scoring API", version="1.0"
)  # Initialize FastAPI app

# ✅ Load trained model from MLflow (replace with your actual path)
try:
    model = mlflow.sklearn.load_model(
        "mlruns/379996855192855743/models/m-e19d4a66b36e4e2393e811669f46219f/artifacts"
    )  # Load the logged model artifact
except Exception as e:
    raise RuntimeError(
        f"❌ Failed to load model: {e}"
    )  # Fail loudly if model loading fails

# -------------------------------------------------------------------------
# 📌 Define /predict Endpoint
# -------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionRequest):
    """
    Predict risk probability for a new BNPL applicant based on input features.
    """
    try:
        # ✅ Convert incoming Pydantic request to a DataFrame for sklearn model
        input_df = input_data.to_dataframe()

        # ✅ Predict risk probability (positive class probability: class 1 = high risk)
        proba = model.predict_proba(input_df)[0][1]

        # ✅ Return the result as rounded risk probability
        return PredictionResponse(risk_probability=round(proba, 4))

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {e}"
        )  # Handle errors gracefully


# -------------------------------------------------------------------------
# 🚀 Run the App Locally (Optional)
# -------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Run FastAPI app locally
