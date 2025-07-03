"""
main.py – FastAPI Inference Service for BNPL Credit Risk Model (B5W5)
------------------------------------------------------------------------------

Exposes a REST API for serving the trained credit risk model using FastAPI.
The API loads the best model directly from a pickle file and provides a /predict
endpoint for real-time scoring of new applicants.

Author: Nabil Mohamed
"""

# -------------------------------------------------------------------------
# 📦 Standard Imports
# -------------------------------------------------------------------------

import uvicorn  # ASGI server to run FastAPI
import pickle  # For loading the serialized model from .pkl
from fastapi import FastAPI, HTTPException  # FastAPI core components
from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
)  # Pydantic models for request/response validation

# -------------------------------------------------------------------------
# 🚀 Initialize FastAPI App and Load Model from Pickle
# -------------------------------------------------------------------------

# ✅ Create FastAPI application instance
app = FastAPI(
    title="BNPL Credit Risk Scoring API", version="1.0"
)  # Initialize FastAPI app

# ✅ Define path to trained model artifact (update path if needed)
model_path = "mlruns/379996855192855743/models/m-e19d4a66b36e4e2393e811669f46219f/artifacts/model.pkl"

# ✅ Load trained model using pickle
try:
    with open(model_path, "rb") as file:  # Open the pickle file safely
        model = pickle.load(file)  # Load the model object into memory

    print("✅ Model loaded successfully for inference.")  # Confirm model load

except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")  # Fail loudly if loading fails

# -------------------------------------------------------------------------
# 📌 Define /predict Endpoint for Real-Time Scoring
# -------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionRequest):
    """
    Predict risk probability for a new BNPL applicant based on input features.
    """
    try:
        # ✅ Convert Pydantic input to DataFrame expected by sklearn model
        input_df = input_data.to_dataframe()  # Convert to pandas DataFrame

        # ✅ Generate probability prediction (probability of class 1 = high risk)
        proba = model.predict_proba(input_df)[0][
            1
        ]  # Extract positive class probability

        # ✅ Return rounded risk probability using response model
        return PredictionResponse(
            risk_probability=round(proba, 4)
        )  # Return as JSON response

    except Exception as e:
        # ❌ Return HTTP 500 with clear error message
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# -------------------------------------------------------------------------
# 🚀 Run FastAPI App Locally (Optional for Development)
# -------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Run FastAPI app on localhost
