"""
pydantic_models.py – Request and Response Schemas for BNPL Credit Risk API (B5W5)
------------------------------------------------------------------------------

Defines the Pydantic models used for validating incoming requests and structuring
outgoing responses for the FastAPI-based credit risk scoring service.

Author: Nabil Mohamed
"""

# -------------------------------------------------------------------------
# 📦 Standard Imports
# -------------------------------------------------------------------------

from pydantic import BaseModel, Field  # For request/response validation
import pandas as pd  # For DataFrame conversion

# -------------------------------------------------------------------------
# 📄 PredictionRequest Schema
# -------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """
    Input schema representing the applicant's features for risk prediction.
    All fields must match the trained model's expected feature set.
    """

    feature_1: float = Field(..., example=0.75, description="Example numeric feature 1")
    feature_2: float = Field(
        ..., example=1500.0, description="Example numeric feature 2"
    )
    feature_3: int = Field(..., example=1, description="Example binary feature")
    feature_4: float = Field(
        ..., example=0.33, description="Example continuous feature"
    )

    def to_dataframe(self):
        """
        Converts the request object into a pandas DataFrame suitable for model input.
        Returns:
            pd.DataFrame: Single-row DataFrame matching model input schema.
        """
        return pd.DataFrame([self.dict()])  # Convert Pydantic model to DataFrame


# -------------------------------------------------------------------------
# 📄 PredictionResponse Schema
# -------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """
    Output schema returning the predicted risk probability.
    """

    risk_probability: float = Field(
        ..., example=0.72, description="Predicted risk probability (0 to 1)"
    )
