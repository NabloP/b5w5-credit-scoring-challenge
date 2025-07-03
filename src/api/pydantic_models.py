"""
pydantic_models.py – Request and Response Schemas for BNPL Credit Risk API (B5W5)
------------------------------------------------------------------------------

Defines the Pydantic models used for validating incoming requests and structuring
outgoing responses for the FastAPI-based credit risk scoring service.

This version is designed to accept a fixed-length feature vector as a list,
matching the model's deployment requirements in Task 6.

Author: Nabil Mohamed
"""

# -------------------------------------------------------------------------
# 📦 Standard Imports
# -------------------------------------------------------------------------

from pydantic import BaseModel, Field, conlist  # Pydantic schema validation
import pandas as pd  # DataFrame conversion

from typing import List  # Type hinting for feature list

# -------------------------------------------------------------------------
# 📄 PredictionRequest Schema (Feature Vector Input)
# -------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """
    Input schema representing the applicant's features for credit risk scoring.
    Expects a fixed-length numeric vector matching the model's training features.
    """

    features: conlist(float, min_items=23, max_items=23) = Field(
        ...,
        example=[
            0.0,
            1.0,
            2.5,
            100.0,
            0.0,
            0.1,
            1,
            0,
            10,
            5,
            200,
            1,
            5,
            1000,
            200,
            50,
            10,
            1,
            2,
            3,
            1,
            0,
            1,
        ],
        description="List of 23 numeric features matching model input schema",
    )

    def to_dataframe(self):
        """
        Converts the request object into a pandas DataFrame suitable for model input.

        Returns:
            pd.DataFrame: Single-row DataFrame matching the trained model's expected shape.
        """
        return pd.DataFrame([self.features])  # Return single-row DataFrame


# -------------------------------------------------------------------------
# 📄 PredictionResponse Schema (Risk Probability Output)
# -------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """
    Output schema returning the predicted risk probability for the applicant.
    """

    risk_probability: float = Field(
        ...,
        example=0.72,
        description="Predicted risk probability (range: 0.0 to 1.0)",
    )
