"""
test_data_processing.py – Minimal Unit Test for DataProcessor (B5W5)
------------------------------------------------------------------------------

Verifies that the DataProcessor class successfully runs its full pipeline
without raising errors and returns a valid DataFrame.

Author: Nabil Mohamed
"""

# -------------------------------------------------------------------------
# 📦 Standard Imports
# -------------------------------------------------------------------------

import pandas as pd  # For creating test DataFrame
from src.feature_engineering.data_processing import (
    DataProcessor,
)  # Import DataProcessor

# -------------------------------------------------------------------------
# 🧪 Test Case: Run Full Pipeline
# -------------------------------------------------------------------------


def test_run_full_pipeline():
    """
    Tests the DataProcessor's full pipeline on minimal dummy data.
    """

    # Create minimal valid input DataFrame matching required columns
    data = {
        "AccountId": ["ACC123"],
        "CustomerId": ["CUST456"],
        "ProductId": ["PROD789"],
        "Amount": [100.0],
        "Value": [150.0],
        "FraudResult": [0],
        "Recency": [10],
        "Frequency": [5],
        "Monetary": [500],
        "TransactionStartTime": ["2025-01-01 10:00:00"],
        "CurrencyCode": ["USD"],
        "ProviderId": ["P1"],
        "ProductCategory": ["CatA"],
        "ChannelId": ["Web"],
    }

    df = pd.DataFrame(data)  # Create test DataFrame

    # Initialize processor and run pipeline
    processor = DataProcessor(df)
    processed_df = processor.run_full_pipeline()

    # Assert that result is a non-empty DataFrame
    assert not processed_df.empty, "Processed DataFrame should not be empty."
    assert isinstance(processed_df, pd.DataFrame), "Output must be a pandas DataFrame."
