"""
customer_behavior_profiler.py – RFM Profiler for Customer Segmentation (B5W5)
------------------------------------------------------------------------------
Computes Recency, Frequency, and Monetary (RFM) metrics for each customer in
the transaction dataset. These behavioral features form the basis for proxy
risk label engineering and creditworthiness modeling.

Core responsibilities:
  • Aggregate per-customer RFM statistics from raw transactions
  • Compute Recency from a fixed snapshot date
  • Handle missing values or duplicate customers defensively
  • Strip timezone info to avoid tz-aware/naive datetime errors

Used in Task 2 EDA and Task 4 proxy target engineering.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For DataFrame operations
import numpy as np  # For date math


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: CustomerBehaviorProfiler
# ───────────────────────────────────────────────────────────────────────────────
class CustomerBehaviorProfiler:
    """
    Computes RFM (Recency, Frequency, Monetary) statistics per customer
    to support credit scoring via behavioral segmentation.
    """

    def __init__(
        self, df: pd.DataFrame, customer_id_col: str, date_col: str, value_col: str
    ):
        """
        Initialize with transaction data and key column names.

        Args:
            df (pd.DataFrame): Raw transaction data
            customer_id_col (str): Name of the customer ID column
            date_col (str): Name of the transaction datetime column
            value_col (str): Name of the monetary value column (e.g., 'Value')

        Raises:
            ValueError: If required columns are missing or malformed
        """
        self.df = df.copy()  # 🔒 Defensive copy to avoid mutating input

        # 🛡️ Validate required columns
        for col in [customer_id_col, date_col, value_col]:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        # 🕒 Attempt to parse date column, forcibly remove timezone info
        try:
            self.df[date_col] = pd.to_datetime(self.df[date_col]).dt.tz_localize(None)
        except Exception as e:
            raise ValueError(f"Failed to convert '{date_col}' to datetime: {e}")

        # ✅ Store configuration for downstream use
        self.customer_col = customer_id_col
        self.date_col = date_col
        self.value_col = value_col

    def compute_rfm(self, snapshot_date: str) -> pd.DataFrame:
        """
        Computes Recency, Frequency, and Monetary value per customer.

        Args:
            snapshot_date (str): The reference date for recency calculation (YYYY-MM-DD)

        Returns:
            pd.DataFrame: RFM table with columns:
                          ['CustomerId', 'Recency', 'Frequency', 'Monetary']

        Raises:
            ValueError: If snapshot_date cannot be parsed
        """
        # 🕒 Convert snapshot date, forcibly remove timezone info to avoid tz mismatch
        try:
            snapshot_dt = pd.to_datetime(snapshot_date).tz_localize(None)
        except Exception as e:
            raise ValueError(f"Invalid snapshot date '{snapshot_date}': {e}")

        # 📦 Group data by customer
        grouped = self.df.groupby(self.customer_col)

        # 🧮 Compute RFM values using aggregation
        rfm = grouped.agg(
            Recency=(
                self.date_col,
                lambda x: (snapshot_dt - x.max()).days,
            ),  # days since last transaction
            Frequency=(self.date_col, "count"),  # total transactions
            Monetary=(self.value_col, "sum"),  # total monetary value
        ).reset_index()

        # 🧾 Rename customer ID column for consistency (optional)
        rfm.rename(columns={self.customer_col: "CustomerId"}, inplace=True)

        # ✅ Log RFM table creation
        print(
            f"✅ RFM profile computed for {rfm.shape[0]:,} customers (Snapshot: {snapshot_date})"
        )

        return rfm
