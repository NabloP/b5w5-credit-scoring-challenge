"""
customer_behavior_profiler.py – Enhanced RFM Profiler with Shared Account Awareness (B5W5)
------------------------------------------------------------------------------
Computes Recency, Frequency, and Monetary (RFM) metrics for each customer in
the transaction dataset, with optional segmentation by shared vs. individual accounts.

Core Features:
  • Aggregate RFM metrics per customer
  • Optional segmentation by shared account flag
  • Defensible input validation and error handling

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For data manipulation
import numpy as np  # For calculations


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: CustomerBehaviorProfiler
# ───────────────────────────────────────────────────────────────────────────────
class CustomerBehaviorProfiler:
    """
    Computes RFM (Recency, Frequency, Monetary) metrics per customer,
    optionally stratified by shared account usage.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        customer_id_col: str,
        date_col: str,
        value_col: str,
        shared_col: str = None,
    ):
        """
        Initialize the profiler with the transaction dataset.

        Args:
            df (pd.DataFrame): The input transaction data.
            customer_id_col (str): Column name for customer ID.
            date_col (str): Column name for transaction date.
            value_col (str): Column name for transaction value.
            shared_col (str, optional): Column name for shared account flag (0/1). Defaults to None.

        Raises:
            ValueError: If required columns are missing or date parsing fails.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        required_cols = [customer_id_col, date_col, value_col]
        if shared_col:
            required_cols.append(shared_col)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        try:
            df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(
                None
            )  # Ensure timezone neutrality
        except Exception as e:
            raise ValueError(f"Failed to parse date column '{date_col}': {e}")

        self.df = df.copy()
        self.customer_col = customer_id_col
        self.date_col = date_col
        self.value_col = value_col
        self.shared_col = shared_col  # Optional shared flag

    def compute_rfm(self, snapshot_date: str, split_by_shared: bool = False) -> dict:
        """
        Computes Recency, Frequency, and Monetary metrics.

        Args:
            snapshot_date (str): The reference date for recency.
            split_by_shared (bool): If True, computes separate RFM for shared vs. individual accounts.

        Returns:
            dict: Dictionary of RFM DataFrames: {'Overall': df, 'Shared': df, 'Individual': df}
        """
        try:
            snapshot_dt = pd.to_datetime(snapshot_date).tz_localize(None)
        except Exception as e:
            raise ValueError(f"Invalid snapshot date '{snapshot_date}': {e}")

        results = {}

        def compute_grouped_rfm(sub_df):
            grouped = sub_df.groupby(self.customer_col)
            rfm = (
                grouped.agg(
                    Recency=(self.date_col, lambda x: (snapshot_dt - x.max()).days),
                    Frequency=(self.date_col, "count"),
                    Monetary=(self.value_col, "sum"),
                )
                .reset_index()
                .rename(columns={self.customer_col: "CustomerId"})
            )
            return rfm

        # Overall RFM
        results["Overall"] = compute_grouped_rfm(self.df)

        # Shared vs. Individual split if requested and shared_col is provided
        if split_by_shared and self.shared_col:
            shared_df = self.df[self.df[self.shared_col] == 1]
            individual_df = self.df[self.df[self.shared_col] == 0]
            results["Shared"] = compute_grouped_rfm(shared_df)
            results["Individual"] = compute_grouped_rfm(individual_df)

        print(f"✅ RFM computed: {list(results.keys())} (Snapshot: {snapshot_date})")
        return results
