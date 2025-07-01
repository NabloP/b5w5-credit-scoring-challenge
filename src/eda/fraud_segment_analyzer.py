"""
fraud_segment_analyzer.py – Fraud Risk Segment Analyzer (B5W5)
------------------------------------------------------------------------------
Analyzes the distribution of fraud flags (FraudResult) across key
categorical segments to identify risk-prone channels, products, or patterns.

Core responsibilities:
  • Summarize fraud frequency and share by any categorical column
  • Plot fraud rates using grouped bar charts
  • Validate input data and guard against misuse
  • Provide pivot tables for advanced fraud pattern inspection

Used in Task 2 EDA to explore risk hotspots and inform proxy labeling.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For data manipulation
import seaborn as sns  # For bar plots
import matplotlib.pyplot as plt  # For axis formatting


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: FraudSegmentAnalyzer
# ───────────────────────────────────────────────────────────────────────────────
class FraudSegmentAnalyzer:
    """
    Explores fraud distribution across categorical segments.
    """

    def __init__(self, df: pd.DataFrame, fraud_col: str = "FraudResult"):
        """
        Initialize the analyzer with raw data and fraud flag column.

        Args:
            df (pd.DataFrame): The transaction dataset.
            fraud_col (str): Name of the binary fraud flag column (0/1).

        Raises:
            ValueError: If fraud_col is missing or not binary.
        """
        self.df = df.copy()

        # 🛡️ Validate fraud column
        if fraud_col not in df.columns:
            raise ValueError(f"Column '{fraud_col}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[fraud_col]):
            raise ValueError(f"'{fraud_col}' must be numeric (0/1).")
        if set(df[fraud_col].dropna().unique()) - {0, 1}:
            raise ValueError(f"'{fraud_col}' must be binary (0 or 1 only).")

        self.fraud_col = fraud_col

    def summarize_by_segment(self, segment_col: str, top_k: int = 10) -> pd.DataFrame:
        """
        Returns a summary table of fraud counts and rates per segment.

        Args:
            segment_col (str): Categorical column to group by.
            top_k (int): Max number of segment values to include.

        Returns:
            pd.DataFrame: Fraud count, total count, and fraud rate per category.
        """
        if segment_col not in self.df.columns:
            raise ValueError(f"Column '{segment_col}' not found in DataFrame.")

        # 📊 Group and compute fraud counts + rate
        grouped = (
            self.df.groupby(segment_col)[self.fraud_col]
            .agg(["sum", "count"])
            .rename(columns={"sum": "FraudCount", "count": "TotalCount"})
        )
        grouped["FraudRate"] = grouped["FraudCount"] / grouped["TotalCount"]

        # 📦 Return top_k most frequent segments
        result = (
            grouped.sort_values("TotalCount", ascending=False).head(top_k).reset_index()
        )
        print(f"✅ Fraud summary computed by '{segment_col}'")
        return result

    def plot_fraud_rate_by_segment(self, segment_col: str, top_k: int = 10) -> None:
        """
        Plots fraud rate by segment as a horizontal bar chart.

        Args:
            segment_col (str): Column to segment fraud rates by.
            top_k (int): Number of top segments to show by volume.
        """
        # ⏬ Summarize fraud metrics
        summary_df = self.summarize_by_segment(segment_col, top_k=top_k)

        # 🎨 Plot fraud rate
        plt.figure(figsize=(10, 5))
        sns.barplot(x="FraudRate", y=segment_col, data=summary_df, palette="Reds_r")
        plt.title(f"🛡️ Fraud Rate by '{segment_col}' (Top {top_k})")
        plt.xlabel("Fraud Rate (%)")
        plt.ylabel(segment_col)
        plt.xlim(0, summary_df["FraudRate"].max() * 1.2)
        plt.tight_layout()
        plt.show()
