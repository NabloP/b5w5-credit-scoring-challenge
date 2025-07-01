"""
schema_auditor.py – Schema Auditor for Credit Risk Dataset (B5W5)
------------------------------------------------------------------------------
Performs structural audits of the transaction dataset to assess data quality,
missingness, column behavior, and data type conformity.

Core responsibilities:
  • Detects missing values across all columns
  • Identifies constant or near-constant columns
  • Inspects data types for potential type coercion
  • Flags high-cardinality variables that may impair modeling
  • Outputs diagnostics to guide cleaning and preprocessing

Used in Task 2 EDA, feature engineering design, and model guardrails.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # Core data handling


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: SchemaAuditor
# ───────────────────────────────────────────────────────────────────────────────
class SchemaAuditor:
    """
    Audits schema-level characteristics of a transaction dataset,
    providing diagnostics on data completeness, structure, and quality.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a pandas DataFrame to be audited.

        Args:
            df (pd.DataFrame): The transaction data.

        Raises:
            TypeError: If input is not a DataFrame.
        """
        # 🛡️ Validate input type for robustness
        if not isinstance(df, pd.DataFrame):
            raise TypeError("SchemaAuditor requires a pandas DataFrame as input.")
        self.df = df.copy()  # Work on a copy to avoid mutation

    def report(self) -> dict:
        """
        Computes a schema audit report with key diagnostics.

        Returns:
            dict: A dictionary containing:
                - Dataset shape
                - Null counts per column
                - Data types per column
                - Constant columns (same value throughout)
                - High-cardinality columns (90%+ unique values)

        Raises:
            ValueError: If DataFrame is empty or has no columns.
        """
        # 🛡️ Defensive check for empty data
        if self.df.empty or self.df.shape[1] == 0:
            raise ValueError("Provided DataFrame is empty or lacks columns.")

        # 📦 Initialize report container
        report = {}

        # 📐 Record basic shape of the dataset
        report["shape"] = self.df.shape

        # 🧱 Compute null counts per column
        report["null_counts"] = self.df.isnull().sum().to_dict()

        # 🔍 Extract raw data types for each column
        report["dtypes"] = self.df.dtypes.astype(str).to_dict()

        # 🧊 Identify constant columns (no variance)
        report["constant_columns"] = [
            col for col in self.df.columns if self.df[col].nunique(dropna=False) == 1
        ]

        # 🔢 Identify columns with very high cardinality
        cardinality_threshold = 0.9 * self.df.shape[0]  # e.g. 90%+ unique
        report["high_cardinality"] = [
            col
            for col in self.df.columns
            if self.df[col].nunique() >= cardinality_threshold
        ]

        # ✅ Output audit complete message
        print(f"✅ Schema audit complete: {self.df.shape[1]} columns analyzed.")

        return report
