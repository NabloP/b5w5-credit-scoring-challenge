"""
schema_auditor.py – DataFrame Schema Diagnostic Tool (B5W5)
------------------------------------------------------------------------------
Provides detailed structural diagnostics on credit risk transactional DataFrames.
Summarizes missingness, uniqueness, type stability, and schema integrity.

Core responsibilities:
  • Computes per-column metrics: dtype, n_unique, % missing, constant-value flags
  • Flags high-null fields with severity bands for risk assessment
  • Checks for duplicate identifier values to ensure record uniqueness
  • Supports styled schema summaries for EDA and documentation
  • Raises clear, actionable exceptions for invalid inputs

Used in Task 2 EDA, feature engineering audits, and modeling guardrails.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
from typing import List

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: SchemaAuditor
# ───────────────────────────────────────────────────────────────────────────────
class SchemaAuditor:
    """
    Class for performing schema-level structural diagnostics on credit risk datasets.
    Provides missingness, uniqueness, type, and stability checks.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the schema auditor with the provided DataFrame.

        Args:
            df (pd.DataFrame): Transaction-level dataset for BNPL credit risk modeling.

        Raises:
            TypeError: If input is not a pandas DataFrame.
            ValueError: If the DataFrame is empty or has no columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, received {type(df)}.")

        if df.empty or df.shape[1] == 0:
            raise ValueError("Input DataFrame is empty or contains no columns.")

        self.df = df.copy()  # Use defensive copying
        self.schema_df = None  # Stores the generated schema summary

    def summarize_schema(self) -> pd.DataFrame:
        """
        Computes and stores per-column schema metrics.

        Returns:
            pd.DataFrame: DataFrame summarizing schema characteristics.
        """
        try:
            schema = pd.DataFrame(
                {
                    "dtype": self.df.dtypes.astype(str),
                    "n_unique": self.df.nunique(dropna=False),
                    "n_missing": self.df.isna().sum(),
                }
            )

            schema["%_missing"] = (schema["n_missing"] / len(self.df) * 100).round(2)
            schema["is_constant"] = schema["n_unique"] <= 1
            schema["high_null_flag"] = pd.cut(
                schema["%_missing"],
                bins=[-1, 0, 20, 50, 100],
                labels=["✅ OK", "🟡 Moderate", "🟠 High", "🔴 Critical"],
            )

            self.schema_df = schema.sort_values("%_missing", ascending=False)
            return self.schema_df

        except Exception as e:
            raise RuntimeError(f"Error generating schema summary: {e}")

    def styled_summary(self):
        """
        Creates a visually enhanced schema summary for Jupyter display.

        Returns:
            pd.io.formats.style.Styler: Styled DataFrame with color cues.
        """
        try:
            if self.schema_df is None:
                self.summarize_schema()

            styled = (
                self.schema_df.style.background_gradient(
                    subset="%_missing", cmap="OrRd"
                )
                .applymap(
                    lambda val: (
                        "background-color: gold; font-weight: bold;" if val else ""
                    ),
                    subset=["is_constant"],
                )
                .format({"%_missing": "{:.2f}"})
            )
            return styled

        except Exception as e:
            raise RuntimeError(f"Error styling schema summary: {e}")

    def print_diagnostics(self) -> None:
        """
        Prints a concise summary of schema health including missingness and stability flags.

        Raises:
            RuntimeError: If diagnostics cannot be generated.
        """
        try:
            if self.schema_df is None:
                self.summarize_schema()

            n_const = self.schema_df["is_constant"].sum()
            n_null_20 = (self.schema_df["%_missing"] > 20).sum()
            n_null_50 = (self.schema_df["%_missing"] > 50).sum()

            print("\n📝 Schema Diagnostics:")
            print(f"• Constant-value columns:    {n_const}")
            print(f"• Columns >20% missing:      {n_null_20}")
            print(f"• Columns >50% missing:      {n_null_50}")

        except Exception as e:
            raise RuntimeError(f"Error printing diagnostics: {e}")

    def check_duplicate_ids(self, id_columns: List[str]) -> None:
        """
        Checks for duplicates in key identifier columns.

        Args:
            id_columns (List[str]): List of identifier columns to check.

        Raises:
            ValueError: If any specified column does not exist in the DataFrame.
        """
        try:
            for col in id_columns:
                if col not in self.df.columns:
                    raise ValueError(f"Identifier column '{col}' not found.")

                n_duplicates = self.df[col].duplicated().sum()
                print(
                    f"• {col}: {n_duplicates:,} duplicates"
                    + (" ⚠️ Potential integrity risk." if n_duplicates > 0 else "")
                )
        except Exception as e:
            raise RuntimeError(f"Error checking duplicate IDs: {e}")
