"""
data_loader.py – Credit Risk Transaction Data Loader (B5W5)
------------------------------------------------------------------------------
Safely loads raw eCommerce transactional data for Bati Bank's credit scoring initiative.
Performs robust file validation, column inspection, and CSV parsing.

Core responsibilities:
  • Validates file path and input type
  • Loads standard comma-separated CSV files from the dataset
  • Checks for structural integrity (non-empty, tabular)
  • Provides helpful diagnostics on shape and columns

Used in Task 2 EDA, proxy target engineering (Task 4), and all downstream modeling tasks.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os  # For file path checks

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For data loading and frame validation


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: CreditDataLoader
# ───────────────────────────────────────────────────────────────────────────────
class CreditDataLoader:
    """
    OOP wrapper for safely loading the eCommerce transactional dataset.
    Ensures CSV parsing and structural validity before further use.
    """

    def __init__(self, filepath: str):
        """
        Initialize the loader with the expected path to the transaction CSV.

        Args:
            filepath (str): Full path to the .csv file.

        Raises:
            TypeError: If the filepath is not a string.
            FileNotFoundError: If the file does not exist at the given path.
        """
        # 🛡️ Validate input type
        if not isinstance(filepath, str):
            raise TypeError(f"filepath must be str, got {type(filepath)}")

        # 🛡️ Ensure the file exists at the given path
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Cannot find file at: {filepath}")

        # ✅ Store path for later use
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        """
        Loads the credit transaction dataset and validates its structure.

        Returns:
            pd.DataFrame: Parsed DataFrame with structural validation passed.

        Raises:
            ValueError: If the file is empty or format is incorrect.
        """
        try:
            # 📥 Load as comma-separated CSV
            df = pd.read_csv(self.filepath)

            # ❌ Raise error if DataFrame is completely empty
            if df.empty:
                raise ValueError("Loaded DataFrame is empty.")

            # ✅ Print success diagnostics
            print(
                f"✅ Transaction dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns"
            )
            return df

        except pd.errors.ParserError as e:
            raise ValueError(f"CSV parsing failed: {e}")

        except Exception as e:
            raise RuntimeError(f"Unexpected error during data load: {e}")
