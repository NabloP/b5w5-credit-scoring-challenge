"""
account_anomaly_detector.py – Shared vs. Individual Account Tagger and Diagnostic (B5W5)
------------------------------------------------------------------------------
Performs anomaly detection on transactional AccountId–CustomerId–SubscriptionId
relationships to flag shared accounts and confirm mapping behaviors.

Key Features:
  • Identifies accounts with unusually high numbers of CustomerIds (IQR method)
  • Flags accounts with multiple SubscriptionIds
  • Tags shared vs. individual accounts directly into the DataFrame
  • Provides clean summary tables for investigation

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: AccountAnomalyDetector
# ───────────────────────────────────────────────────────────────────────────────
class AccountAnomalyDetector:
    """
    Class for detecting shared accounts and tagging account anomalies
    in transactional credit datasets.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a validated DataFrame.

        Args:
            df (pd.DataFrame): Input transaction-level data.

        Raises:
            TypeError: If df is not a DataFrame.
            ValueError: If required columns are missing.
        """
        if not isinstance(df, pd.DataFrame):  # Validate type
            raise TypeError("Input must be a pandas DataFrame.")

        required_cols = [
            "AccountId",
            "CustomerId",
            "SubscriptionId",
        ]  # Required columns
        missing_cols = [
            col for col in required_cols if col not in df.columns
        ]  # Check for missing

        if missing_cols:  # Raise if any required columns missing
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.df = df.copy()  # Defensive copy to avoid mutating original data
        self.account_stats = None  # Placeholder for computed stats

    def compute_account_statistics(self) -> pd.DataFrame:
        """
        Computes per-AccountId statistics for unique customers, subscriptions, and transaction counts.

        Returns:
            pd.DataFrame: Summary stats table per AccountId.
        """
        try:
            # ✅ Group by AccountId and compute unique counts
            stats = (
                self.df.groupby("AccountId")
                .agg(
                    {
                        "CustomerId": pd.Series.nunique,  # Unique customers per account
                        "SubscriptionId": pd.Series.nunique,  # Unique subscriptions per account
                        "TransactionId": "count",  # Total transaction count
                    }
                )
                .rename(
                    columns={
                        "CustomerId": "UniqueCustomers",
                        "SubscriptionId": "UniqueSubscriptions",
                        "TransactionId": "TransactionCount",
                    }
                )
            )

            self.account_stats = stats  # Store result for reuse
            return stats  # Return DataFrame

        except Exception as e:
            raise RuntimeError(f"Error computing account statistics: {e}")

    def detect_shared_accounts(self) -> pd.DataFrame:
        """
        Detects accounts with unusually high customer counts (IQR method) or multiple subscriptions.

        Returns:
            pd.DataFrame: DataFrame with shared accounts flagged.
        """
        try:
            if self.account_stats is None:  # Defensive: Check stats computed
                self.compute_account_statistics()

            # ✅ Compute IQR for UniqueCustomers
            q1 = self.account_stats["UniqueCustomers"].quantile(0.25)  # 25th percentile
            q3 = self.account_stats["UniqueCustomers"].quantile(0.75)  # 75th percentile
            iqr = q3 - q1  # Interquartile range
            upper_bound = q3 + 1.5 * iqr  # Upper outlier threshold

            # ✅ Flag accounts with unusually high customers
            self.account_stats["HighCustomerOutlier"] = np.where(
                self.account_stats["UniqueCustomers"] > upper_bound, 1, 0
            )

            # ✅ Flag accounts with multiple subscriptions
            self.account_stats["MultipleSubscriptions"] = np.where(
                self.account_stats["UniqueSubscriptions"] > 1, 1, 0
            )

            # ✅ Flag shared accounts if either anomaly is present
            self.account_stats["IsSharedAccount"] = np.where(
                (self.account_stats["HighCustomerOutlier"] == 1)
                | (self.account_stats["MultipleSubscriptions"] == 1),
                1,
                0,
            )

            return self.account_stats.copy()  # Return defensively

        except Exception as e:
            raise RuntimeError(f"Error detecting shared accounts: {e}")

    def tag_dataframe(self) -> pd.DataFrame:
        """
        Tags the original DataFrame with IsSharedAccount flags based on detected anomalies.

        Returns:
            pd.DataFrame: The original DataFrame with an added IsSharedAccount column.
        """
        try:
            if self.account_stats is None:  # Defensive check
                self.detect_shared_accounts()

            # ✅ Create mapping dictionary for IsSharedAccount
            shared_map = self.account_stats[
                "IsSharedAccount"
            ].to_dict()  # AccountId → 0/1

            # ✅ Map values back to original DataFrame
            self.df["IsSharedAccount"] = (
                self.df["AccountId"].map(shared_map).fillna(0).astype(int)
            )  # Ensure binary flag

            return self.df.copy()  # Return updated DataFrame

        except Exception as e:
            raise RuntimeError(f"Error tagging DataFrame: {e}")

    def print_anomaly_summary(self) -> None:
        """
        Prints a concise summary of detected shared accounts.

        Raises:
            RuntimeError: If detection has not been run.
        """
        try:
            if self.account_stats is None:
                raise RuntimeError(
                    "Anomaly detection has not been performed. Run detect_shared_accounts() first."
                )

            # ✅ Summary counts
            shared_count = self.account_stats["IsSharedAccount"].sum()  # Total flagged
            total_accounts = len(self.account_stats)  # Total accounts

            print(
                f"🔍 Shared Accounts Detected: {shared_count:,} of {total_accounts:,} total accounts."
            )

            # ✅ Optional: Display top suspicious accounts
            suspicious = self.account_stats[self.account_stats["IsSharedAccount"] == 1]
            display(suspicious.sort_values("UniqueCustomers", ascending=False).head(10))

        except Exception as e:
            raise RuntimeError(f"Error printing anomaly summary: {e}")
