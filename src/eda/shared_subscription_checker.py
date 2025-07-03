# shared_subscription_checker.py – Shared Subscription Diagnostic Layer (B5W5)
# ------------------------------------------------------------------------------
# Detects whether SubscriptionIds are shared across multiple AccountIds,
# signaling potential corporate or multi-user subscriptions for enhanced
# segmentation or risk modeling.

import pandas as pd  # Data handling
import numpy as np  # Numerical operations


class SharedSubscriptionChecker:
    """
    Diagnoses SubscriptionId-to-AccountId sharing patterns to detect
    potential corporate or multi-linked subscriptions.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a validated DataFrame.

        Args:
            df (pd.DataFrame): Input transactional data.

        Raises:
            TypeError: If df is not a DataFrame.
            ValueError: If required columns are missing.
        """
        if not isinstance(df, pd.DataFrame):  # Defensive type check
            raise TypeError("Input must be a pandas DataFrame.")

        required_cols = [
            "SubscriptionId",
            "AccountId",
            "CustomerId",
        ]  # Mandatory fields
        missing_cols = [col for col in required_cols if col not in df.columns]  # Check

        if missing_cols:  # Raise if missing any
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.df = df.copy()  # Defensive copy
        self.subscription_stats = None  # Placeholder for results

    def compute_subscription_sharing(self) -> pd.DataFrame:
        """
        Computes sharing metrics for each SubscriptionId.

        Returns:
            pd.DataFrame: Summary table with sharing flags.
        """
        try:
            # ✅ Group by SubscriptionId: count unique AccountIds and CustomerIds
            stats = (
                self.df.groupby("SubscriptionId")
                .agg(
                    UniqueAccountIds=("AccountId", pd.Series.nunique),
                    UniqueCustomerIds=("CustomerId", pd.Series.nunique),
                )
                .reset_index()
            )

            # ✅ Flag shared subscriptions (where more than 1 AccountId exists)
            stats["IsSharedSubscription"] = (stats["UniqueAccountIds"] > 1).astype(int)

            self.subscription_stats = stats  # Store result
            return stats.copy()  # Return defensively

        except Exception as e:
            raise RuntimeError(f"Error computing shared subscription stats: {e}")

    def print_top_shared_subscriptions(self, top_n: int = 10) -> None:
        """
        Displays top SubscriptionIds with highest AccountId sharing.

        Args:
            top_n (int): Number of top subscriptions to display.

        Raises:
            RuntimeError: If compute_subscription_sharing has not been run.
        """
        try:
            if self.subscription_stats is None:  # Defensive check
                raise RuntimeError("Subscription sharing has not been computed yet.")

            top_shared = self.subscription_stats.sort_values(
                "UniqueAccountIds", ascending=False
            ).head(top_n)

            print(f"🔍 Top {top_n} Shared Subscriptions by AccountId:")
            display(top_shared)

        except Exception as e:
            raise RuntimeError(f"Error displaying top shared subscriptions: {e}")
