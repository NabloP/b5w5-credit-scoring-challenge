"""
relationship_explorer.py – Enhanced Customer-Account-Subscription Relationship Visualizer with Dynamic Frequency Capping (B5W5)
------------------------------------------------------------------------------
Performs detailed structural exploration of key identifier relationships within
transactional credit data for risk modeling and feature engineering.

Supports visual diagnostics of:
  • Number of AccountIds per CustomerId
  • Number of SubscriptionIds per CustomerId
  • Number of SubscriptionIds per AccountId
  • Number of CustomerIds per AccountId (reverse mapping)
  • Frequency of transactions per CustomerId (with optional percentile capping)

Outputs include polished, visually enhanced plots and summary tables to
inform aggregation choices, segmentation, and defensible proxy target design.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: RelationshipExplorer
# ───────────────────────────────────────────────────────────────────────────────
class RelationshipExplorer:
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        required_cols = ["CustomerId", "AccountId", "SubscriptionId", "TransactionId"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.df = df.copy()
        sns.set(style="whitegrid", context="notebook")

    def plot_transaction_frequency(
        self, apply_cap: bool = False, cap_percentile: float = 0.95, hue: str = None
    ) -> None:
        try:
            txn_counts = self.df.groupby("CustomerId")["TransactionId"].count()

            if apply_cap:
                cap_value = txn_counts.quantile(cap_percentile)
                txn_counts = txn_counts.clip(upper=cap_value)

            plot_df = (
                self.df.copy()
                .groupby("CustomerId")
                .agg(
                    {"TransactionId": "count", hue: "max"}
                    if hue
                    else {"TransactionId": "count"}
                )
                .reset_index()
            )

            if apply_cap:
                cap_value = plot_df["TransactionId"].quantile(cap_percentile)
                plot_df["TransactionId"] = plot_df["TransactionId"].clip(
                    upper=cap_value
                )

            plt.figure(figsize=(10, 6))
            sns.histplot(
                data=plot_df,
                x="TransactionId",
                hue=hue,
                kde=False,
                bins=30,
                palette="muted",
            )
            plt.title(
                "📊 Transaction Frequency per CustomerId", fontsize=14, weight="bold"
            )
            plt.xlabel("Number of Transactions", fontsize=12)
            plt.ylabel("Number of Customers", fontsize=12)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            raise RuntimeError(f"Error plotting transaction frequency: {e}")

    def plot_all_relationships(
        self, apply_cap: bool = False, cap_percentile: float = 0.95, hue: str = None
    ) -> None:
        try:
            self.plot_accounts_per_customer()
            self.plot_subscriptions_per_customer()
            self.plot_subscriptions_per_account()
            self.plot_customers_per_account()
            self.plot_transaction_frequency(
                apply_cap=apply_cap, cap_percentile=cap_percentile, hue=hue
            )

        except Exception as e:
            raise RuntimeError(f"Error generating relationship plots: {e}")

    def compute_summary_stats(
        self, apply_cap: bool = False, cap_percentile: float = 0.95
    ) -> pd.io.formats.style.Styler:
        try:
            account_counts = self.df.groupby("CustomerId")["AccountId"].nunique()
            subscription_counts = self.df.groupby("CustomerId")[
                "SubscriptionId"
            ].nunique()
            txn_counts = self.df.groupby("CustomerId")["TransactionId"].count()

            if apply_cap:
                cap_value = txn_counts.quantile(cap_percentile)
                txn_counts = txn_counts.clip(upper=cap_value)

            summary_df = pd.DataFrame(
                {
                    "AccountsPerCustomer": account_counts,
                    "SubscriptionsPerCustomer": subscription_counts,
                    "TransactionsPerCustomer": txn_counts,
                }
            )

            styled_summary = (
                summary_df.describe()
                .rename(index={"count": "NumCustomers"})
                .round(2)
                .style.background_gradient(cmap="coolwarm")
            )

            return styled_summary

        except Exception as e:
            raise RuntimeError(f"Error computing summary statistics: {e}")

    def compute_account_level_stats(self) -> pd.io.formats.style.Styler:
        try:
            customer_counts = self.df.groupby("AccountId")["CustomerId"].nunique()
            subscription_counts = self.df.groupby("AccountId")[
                "SubscriptionId"
            ].nunique()
            txn_counts = self.df.groupby("AccountId")["TransactionId"].count()

            summary_df = pd.DataFrame(
                {
                    "CustomersPerAccount": customer_counts,
                    "SubscriptionsPerAccount": subscription_counts,
                    "TransactionsPerAccount": txn_counts,
                }
            )

            styled_summary = (
                summary_df.describe()
                .rename(index={"count": "NumAccounts"})
                .round(2)
                .style.background_gradient(cmap="coolwarm")
            )

            return styled_summary

        except Exception as e:
            raise RuntimeError(f"Error computing account-level statistics: {e}")

    # Existing plotting methods remain unchanged
    def plot_accounts_per_customer(self) -> None:
        try:
            account_counts = self.df.groupby("CustomerId")["AccountId"].nunique()
            count_distribution = account_counts.value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=count_distribution.index,
                y=count_distribution.values,
                palette="Blues_d",
            )
            plt.title(
                "🔗 Number of AccountIds per CustomerId", fontsize=14, weight="bold"
            )
            plt.xlabel("Unique AccountIds", fontsize=12)
            plt.ylabel("Number of Customers", fontsize=12)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            raise RuntimeError(f"Error plotting accounts per customer: {e}")

    def plot_subscriptions_per_customer(self) -> None:
        try:
            subscription_counts = self.df.groupby("CustomerId")[
                "SubscriptionId"
            ].nunique()
            count_distribution = subscription_counts.value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=count_distribution.index,
                y=count_distribution.values,
                palette="Greens_d",
            )
            plt.title(
                "🔗 Number of SubscriptionIds per CustomerId",
                fontsize=14,
                weight="bold",
            )
            plt.xlabel("Unique SubscriptionIds", fontsize=12)
            plt.ylabel("Number of Customers", fontsize=12)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            raise RuntimeError(f"Error plotting subscriptions per customer: {e}")

    def plot_subscriptions_per_account(self) -> None:
        try:
            subscription_counts = self.df.groupby("AccountId")[
                "SubscriptionId"
            ].nunique()
            count_distribution = subscription_counts.value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=count_distribution.index,
                y=count_distribution.values,
                palette="Purples_d",
            )
            plt.title(
                "🔗 Number of SubscriptionIds per AccountId", fontsize=14, weight="bold"
            )
            plt.xlabel("Unique SubscriptionIds", fontsize=12)
            plt.ylabel("Number of Accounts", fontsize=12)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            raise RuntimeError(f"Error plotting subscriptions per account: {e}")

    def plot_customers_per_account(self) -> None:
        try:
            customer_counts = self.df.groupby("AccountId")["CustomerId"].nunique()
            count_distribution = customer_counts.value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=count_distribution.index,
                y=count_distribution.values,
                palette="Oranges_d",
            )
            plt.title(
                "🔄 Number of CustomerIds per AccountId", fontsize=14, weight="bold"
            )
            plt.xlabel("Unique CustomerIds", fontsize=12)
            plt.ylabel("Number of Accounts", fontsize=12)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            raise RuntimeError(f"Error plotting customers per account: {e}")
