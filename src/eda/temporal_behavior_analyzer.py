"""
temporal_behavior_analyzer.py – Time-Based Transaction Trend Visualizer (B5W5)
------------------------------------------------------------------------------
Analyzes temporal behavioral patterns in the credit risk transaction dataset,
including user activity by hour, day, and monthly volume trends.

Core responsibilities:
  • Converts timestamp column to datetime format
  • Extracts hour, day-of-week, and month/year components
  • Visualizes transaction trends over time
  • Supports fraud/segment overlays for advanced insights

Used in Task 2 EDA and proxy label design based on recency or activity clustering.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For datetime parsing and grouping
import seaborn as sns  # For plotting
import matplotlib.pyplot as plt  # For visual formatting


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: TemporalBehaviorAnalyzer
# ───────────────────────────────────────────────────────────────────────────────
class TemporalBehaviorAnalyzer:
    """
    Analyzes transaction behavior over time and visualizes
    user activity by hour, day, and month.
    """

    def __init__(self, df: pd.DataFrame, datetime_col: str):
        """
        Initialize with a DataFrame and the name of a datetime column.

        Args:
            df (pd.DataFrame): The transaction dataset.
            datetime_col (str): Name of the timestamp column.

        Raises:
            ValueError: If the column is missing or cannot be parsed to datetime.
        """
        if datetime_col not in df.columns:
            raise ValueError(f"Column '{datetime_col}' not found in DataFrame.")

        # Attempt to convert the datetime column safely
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        except Exception as e:
            raise ValueError(f"Failed to parse '{datetime_col}' to datetime: {e}")

        # ✅ Store copy with extracted time features
        self.df = df.copy()
        self.datetime_col = datetime_col

        # ⏱️ Add derived time features for grouping
        self.df["Hour"] = self.df[datetime_col].dt.hour
        self.df["DayOfWeek"] = self.df[datetime_col].dt.day_name()
        self.df["Month"] = self.df[datetime_col].dt.to_period("M").astype(str)

        # 🎨 Default seaborn style
        sns.set(style="whitegrid", context="notebook")

    def plot_transactions_by_hour(self) -> None:
        """
        Visualizes transaction volume across the 24-hour day.
        """
        plt.figure(figsize=(10, 4))
        sns.countplot(data=self.df, x="Hour", color="skyblue")
        plt.title("🕒 Transaction Volume by Hour of Day")
        plt.xlabel("Hour (0–23)")
        plt.ylabel("Transaction Count")
        plt.tight_layout()
        plt.show()

    def plot_transactions_by_dayofweek(self) -> None:
        """
        Visualizes transaction volume across weekdays.
        """
        # Preserve weekday order
        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        plt.figure(figsize=(10, 4))
        sns.countplot(data=self.df, x="DayOfWeek", order=weekday_order, color="orchid")
        plt.title("📆 Transaction Volume by Day of Week")
        plt.xlabel("Day")
        plt.ylabel("Transaction Count")
        plt.tight_layout()
        plt.show()

    def plot_monthly_trend(self) -> None:
        """
        Plots number of transactions per month over time.
        """
        # Aggregate by month
        monthly_counts = self.df["Month"].value_counts().sort_index()

        # Plot line chart
        plt.figure(figsize=(12, 4))
        sns.lineplot(
            x=monthly_counts.index, y=monthly_counts.values, marker="o", color="teal"
        )
        plt.xticks(rotation=45)
        plt.title("📈 Monthly Transaction Trend")
        plt.xlabel("Month")
        plt.ylabel("Number of Transactions")
        plt.tight_layout()
        plt.show()
