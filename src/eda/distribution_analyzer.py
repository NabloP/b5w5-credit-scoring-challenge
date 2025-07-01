"""
distribution_analyzer.py – Univariate Distribution Visualizer (B5W5)
------------------------------------------------------------------------------
Visualizes the distribution of numerical and categorical variables to reveal
skews, outliers, and dominant categories in the eCommerce credit risk dataset.

Core responsibilities:
  • Histogram + boxplot for numeric variables
  • Bar plots for top-K categorical levels
  • Defensive handling of missing or misclassified columns
  • Automated seaborn-based visualization with axis formatting

Used in Task 2 EDA for exploring monetary values, user behavior, and platform activity.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For DataFrame operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For statistical visualizations


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: DistributionAnalyzer
# ───────────────────────────────────────────────────────────────────────────────
class DistributionAnalyzer:
    """
    Visualizes univariate distributions of numeric and categorical columns
    using histograms, boxplots, and bar charts.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a pandas DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Raises:
            TypeError: If input is not a DataFrame.
        """
        # 🛡️ Defensive type check
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = df.copy()  # Avoid mutating original data

        # 🎨 Set default seaborn style
        sns.set(style="whitegrid", context="notebook")

    def plot_numeric(self, column: str, log_scale: bool = False) -> None:
        """
        Plots histogram and boxplot for a numeric column.

        Args:
            column (str): Column name to visualize.
            log_scale (bool): Whether to use log scale on x-axis.

        Raises:
            ValueError: If column is missing or non-numeric.
        """
        # 🛡️ Check if column exists
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        # 🛡️ Check if column is numeric
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' must be numeric for this plot.")

        # 🎯 Drop NA values for clean plotting
        series = self.df[column].dropna()

        # 📊 Plot histogram + boxplot
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.suptitle(f"Distribution of {column}", fontsize=14, fontweight="bold")

        # Histogram (top)
        sns.histplot(series, bins=50, ax=axes[0], kde=True, color="skyblue")
        axes[0].set_ylabel("Frequency")
        if log_scale:
            axes[0].set_xscale("log")
            axes[0].set_title(f"Histogram of {column} (log scale)", fontsize=12)
        else:
            axes[0].set_title(f"Histogram of {column}", fontsize=12)

        # Boxplot (bottom)
        sns.boxplot(x=series, ax=axes[1], color="lightcoral")
        axes[1].set_xlabel(column)

        # 📦 Render plots
        plt.tight_layout()
        plt.show()

    def plot_categorical(self, column: str, top_k: int = 10) -> None:
        """
        Plots a bar chart of the top K categories in a categorical column.

        Args:
            column (str): Column name to visualize.
            top_k (int): Number of most frequent categories to plot.

        Raises:
            ValueError: If column is missing or not object/categorical type.
        """
        # 🛡️ Check if column exists
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        # 🛡️ Check if column is categorical or object
        if not pd.api.types.is_object_dtype(
            self.df[column]
        ) and not pd.api.types.is_categorical_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' must be categorical or object-type.")

        # 🧮 Compute value counts and truncate
        value_counts = self.df[column].value_counts().head(top_k)

        # 📊 Plot bar chart
        plt.figure(figsize=(10, 5))
        sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis")
        plt.title(f"Top {top_k} Most Frequent Categories in '{column}'", fontsize=13)
        plt.xlabel("Frequency")
        plt.ylabel(column)
        plt.tight_layout()
        plt.show()
