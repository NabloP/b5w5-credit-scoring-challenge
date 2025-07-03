"""
distribution_analyzer.py – Enhanced Distribution Visualizer with Dynamic Capping (B5W5)
------------------------------------------------------------------------------
Visualizes univariate distributions of numeric and categorical features with:
  • Optional capping at any upper percentile (default P95) for visual clarity
  • Option to apply permanent capping to the working DataFrame
  • Optional stratification (hue) for subgroup analysis
  • Robust defensive programming and inline documentation

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For DataFrame handling
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For stylish visualization
import numpy as np  # For numerical operations


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: DistributionAnalyzer
# ───────────────────────────────────────────────────────────────────────────────
class DistributionAnalyzer:
    """
    Class for visualizing numeric and categorical distributions
    with optional capping and stratification.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with validated DataFrame.

        Args:
            df (pd.DataFrame): Input transaction data.

        Raises:
            TypeError: If df is not a DataFrame.
        """
        if not isinstance(df, pd.DataFrame):  # Defensive type check
            raise TypeError("Input must be a pandas DataFrame.")  # Raise error

        self.df = df.copy()  # Defensive copy to preserve original
        sns.set(style="whitegrid", context="notebook")  # Apply Seaborn style

    def cap_numeric_column(
        self, column: str, cap_percentile: float = 0.95
    ) -> pd.Series:
        """
        Returns a capped version of a numeric Series at the specified percentile.

        Args:
            column (str): Name of numeric column.
            cap_percentile (float): Percentile for upper cap (0-1).

        Returns:
            pd.Series: Capped series.

        Raises:
            ValueError: If invalid input.
        """
        if column not in self.df.columns:  # Check column presence
            raise ValueError(f"Column '{column}' not found in DataFrame.")  # Raise

        if not pd.api.types.is_numeric_dtype(self.df[column]):  # Check type
            raise ValueError(f"Column '{column}' must be numeric.")  # Raise

        cap_value = self.df[column].quantile(cap_percentile)  # Compute cap value
        return self.df[column].clip(upper=cap_value)  # Return capped Series

    def plot_numeric(
        self,
        column: str,
        log_scale: bool = False,
        apply_cap: bool = False,
        cap_percentile: float = 0.95,
        hue: str = None,
    ) -> None:
        """
        Plots histogram and boxplot of a numeric feature with optional capping and stratification.

        Args:
            column (str): Numeric column to visualize.
            log_scale (bool): Apply log scale to x-axis.
            apply_cap (bool): Apply capping for visualization.
            cap_percentile (float): Percentile threshold for capping.
            hue (str): Optional categorical hue.

        Raises:
            ValueError: For invalid inputs.
        """
        try:
            # Defensive validation
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found.")
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                raise ValueError(f"Column '{column}' must be numeric.")

            # Apply capping for visual use if requested
            plot_data = self.df.copy()
            if apply_cap:
                plot_data[column] = self.cap_numeric_column(
                    column, cap_percentile
                )  # Apply cap

            # Drop NAs while preserving hue if used
            plot_data = (
                plot_data[[column, hue]].dropna()
                if hue
                else plot_data[[column]].dropna()
            )

            # Create subplots
            fig, axes = plt.subplots(
                2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}
            )
            fig.suptitle(f"Distribution of {column}", fontsize=16, fontweight="bold")

            # Plot histogram with or without hue
            sns.histplot(
                data=plot_data,
                x=column,
                hue=hue,
                kde=True,
                palette="pastel",
                ax=axes[0],
                bins=50,
            )
            axes[0].set_ylabel("Frequency", fontsize=11)
            axes[0].set_xlabel("")  # No x-label on top plot

            # Add optional log scale
            title_suffix = ""
            if log_scale:
                axes[0].set_xscale("log")
                title_suffix += " (Log Scale)"
            if apply_cap:
                title_suffix += f" (Capped @ {int(cap_percentile * 100)}%)"

            axes[0].set_title(f"Histogram of {column}{title_suffix}", fontsize=12)

            # Plot boxplot (with or without hue)
            if hue:
                sns.boxplot(
                    data=plot_data,
                    x=column,
                    y=hue,
                    palette="pastel",
                    ax=axes[1],
                    orient="h",
                )
            else:
                sns.boxplot(x=plot_data[column], ax=axes[1], color="skyblue")

            axes[1].set_xlabel(column, fontsize=11)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            raise RuntimeError(f"Error plotting numeric feature '{column}': {e}")

    def plot_categorical(self, column: str, top_k: int = 10) -> None:
        """
        Plots bar chart of top K categories.

        Args:
            column (str): Categorical column name.
            top_k (int): Number of top categories to plot.

        Raises:
            ValueError: For invalid inputs.
        """
        try:
            # Validate column
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found.")

            if not (
                pd.api.types.is_object_dtype(self.df[column])
                or pd.api.types.is_categorical_dtype(self.df[column])
            ):
                raise ValueError(
                    f"Column '{column}' must be categorical or object-type."
                )

            # Compute top categories
            counts = self.df[column].value_counts().head(top_k)

            plt.figure(figsize=(10, 5))
            sns.barplot(x=counts.values, y=counts.index, palette="muted")
            plt.title(
                f"Top {top_k} Categories in '{column}'", fontsize=14, weight="bold"
            )
            plt.xlabel("Frequency", fontsize=11)
            plt.ylabel(column, fontsize=11)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            raise RuntimeError(f"Error plotting categorical feature '{column}': {e}")

    def apply_capping_to_dataframe(
        self, column: str, cap_percentile: float = 0.95
    ) -> pd.DataFrame:
        """
        Applies permanent capping to the DataFrame for modeling or proxy creation.

        Args:
            column (str): Column to cap.
            cap_percentile (float): Percentile cap threshold.

        Returns:
            pd.DataFrame: New DataFrame with capped column.

        Raises:
            ValueError: For invalid inputs.
        """
        try:
            capped_df = self.df.copy()
            capped_df[column] = self.cap_numeric_column(column, cap_percentile)
            return capped_df  # Return safely capped DataFrame

        except Exception as e:
            raise RuntimeError(f"Error applying capping to column '{column}': {e}")
