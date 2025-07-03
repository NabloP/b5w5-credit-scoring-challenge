"""
data_processing.py – Credit Risk Feature Engineering Pipeline (B5W5)
------------------------------------------------------------------------------
Transforms enriched eCommerce transaction data into a model-ready, fully numeric
feature set for Bati Bank’s BNPL credit scoring initiative.

Core responsibilities:
  • Creates aggregate customer-level behavioral features (Amount, Value, RFM)
  • Extracts temporal transaction patterns (hour, day of week, month, year)
  • Encodes categorical variables (Label Encoding + optional One-Hot Encoding)
  • Converts ID fields from string to numeric using extraction logic
  • Handles missing values and standardizes numeric features via sklearn Pipeline
  • Detects and visualizes collinearity using a correlation heatmap
  • Assembles clean, deduplicated final dataset for modeling

Used in Task 4 Feature Engineering and all downstream modeling and explainability.

Author: Nabil Mohamed
"""

# ------------------- Imports -------------------
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="notebook")

# ------------------- DataProcessor Class -------------------


class DataProcessor:
    def __init__(self, df, use_one_hot=False):
        try:
            self.df = df.copy()
            self.use_one_hot = use_one_hot
        except Exception as e:
            raise ValueError(f"❌ Invalid DataFrame provided: {e}")

    def convert_ids_to_numeric(self):
        try:
            id_cols = ["AccountId", "CustomerId", "ProductId"]
            for col in id_cols:
                if col in self.df.columns:
                    self.df[col] = (
                        self.df[col].astype(str).str.extract(r"(\d+)").astype(float)
                    )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to convert ID columns: {e}")

    def engineer_aggregate_features(self):
        try:
            agg_df = self.df.groupby("CustomerId").agg(
                {
                    "Amount": ["count", "sum", "mean", "std", "max", "min"],
                    "Value": ["sum", "mean", "std"],
                    "FraudResult": "sum",
                    "Recency": "mean",
                    "Frequency": "mean",
                    "Monetary": "mean",
                }
            )
            agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
            agg_df.reset_index(inplace=True)
            self.df = pd.merge(self.df, agg_df, on="CustomerId", how="left")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to engineer aggregate features: {e}")

    def engineer_temporal_features(self):
        try:
            self.df["TransactionStartTime"] = pd.to_datetime(
                self.df["TransactionStartTime"], errors="coerce"
            )
            self.df["TransactionHour"] = self.df["TransactionStartTime"].dt.hour
            self.df["TransactionDayOfWeek"] = self.df[
                "TransactionStartTime"
            ].dt.dayofweek
            self.df["TransactionMonth"] = self.df["TransactionStartTime"].dt.month
            self.df["TransactionYear"] = self.df["TransactionStartTime"].dt.year
        except Exception as e:
            raise RuntimeError(f"❌ Failed to engineer temporal features: {e}")

    def encode_categorical_variables(self):
        try:
            cat_cols = ["CurrencyCode", "ProviderId", "ProductCategory", "ChannelId"]
            if self.use_one_hot:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
                ohe_df = pd.DataFrame(
                    ohe.fit_transform(self.df[cat_cols]),
                    columns=ohe.get_feature_names_out(cat_cols),
                    index=self.df.index,
                )
                self.df.drop(columns=cat_cols, inplace=True)
                self.df = pd.concat([self.df, ohe_df], axis=1)
            else:
                for col in cat_cols:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
        except Exception as e:
            raise RuntimeError(f"❌ Failed to encode categorical variables: {e}")

    def validate_and_encode_remaining_categoricals(self):
        try:
            target_cols = [
                "FraudRisk",
                "BehaviorCluster",
                "BehavioralVarianceFlag",
                "RiskScore",
            ]
            for col in target_cols:
                if col in self.df.columns and not pd.api.types.is_numeric_dtype(
                    self.df[col]
                ):
                    self.df[col] = pd.factorize(self.df[col])[0]
        except Exception as e:
            raise RuntimeError(f"❌ Failed to encode behavioral flag columns: {e}")

    def handle_missing_and_scale(self):
        try:
            numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
            exclude_cols = ["risk_category", "is_high_risk"] + [
                col for col in self.df.columns if "Id" in col
            ]
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            transformer = ColumnTransformer(
                [("num", num_pipeline, numeric_cols)], remainder="passthrough"
            )

            transformed = transformer.fit_transform(self.df)

            # Fix: Ensure numeric columns stay numeric after transform
            non_numeric_cols = [
                col for col in self.df.columns if col not in numeric_cols
            ]
            new_columns = numeric_cols + non_numeric_cols

            self.df = pd.DataFrame(transformed, columns=new_columns)

            # Force numeric dtypes to avoid ML errors downstream
            for col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        except Exception as e:
            raise RuntimeError(f"❌ Failed to scale or impute numeric features: {e}")

    def assemble_final_dataset(self):
        try:
            drop_cols = [
                "TransactionId",
                "BatchId",
                "SubscriptionId",
                "TransactionStartTime",
                "CustomerRiskTag",
            ]
            self.df.drop(
                columns=[col for col in drop_cols if col in self.df.columns],
                inplace=True,
            )
            self.df.drop_duplicates(inplace=True)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to assemble final dataset: {e}")

    def run_full_pipeline(self):
        try:
            print("🚀 Starting full feature engineering pipeline...")
            self.convert_ids_to_numeric()
            print("✅ ID columns converted to numeric.")
            self.engineer_aggregate_features()
            print("✅ Aggregates engineered.")
            self.engineer_temporal_features()
            print("✅ Temporal features extracted.")
            self.encode_categorical_variables()
            self.validate_and_encode_remaining_categoricals()
            print("✅ All categorical features encoded.")
            self.handle_missing_and_scale()
            print("✅ Missing values handled and scaling applied.")
            self.assemble_final_dataset()
            print(
                f"✅ Final dataset assembled: {self.df.shape[0]:,} rows × {self.df.shape[1]:,} columns."
            )
            return self.df
        except Exception as e:
            raise RuntimeError(f"❌ Full pipeline failed: {e}")


# ------------------- CorrelationHeatmapGenerator Class -------------------


class CorrelationHeatmapGenerator:
    def __init__(self, df, threshold=0.85):
        self.df = df.select_dtypes(include="number")
        self.threshold = threshold

    def plot_heatmap(self, figsize=(15, 12), save_path=None):
        try:
            if self.df.empty or self.df.shape[1] < 2:
                raise ValueError(
                    "❌ No numeric columns available for heatmap generation."
                )

            corr = self.df.corr()
            plt.figure(figsize=figsize)
            sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
            plt.title("🔍 Correlation Heatmap of Numeric Features")
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                print(f"📁 Correlation heatmap saved to: {save_path}")
            plt.show()

            upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = [
                (col, row, val)
                for col in upper_tri.columns
                for row, val in upper_tri[col].dropna().items()
                if abs(val) > self.threshold
            ]
            if high_corr:
                print(f"⚠️ Highly correlated pairs (>|{self.threshold:.2f}|):")
                for x, y, val in high_corr:
                    print(f"  {x} ⬌ {y}: {val:.2f}")
            else:
                print("✅ No high correlations detected.")

        except Exception as e:
            raise RuntimeError(f"❌ Failed to generate correlation heatmap: {e}")
