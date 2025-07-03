"""
data_processing.py – Credit Risk Feature Engineering Pipeline (B5W5)
------------------------------------------------------------------------------
Transforms enriched eCommerce transaction data into a model-ready feature set
for Bati Bank’s BNPL credit scoring initiative.

Core responsibilities:
  • Creates aggregate customer-level behavioral features (Amount, Value, RFM)
  • Extracts temporal transaction patterns (hour, day of week, month, year)
  • Encodes categorical variables (Label Encoding + optional One-Hot Encoding)
  • Handles missing values and standardizes numeric features via sklearn Pipeline
  • Assembles clean, deduplicated final dataset for modeling

Used in Task 4 Feature Engineering and all downstream modeling and explainability.

Author: Nabil Mohamed
"""

# ------------------------------------------------------------------------------
# 📦 Standard Library Imports
# ------------------------------------------------------------------------------
import os  # For file path operations
import warnings  # For suppressing non-critical warnings

# ------------------------------------------------------------------------------
# 📦 Third-Party Imports
# ------------------------------------------------------------------------------
import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder  # For scaling and encoding
from sklearn.impute import SimpleImputer  # For missing value imputation
from sklearn.pipeline import Pipeline  # For reproducible pipeline chaining
from sklearn.compose import ColumnTransformer  # For applying transformers to specific columns

# Suppress warnings for cleaner outputs
warnings.filterwarnings("ignore")  # Ignore non-critical warnings

# ------------------------------------------------------------------------------
# 📌 DataProcessor Class Definition
# ------------------------------------------------------------------------------

class DataProcessor:
    """
    A modular, reproducible class for transforming transaction data into
    model-ready features for BNPL credit risk modeling.
    """

    def __init__(self, df, use_one_hot=False):
        """
        Initialize the DataProcessor with input DataFrame.

        Args:
            df (pd.DataFrame): Enriched transaction dataset with proxy labels.
            use_one_hot (bool): If True, apply One-Hot Encoding instead of Label Encoding.
        """
        try:
            self.df = df.copy()  # Defensive copy of input data
            self.use_one_hot = use_one_hot  # Store encoding choice
        except Exception as e:
            raise ValueError(f"❌ Invalid DataFrame provided: {e}")

    def engineer_aggregate_features(self):
        """
        Create aggregate transaction features at the customer level.
        """
        try:
            # Group transactions by CustomerId and compute aggregates
            agg_df = self.df.groupby('CustomerId').agg({
                'Amount': ['count', 'sum', 'mean', 'std', 'max', 'min'],
                'Value': ['sum', 'mean', 'std'],
                'FraudResult': 'sum',
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
            })

            # Flatten MultiIndex columns
            agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
            agg_df.reset_index(inplace=True)

            # Merge aggregates back into main DataFrame
            self.df = pd.merge(self.df, agg_df, on='CustomerId', how='left')
        except Exception as e:
            raise RuntimeError(f"❌ Failed to engineer aggregate features: {e}")

    def engineer_temporal_features(self):
        """
        Extract temporal transaction features.
        """
        try:
            self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])  # Convert to datetime
            self.df['TransactionHour'] = self.df['TransactionStartTime'].dt.hour  # Extract hour
            self.df['TransactionDayOfWeek'] = self.df['TransactionStartTime'].dt.dayofweek  # Extract weekday
            self.df['TransactionMonth'] = self.df['TransactionStartTime'].dt.month  # Extract month
            self.df['TransactionYear'] = self.df['TransactionStartTime'].dt.year  # Extract year (missing in original)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to engineer temporal features: {e}")

    def encode_categorical_variables(self):
        """
        Apply either Label Encoding or One-Hot Encoding to selected categorical columns.
        """
        try:
            cat_cols = ['CurrencyCode', 'ProviderId', 'ProductCategory', 'ChannelId']

            if self.use_one_hot:
                # Apply One-Hot Encoding with handle_unknown safeguard
                ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
                ohe_df = pd.DataFrame(ohe.fit_transform(self.df[cat_cols]), 
                                      columns=ohe.get_feature_names_out(cat_cols), 
                                      index=self.df.index)
                self.df = pd.concat([self.df.drop(columns=cat_cols), ohe_df], axis=1)
            else:
                # Apply Label Encoding for rapid prototyping
                for col in cat_cols:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
        except Exception as e:
            raise RuntimeError(f"❌ Failed to encode categorical variables: {e}")

    def handle_missing_and_scale(self):
        """
        Apply missing value imputation and feature scaling using sklearn Pipeline.
        """
        try:
            # Identify numeric columns (excluding IDs and targets)
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            exclude_cols = ['risk_category', 'is_high_risk'] + [col for col in self.df.columns if 'Id' in col]
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

            # Define pipeline: Imputer + Scaler
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Apply pipeline to numeric columns using ColumnTransformer
            transformer = ColumnTransformer(transformers=[
                ('num', num_pipeline, numeric_cols)
            ], remainder='passthrough')

            # Fit-transform data
            transformed = transformer.fit_transform(self.df)

            # Reconstruct DataFrame preserving non-numeric columns
            new_columns = numeric_cols + [col for col in self.df.columns if col not in numeric_cols]
            self.df = pd.DataFrame(transformed, columns=new_columns)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to handle missing values or scale features: {e}")

    def assemble_final_dataset(self):
        """
        Clean and finalize dataset for model consumption.
        """
        try:
            drop_cols = ['TransactionId', 'BatchId', 'SubscriptionId', 'TransactionStartTime', 'CustomerRiskTag']
            self.df.drop(columns=[col for col in drop_cols if col in self.df.columns], inplace=True)
            self.df.drop_duplicates(inplace=True)  # Ensure no duplicate rows
        except Exception as e:
            raise RuntimeError(f"❌ Failed to assemble final dataset: {e}")

    def run_full_pipeline(self):
        """
        Run the complete feature engineering pipeline with verbose diagnostics.
        """
        try:
            print("🚀 Starting feature engineering pipeline...")
            self.engineer_aggregate_features()
            print("✅ Aggregate features created.")

            self.engineer_temporal_features()
            print("✅ Temporal features extracted.")

            self.encode_categorical_variables()
            print(f"✅ Categorical variables encoded using {'One-Hot' if self.use_one_hot else 'Label'} Encoding.")

            self.handle_missing_and_scale()
            print("✅ Missing values handled and numerical features scaled.")

            self.assemble_final_dataset()
            print(f"✅ Final dataset assembled. Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]:,} columns.")

            return self.df
        except Exception as e:
            raise RuntimeError(f"❌ Feature engineering pipeline failed: {e}")
