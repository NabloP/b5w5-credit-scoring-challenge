# eda_runner.py – CLI-Compatible Runner for Task 2: Exploratory Data Analysis
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Version: 2025-07-01
# Purpose: Executes all layers of Task 2 EDA pipeline for Bati Bank’s BNPL data

# ------------------------------------------------------------------------------
# 🛠 Ensure Script Runs from Project Root
# ------------------------------------------------------------------------------
import os
import sys

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
    print("📂 Changed working directory to project root")

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✅ Added to sys.path: {project_root}")

# ------------------------------------------------------------------------------
# 📦 Imports
# ------------------------------------------------------------------------------
import pandas as pd
import warnings

from src.data_loader import CreditDataLoader
from src.eda.schema_auditor import SchemaAuditor
from src.eda.distribution_analyzer import DistributionAnalyzer
from src.eda.temporal_behavior_analyzer import TemporalBehaviorAnalyzer
from src.eda.customer_behavior_analyzer import CustomerBehaviorProfiler
from src.eda.fraud_segment_analyzer import FraudSegmentAnalyzer

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 📥 Load Raw Transaction Data
# ------------------------------------------------------------------------------
data_path = "data/raw/data.csv"
loader = CreditDataLoader(filepath=data_path)

try:
    df = loader.load()
except Exception as e:
    print(f"❌ Failed to load transaction data: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 🧾 Layer 1: Schema Audit
# ------------------------------------------------------------------------------
auditor = SchemaAuditor(df)
schema_report = auditor.report()

print("\n📦 Dataset Overview")
print(f"→ Rows: {schema_report['shape'][0]:,}")
print(f"→ Columns: {schema_report['shape'][1]}")

print("\n🧊 Constant Columns:")
print(schema_report["constant_columns"] or "✅ None")

print("\n🔢 High-Cardinality Columns:")
print(schema_report["high_cardinality"] or "✅ None")

print("\n📉 Null Counts:")
for col, count in schema_report["null_counts"].items():
    if count > 0:
        print(f"• {col}: {count:,} missing")

print("\n🔍 Data Types:")
for col, dtype in schema_report["dtypes"].items():
    print(f"• {col}: {dtype}")

# ------------------------------------------------------------------------------
# 💰 Layer 2: Numeric Distributions
# ------------------------------------------------------------------------------
dist_viz = DistributionAnalyzer(df)
print("\n📊 Plotting Amount distribution...")
dist_viz.plot_numeric("Amount")

print("\n📊 Plotting Value distribution (log-scaled)...")
dist_viz.plot_numeric("Value", log_scale=True)

# ------------------------------------------------------------------------------
# 🧮 Layer 3: Categorical Distributions
# ------------------------------------------------------------------------------
if "PricingStrategy" in df.columns:
    df["PricingStrategy"] = df["PricingStrategy"].astype("category")
    dist_viz.df["PricingStrategy"] = df["PricingStrategy"]

print("\n🛍 Top Product Categories")
dist_viz.plot_categorical("ProductCategory")

print("\n🌐 Transaction Channels")
dist_viz.plot_categorical("ChannelId")

print("\n🏢 Provider Distribution")
dist_viz.plot_categorical("ProviderId")

print("\n⚙️ Pricing Strategy Distribution")
dist_viz.plot_categorical("PricingStrategy")

print("\n💱 Currency Codes")
dist_viz.plot_categorical("CurrencyCode")

# ------------------------------------------------------------------------------
# ⏰ Layer 4: Temporal Patterns
# ------------------------------------------------------------------------------
time_analyzer = TemporalBehaviorAnalyzer(df, datetime_col="TransactionStartTime")
print("\n🕒 Hourly Volume")
time_analyzer.plot_transactions_by_hour()

print("\n📆 Daily Volume")
time_analyzer.plot_transactions_by_dayofweek()

print("\n📈 Monthly Trend")
time_analyzer.plot_monthly_trend()

# ------------------------------------------------------------------------------
# 🧑‍💼 Layer 5: RFM Profiling
# ------------------------------------------------------------------------------
rfm_profiler = CustomerBehaviorProfiler(
    df=df,
    customer_id_col="CustomerId",
    date_col="TransactionStartTime",
    value_col="Value",
)

print("\n📊 RFM Metrics Snapshot")
try:
    rfm_df = rfm_profiler.compute_rfm(snapshot_date="2024-01-01")
    print(rfm_df.head())
except Exception as e:
    print(f"⚠️ Failed to compute RFM metrics: {e}")

# ------------------------------------------------------------------------------
# 🛡 Layer 6: Fraud Segment Analysis
# ------------------------------------------------------------------------------
fraud_analyzer = FraudSegmentAnalyzer(df, fraud_col="FraudResult")
candidate_cols = ["ProductCategory", "ChannelId", "ProviderId"]

for col in candidate_cols:
    print(f"\n🔎 Fraud Rate by {col}")
    try:
        fraud_analyzer.plot_fraud_rate_by_segment(segment_col=col)
    except Exception as e:
        print(f"⚠️ Skipped {col} due to error: {e}")
