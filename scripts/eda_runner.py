# eda_runner.py – Final Modular CLI Runner for Task 2: Exploratory Data Analysis (B5W5)
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Version: 2025-07-03 (Final Submission Version)
# Purpose: Executes full EDA pipeline for Bati Bank’s BNPL Credit Scoring Initiative

# ------------------------------------------------------------------------------
# 🛠 Environment Setup: Project Root and Path
# ------------------------------------------------------------------------------
import os
import sys
import warnings

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
    print("📂 Changed working directory to project root")

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✅ Added to sys.path: {project_root}")

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 📦 Imports: Data Loader and EDA Layers
# ------------------------------------------------------------------------------
import pandas as pd

from src.data_loader import CreditDataLoader
from src.eda.schema_auditor import SchemaAuditor
from src.eda.account_anomaly_detector import AccountAnomalyDetector
from src.eda.relationship_explorer import RelationshipExplorer
from src.eda.distribution_analyzer import DistributionAnalyzer
from src.eda.shared_subscription_checker import SharedSubscriptionChecker
from src.eda.temporal_behavior_analyzer import TemporalBehaviorAnalyzer
from src.eda.customer_behavior_profiler import CustomerBehaviorProfiler
from src.eda.fraud_segment_analyzer import FraudSegmentAnalyzer

# ------------------------------------------------------------------------------
# 📥 Load Raw Data Safely
# ------------------------------------------------------------------------------
data_path = "data/raw/data.csv"
loader = CreditDataLoader(filepath=data_path)

try:
    df = loader.load()
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 🧾 Layer 1: Schema Audit
# ------------------------------------------------------------------------------
auditor = SchemaAuditor(df)
schema_summary = auditor.report()
auditor.print_diagnostics()

# ------------------------------------------------------------------------------
# 🏢 Layer 2: Shared Account Anomaly Detection
# ------------------------------------------------------------------------------
anomaly_detector = AccountAnomalyDetector(df)
anomaly_detector.compute_account_statistics()
anomaly_detector.detect_shared_accounts()
tagged_df = anomaly_detector.tag_dataframe()
anomaly_detector.print_anomaly_summary()

# ------------------------------------------------------------------------------
# 🔍 Layer 3: Relationship Diagnostics with Shared Split
# ------------------------------------------------------------------------------
relationship_explorer = RelationshipExplorer(tagged_df)
relationship_explorer.plot_all_relationships(
    apply_cap=True, cap_percentile=0.95, hue="IsSharedAccount"
)
customer_summary = relationship_explorer.compute_summary_stats(apply_cap=True)
account_summary = relationship_explorer.compute_account_level_stats()

# ------------------------------------------------------------------------------
# 💰 Layer 4: Monetary Value Distributions with Shared Split
# ------------------------------------------------------------------------------
dist_viz = DistributionAnalyzer(tagged_df)

print("\n📊 Amount Distribution (Log)")
dist_viz.plot_numeric(
    column="Amount", log_scale=True, apply_cap=False, hue="IsSharedAccount"
)

print("\n📊 Value Distribution (Log)")
dist_viz.plot_numeric(
    column="Value", log_scale=True, apply_cap=False, hue="IsSharedAccount"
)

print("\n📊 Amount Distribution (Capped)")
dist_viz.plot_numeric(
    column="Amount",
    log_scale=False,
    apply_cap=True,
    cap_percentile=0.95,
    hue="IsSharedAccount",
)

print("\n📊 Value Distribution (Capped)")
dist_viz.plot_numeric(
    column="Value",
    log_scale=False,
    apply_cap=True,
    cap_percentile=0.95,
    hue="IsSharedAccount",
)

# ------------------------------------------------------------------------------
# 🏢 Layer 5: Shared Subscription Diagnostics
# ------------------------------------------------------------------------------
subscription_checker = SharedSubscriptionChecker(tagged_df)
subscription_checker.compute_subscription_sharing()
subscription_checker.print_top_shared_subscriptions(top_n=10)

# ------------------------------------------------------------------------------
# 🧮 Layer 6: Categorical Feature Exploration
# ------------------------------------------------------------------------------
print("\n🛍 Product Category")
dist_viz.plot_categorical("ProductCategory", top_k=10)

print("\n🌐 Channel ID")
dist_viz.plot_categorical("ChannelId", top_k=10)

print("\n🏢 Provider ID")
dist_viz.plot_categorical("ProviderId", top_k=10)

if "PricingStrategy" in tagged_df.columns:
    tagged_df["PricingStrategy"] = tagged_df["PricingStrategy"].astype("category")
    dist_viz.df["PricingStrategy"] = tagged_df["PricingStrategy"]
    print("\n⚙️ Pricing Strategy")
    dist_viz.plot_categorical("PricingStrategy", top_k=10)

print("\n💱 Currency Code")
dist_viz.plot_categorical("CurrencyCode", top_k=10)

# ------------------------------------------------------------------------------
# ⏰ Layer 7: Temporal Behavior Analysis
# ------------------------------------------------------------------------------
time_analyzer = TemporalBehaviorAnalyzer(tagged_df, datetime_col="TransactionStartTime")

print("\n🕒 Hourly Volume")
time_analyzer.plot_transactions_by_hour()

print("\n📆 Daily Volume")
time_analyzer.plot_transactions_by_dayofweek()

print("\n📈 Monthly Trend")
time_analyzer.plot_monthly_trend()

# ------------------------------------------------------------------------------
# 💳 Layer 8: RFM Behavioral Profiling with Shared Split
# ------------------------------------------------------------------------------
rfm_profiler = CustomerBehaviorProfiler(
    df=tagged_df,
    customer_id_col="CustomerId",
    date_col="TransactionStartTime",
    value_col="Value",
    shared_col="IsSharedAccount",
)
rfm_results = rfm_profiler.compute_rfm(snapshot_date="2024-01-01", split_by_shared=True)

print("\n🧩 Overall RFM")
print(rfm_results["Overall"].head())

print("\n🧩 Individual RFM")
print(rfm_results["Individual"].head())

print("\n🧩 Shared RFM")
print(rfm_results["Shared"].head())

# ------------------------------------------------------------------------------
# 🛡 Layer 9: Fraud Segment Analysis
# ------------------------------------------------------------------------------
fraud_analyzer = FraudSegmentAnalyzer(tagged_df, fraud_col="FraudResult")

for col in ["ProductCategory", "ChannelId", "ProviderId"]:
    print(f"\n🔍 Fraud Rate by {col}")
    fraud_analyzer.plot_fraud_rate_by_segment(segment_col=col)

print("✅ Task 2 EDA Completed Successfully.")
