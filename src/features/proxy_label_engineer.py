# ===========================================================================================
"""
proxy_label_engineer.py – Proxy Risk Label Engineering for Behavioral Credit Scoring (B5W5)
--------------------------------------------------------------------------------------------
Generates statistically defensible proxy risk labels for BNPL credit risk modeling using
transactional behavioral patterns, fraud signals, and customer segmentation.

Key functionalities include:
  • Statistical evaluation of behavioral differences between shared vs non-shared accounts
  • Per-customer behavioral consistency checks across shared vs non-shared usage
  • Automated fraud tagging using known fraud flags (FraudResult)
  • Behavioral segmentation using K-Means clustering on Recency, Frequency, Monetary (RFM)
  • Additive RFM point scoring for more granular behavioral risk
  • Weighted risk scoring for multi-tier risk category assignment (Low, Medium, High)
  • Generation of binary target labels (is_high_risk) for predictive modeling
  • Assignment of descriptive CustomerRiskTag for explainability
  • Optional visualization of risk category distributions

Author: Nabil Mohamed
Date: July 2025
"""
# ===========================================================================================

# 📦 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO)


# ===========================================================================================
# 📊 Class: StatisticalTester
# ===========================================================================================


class StatisticalTester:
    """Performs statistical tests to evaluate behavioral differences."""

    def __init__(self, confidence_level=0.85):
        self.confidence_level = confidence_level

    def test_normality(self, series):
        try:
            if series.dropna().shape[0] < 3:
                return np.nan
            _, p_value = shapiro(series.dropna())
            return p_value
        except Exception as e:
            logging.error(f"Normality test failed: {e}")
            return np.nan

    def compare_groups(self, data, group_col, value_col):
        try:
            group_true = data[data[group_col] == True][value_col].dropna()
            group_false = data[data[group_col] == False][value_col].dropna()

            if group_true.shape[0] < 5 or group_false.shape[0] < 5:
                return {"p_value": np.nan}

            p_true = self.test_normality(group_true)
            p_false = self.test_normality(group_false)

            if p_true > 0.05 and p_false > 0.05:
                _, p_val = ttest_ind(group_true, group_false, equal_var=False)
            else:
                _, p_val = mannwhitneyu(
                    group_true, group_false, alternative="two-sided"
                )

            return {"p_value": p_val}
        except Exception as e:
            logging.error(f"Group comparison failed: {e}")
            return {"p_value": np.nan}

    def evaluate_shared_vs_non_shared(self, df):
        try:
            results = {}
            for col in ["Recency", "Frequency"]:
                results[col] = self.compare_groups(df, "IsSharedAccount", col)

            significant = any(
                [
                    res["p_value"] < (1 - self.confidence_level)
                    for res in results.values()
                    if not np.isnan(res["p_value"])
                ]
            )

            return results, significant
        except Exception as e:
            logging.error(f"Shared vs non-shared evaluation failed: {e}")
            return {}, False

    def evaluate_individual_behavior_variance(self, df):
        try:
            flags = {}
            for cust in df["CustomerId"].unique():
                subset = df[df["CustomerId"] == cust]
                if subset["IsSharedAccount"].nunique() < 2:
                    flags[cust] = 0
                    continue
                group_sizes = subset.groupby("IsSharedAccount").size()
                if group_sizes.min() < 5:
                    flags[cust] = 0
                    continue

                flag = 0
                for col in ["Recency", "Frequency"]:
                    result = self.compare_groups(subset, "IsSharedAccount", col)
                    if result["p_value"] < (1 - self.confidence_level):
                        flag = 1

                flags[cust] = flag

            df["BehavioralVarianceFlag"] = (
                df["CustomerId"].map(flags).fillna(0).astype(int)
            )
            return df
        except Exception as e:
            logging.error(f"Behavioral variance evaluation failed: {e}")
            raise


# ===========================================================================================
# 🔐 Class: RiskCategorizer
# ===========================================================================================


class RiskCategorizer:
    """Generates risk categories and customer tags."""

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def apply_fraud_flag(self, df):
        try:
            df["FraudRisk"] = np.where(df["FraudResult"] == 1, 1, 0)
            return df
        except Exception as e:
            logging.error(f"Fraud flag failed: {e}")
            raise

    def apply_behavioral_segmentation(self, df):
        try:
            rfm = df[["Recency", "Frequency", "Monetary"]].fillna(0)
            scaled = self.scaler.fit_transform(rfm)
            df["BehaviorCluster"] = self.kmeans.fit_predict(scaled)
            return df
        except Exception as e:
            logging.error(f"Behavioral segmentation failed: {e}")
            raise

    def apply_behavioral_risk_flag(self, df):
        try:
            recency_cut = df["Recency"].quantile(0.80)
            frequency_cut = df["Frequency"].quantile(0.20)
            monetary_cut = df["Monetary"].quantile(0.20)

            df["BehavioralRiskFlag"] = (
                (
                    (df["Recency"] > recency_cut).astype(int)
                    + (df["Frequency"] < frequency_cut).astype(int)
                    + (df["Monetary"] < monetary_cut).astype(int)
                )
                == 3
            ).astype(int)

            return df
        except Exception as e:
            logging.error(f"Behavioral risk flag failed: {e}")
            raise

    def assign_risk_category(self, df):
        try:
            medians = df.groupby("BehaviorCluster")["Monetary"].median().sort_values()
            weights = {medians.index[0]: 70, medians.index[1]: 30, medians.index[2]: 0}

            recency_90 = df["Recency"].quantile(0.90)
            frequency_90 = df["Frequency"].quantile(0.90)
            monetary_10 = df["Monetary"].quantile(0.10)

            def calculate_score(row):
                score = 0
                if row["FraudRisk"] == 1:
                    score += 100
                if row.get("BehavioralVarianceFlag", 0) == 1:
                    score += 40
                if row.get("BehavioralRiskFlag", 0) == 1:
                    score += 30

                if row["Recency"] > recency_90:
                    score += 15
                if row["Frequency"] > frequency_90:
                    score += 15
                if row["Monetary"] < monetary_10:
                    score += 15

                score += weights.get(row["BehaviorCluster"], 30)

                return score

            df["RiskScore"] = df.apply(calculate_score, axis=1)
            df["risk_category"] = df["RiskScore"].apply(
                lambda x: 2 if x >= 70 else (1 if x >= 35 else 0)
            )
            return df
        except Exception as e:
            logging.error(f"Risk category assignment failed: {e}")
            raise

    def assign_customer_risk_tag(self, df):
        try:

            def tag(row):
                if row["FraudRisk"] == 1:
                    return "Fraud Flagged"
                elif row.get("BehavioralVarianceFlag", 0) == 1:
                    return "Behaviorally Volatile"
                elif row.get("BehavioralRiskFlag", 0) == 1:
                    return "Low RFM Risk"
                elif row["risk_category"] == 0:
                    return "Low Risk"
                else:
                    return "Standard Risk"

            df["CustomerRiskTag"] = df.apply(tag, axis=1)
            return df
        except Exception as e:
            logging.error(f"Customer risk tag failed: {e}")
            raise

    def plot_risk_distribution(self, df):
        try:
            sns.countplot(x="risk_category", data=df, palette="viridis")
            plt.title("Distribution of Risk Categories")
            plt.xlabel("Risk Category (0 = Low, 1 = Medium, 2 = High)")
            plt.ylabel("Number of Customers")
            plt.show()
        except Exception as e:
            logging.error(f"Plot failed: {e}")
            raise


# ===========================================================================================
# 🚦 Class: BinaryLabelGenerator
# ===========================================================================================


class BinaryLabelGenerator:
    """Generates binary high-risk label."""

    def __init__(self):
        pass

    def generate_binary_label(self, df):
        try:
            df["is_high_risk"] = np.where(df["risk_category"] == 2, 1, 0)
            return df
        except Exception as e:
            logging.error(f"Binary label generation failed: {e}")
            raise


# ===========================================================================================
# End of Module
# ===========================================================================================
