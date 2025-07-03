"""
train.py – Comprehensive Model Training, Evaluation, and Tracking Pipeline (B5W5)
------------------------------------------------------------------------------

Trains, evaluates, visualizes, and tracks credit risk classification models for Bati Bank's
BNPL credit scoring initiative using structured, modular, and reproducible code.

Author: Nabil Mohamed
"""

# -------------------------------------------------------------------------
# 📦 Standard Imports
# -------------------------------------------------------------------------

import os  # File system operations
import warnings  # Suppress unnecessary warnings
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Basic plotting
import seaborn as sns  # Statistical plotting
import mlflow  # MLflow tracking
import mlflow.sklearn  # Sklearn integration

from sklearn.model_selection import train_test_split, GridSearchCV  # Splitting/tuning
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)  # Evaluation metrics

warnings.filterwarnings("ignore")  # Ignore global warnings
sns.set(style="whitegrid", context="notebook")  # Seaborn style

# -------------------------------------------------------------------------
# 📦 DataSplitter Class
# -------------------------------------------------------------------------


class DataSplitter:
    def __init__(self, df, target_column):
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        self.df = df.copy()  # Store copy safely
        self.target_column = target_column  # Store target name

    def split_and_balance(self, test_size=0.2, random_state=42, pos_neg_ratio=0.25):
        try:
            # Validate data types
            if not isinstance(self.df, pd.DataFrame):
                raise TypeError("Input data must be a pandas DataFrame.")

            # Separate features and target
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]

            # Split the data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )

            # Balance training set
            X_train_balanced, y_train_balanced = self._balance_classes(
                X_train, y_train, pos_neg_ratio
            )

            # Validate feature consistency
            missing_cols = set(X_train_balanced.columns).symmetric_difference(
                X_test.columns
            )
            if missing_cols:
                raise ValueError(
                    f"Train/Test feature mismatch detected: {missing_cols}"
                )

            # Print diagnostics
            print(
                f"✅ Training set: {X_train_balanced.shape}, Class distribution:\n{y_train_balanced.value_counts(normalize=True)}\n"
            )
            print(
                f"✅ Test set: {X_test.shape}, Class distribution:\n{y_test.value_counts(normalize=True)}\n"
            )

            return X_train_balanced, X_test, y_train_balanced, y_test

        except Exception as e:
            raise RuntimeError(f"Data splitting failed: {e}")

    def _balance_classes(self, X, y, ratio):
        try:
            # Validate inputs
            if not set(y.unique()).issubset({0, 1}):
                raise ValueError("Target must be binary (0 and 1).")

            pos = y[y == 1]
            neg = y[y == 0]
            n_pos = len(pos)
            n_neg_required = int(n_pos / ratio)

            if n_neg_required > len(neg):
                raise ValueError(
                    "Not enough negative samples to achieve desired ratio."
                )

            neg_sample = neg.sample(n=n_neg_required, random_state=42)
            pos_sample = pos

            balanced_idx = pd.concat([pos_sample, neg_sample]).index
            return X.loc[balanced_idx], y.loc[balanced_idx]

        except Exception as e:
            raise RuntimeError(f"Class balancing failed: {e}")


# -------------------------------------------------------------------------
# 📦 ModelTrainer Class
# -------------------------------------------------------------------------


class ModelTrainer:
    def __init__(self, X_train, y_train):
        if X_train.empty or y_train.empty:
            raise ValueError("Training data cannot be empty.")
        self.X_train = X_train
        self.y_train = y_train

    def train_logistic_regression(self):
        try:
            model = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )
            model.fit(self.X_train, self.y_train)
            print("✅ Logistic Regression trained.")
            return model
        except Exception as e:
            raise RuntimeError(f"Logistic Regression training failed: {e}")

    def train_random_forest(self, param_grid=None):
        try:
            rf = RandomForestClassifier(class_weight="balanced", random_state=42)
            if param_grid:
                grid = GridSearchCV(rf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
                grid.fit(self.X_train, self.y_train)
                print(
                    f"✅ Random Forest trained with Grid Search. Best params: {grid.best_params_}"
                )
                return grid.best_estimator_
            else:
                rf.fit(self.X_train, self.y_train)
                print("✅ Random Forest trained without tuning.")
                return rf
        except Exception as e:
            raise RuntimeError(f"Random Forest training failed: {e}")


# -------------------------------------------------------------------------
# 📦 ModelEvaluator Class
# -------------------------------------------------------------------------


class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        if not hasattr(model, "predict"):
            raise TypeError("Provided model must implement a predict method.")
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self, plot=False):
        try:
            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred),
                "recall": recall_score(self.y_test, y_pred),
                "f1": f1_score(self.y_test, y_pred),
                "roc_auc": roc_auc_score(self.y_test, y_proba),
            }

            print("✅ Model Evaluation Metrics:")
            for k, v in sorted(metrics.items()):
                print(f"{k:<10}: {v:.4f}")

            if plot:
                self._plot_confusion_matrix(y_pred)
                self._plot_roc_curve(y_proba)
                self._plot_precision_recall_curve(y_proba)

            return metrics

        except Exception as e:
            raise RuntimeError(f"Model evaluation failed: {e}")

    def _plot_confusion_matrix(self, y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def _plot_roc_curve(self, y_proba):
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc_score = roc_auc_score(self.y_test, y_proba)
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

    def _plot_precision_recall_curve(self, y_proba):
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        avg_precision = average_precision_score(self.y_test, y_proba)
        plt.plot(recall, precision, label=f"PR Curve (AP={avg_precision:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()


# -------------------------------------------------------------------------
# 📦 ExperimentTracker Class
# -------------------------------------------------------------------------


class ExperimentTracker:
    def __init__(self, experiment_name="BNPL-Credit-Risk"):
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError("Experiment name must be a non-empty string.")
        mlflow.set_experiment(experiment_name)

    def log_model(self, model, metrics, model_name="model"):
        try:
            with mlflow.start_run():
                mlflow.sklearn.log_model(model, model_name)
                mlflow.log_metrics(metrics)
                print("✅ Model and metrics logged to MLflow.")
        except Exception as e:
            raise RuntimeError(f"MLflow logging failed: {e}")

# -------------------------------------------------------------------------
# 📦 SHAPAnalyzer Class for Model Explainability
# -------------------------------------------------------------------------

import shap  # SHAP for explainability


class SHAPAnalyzer:
    def __init__(self, model, X_sample):
        self.model = model
        self.X_sample = X_sample

    def explain(self, plot_type="bar", max_display=10):
        try:
            if self.X_sample.empty:
                raise ValueError("Input sample cannot be empty for SHAP analysis.")

            # ✅ Force feature names explicitly
            explainer = shap.Explainer(
                self.model, self.X_sample, feature_names=self.X_sample.columns.tolist()
            )
            shap_values = explainer(self.X_sample)

            if plot_type == "bar":
                shap.plots.bar(shap_values, max_display=max_display)
            elif plot_type == "dot":
                shap.summary_plot(shap_values, self.X_sample, max_display=max_display)
            else:
                raise ValueError("Invalid plot_type. Use 'bar' or 'dot'.")

            print("✅ SHAP analysis completed.")
            return shap_values

        except Exception as e:
            raise RuntimeError(f"SHAP analysis failed: {e}")
