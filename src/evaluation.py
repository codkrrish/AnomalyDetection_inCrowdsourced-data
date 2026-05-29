# src/evaluation.py

import os
import json
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import CONSISTENCY_THRESHOLD


class Evaluator:

    def __init__(self, df):
        self.df = df

    def evaluate(self, target_column="is_anomaly"):
        """
        Evaluate predictions against ground-truth labels if available.
        Uses consistency_score < CONSISTENCY_THRESHOLD as the predicted
        anomaly indicator for metrics computation.
        """
        if target_column not in self.df.columns:
            print(
                f"'{target_column}' not found. "
                "Skipping supervised evaluation."
            )
            return None

        y_true = self.df[target_column].astype(int)

        # Predicted anomaly: consistency_score below threshold
        y_pred = (
            self.df["consistency_score"] < CONSISTENCY_THRESHOLD
        ).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        if "consistency_score" in self.df.columns:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_true,
                    1 - self.df["consistency_score"],
                )
            except Exception:
                metrics["roc_auc"] = None

        print("\n========== METRICS ==========")
        for k, v in metrics.items():
            if v is not None:
                print(f"{k}: {v:.4f}")

        print("\n========== CONFUSION MATRIX ==========")
        print(confusion_matrix(y_true, y_pred))

        print("\n========== CLASSIFICATION REPORT ==========")
        print(classification_report(y_true, y_pred, zero_division=0))

        return metrics

    def decision_summary(self):
        """Print distribution of Approve/Review/Reject decisions."""
        if "decision" not in self.df.columns:
            return

        print("\n========== DECISION DISTRIBUTION ==========")
        print(self.df["decision"].value_counts().to_string())

    def segment_summary(self):
        """Print per-segment aggregated statistics."""
        if "segment" not in self.df.columns:
            return None

        summary = (
            self.df.groupby("segment")
            .agg(
                count=("segment", "count"),
                lof_ran=("lof_score", lambda x: x.notna().sum()),
                lof_outliers=("lof_label", lambda x: (x == "outlier").sum()),
                mean_score=("consistency_score", "mean"),
                min_score=("consistency_score", "min"),
                rule_violations=("rule_violations", "sum"),
            )
            .sort_values("min_score")
            .reset_index()
        )

        print("\n========== SEGMENT SUMMARY (top 20) ==========")
        print(summary.head(20).to_string(index=False))

        return summary

    def save_metrics(self, metrics, filename="reports/statistics/metrics.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(metrics, f, indent=4)