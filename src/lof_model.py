# src/lof_model.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import LOF_FEATURES, LOF_N_NEIGHBORS, LOF_CONTAMINATION, MIN_SEGMENT_SIZE, KNN_MIN_SEGMENT_SIZE


class SegmentLOF:

    def __init__(self, df):
        self.df = df.copy()
        self.models = {}
        self.scalers = {}

    def run_lof(self):
        """
        Assigns per-entry columns:
          segment       – human-readable segment label
          segment_size  – how many entries are in that segment
          lof_score     – raw LOF factor (≥1.0; higher = more anomalous)
          lof_label     – 'inlier' | 'outlier' | 'unscored' (segment too small)
        """
        self.df["segment"] = (
            self.df["jobFamily_clean"] + " | "
            + self.df["level_group"] + " | "
            + self.df["region"]
        )
        self.df["segment_size"] = 0
        self.df["lof_score"] = np.nan
        self.df["lof_label"] = "unscored"

        for seg_label, group in self.df.groupby("segment"):
            idx = group.index
            self.df.loc[idx, "segment_size"] = len(group)

            if len(group) < KNN_MIN_SEGMENT_SIZE:
                continue  # segment too small → stays 'unscored' (marked for review later)

            X = group[LOF_FEATURES].fillna(0).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if len(group) >= MIN_SEGMENT_SIZE:
                # LOF branch
                k = min(LOF_N_NEIGHBORS, len(group) - 1)

                lof = LocalOutlierFactor(
                    n_neighbors=k,
                    contamination=LOF_CONTAMINATION,
                    novelty=False,
                )
                labels = lof.fit_predict(X_scaled)        # -1 outlier, 1 inlier
                scores = -lof.negative_outlier_factor_     # higher = more anomalous

                self.df.loc[idx, "lof_score"] = scores
                self.df.loc[idx, "lof_label"] = np.where(
                    labels == -1, "outlier", "inlier"
                )
                
                self.models[seg_label] = ('lof', lof)
                self.scalers[seg_label] = scaler
                
            elif KNN_MIN_SEGMENT_SIZE <= len(group) < MIN_SEGMENT_SIZE:
                # KNN distance-based outlier branch
                k = min(5, len(group) - 1)
                
                knn = NearestNeighbors(n_neighbors=k)
                knn.fit(X_scaled)
                distances, _ = knn.kneighbors(X_scaled)
                
                # Use mean distance to k neighbors as outlier score
                scores = distances.mean(axis=1)
                
                # Threshold top 10% (LOF_CONTAMINATION) as outliers, or anything > 2 stdev
                threshold = np.percentile(scores, 100 * (1 - LOF_CONTAMINATION))
                # For very small segments, contamination might flag 1 out of 6.
                
                self.df.loc[idx, "lof_score"] = scores
                self.df.loc[idx, "lof_label"] = np.where(
                    scores >= threshold, "outlier", "inlier"
                )
                
                self.models[seg_label] = ('knn', knn)
                self.scalers[seg_label] = scaler
            else:
                # Handled by 'continue' above, but just in case
                pass

        return self.df

    def save_models(self):
        """Persist LOF models and scalers to disk."""
        os.makedirs("models/lof_models", exist_ok=True)
        os.makedirs("models/scalers", exist_ok=True)

        for key, model_tuple in self.models.items():
            filename = f"{key.replace(' | ', '__')}.pkl"
            joblib.dump(model_tuple, os.path.join("models/lof_models", filename))

        for key, scaler in self.scalers.items():
            filename = f"{key.replace(' | ', '__')}.pkl"
            joblib.dump(scaler, os.path.join("models/scalers", filename))

        print(f"[lof] Saved {len(self.models)} segment models.")