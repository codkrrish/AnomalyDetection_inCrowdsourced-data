# src/data_analysis.py

import numpy as np
import pandas as pd


class DatasetAnalyzer:

    def __init__(self, df):
        self.df = df

    def basic_info(self):
        print("\n========= DATASET INFO =========")
        print(self.df.info())

    def statistical_summary(self):
        print("\n========= NUMERICAL SUMMARY =========")
        print(self.df.describe(include=[np.number]))
        print("\n========= CATEGORICAL SUMMARY =========")
        print(self.df.describe(include=["object", "bool"]))

    def missing_values(self):
        missing = self.df.isnull().sum()
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Values": missing.values,
            "Missing Percentage": (missing.values / len(self.df)) * 100,
        })
        missing_df = missing_df.sort_values(
            by="Missing Percentage", ascending=False
        )
        print("\n========= MISSING VALUES =========")
        print(missing_df[missing_df["Missing Values"] > 0].to_string(index=False))

    def duplicate_statistics(self):
        for col in self.df.columns:
            sample = self.df[col].dropna()
            if len(sample):
                if isinstance(sample.iloc[0], list):
                    print(f"{col} contains LIST values")
                if isinstance(sample.iloc[0], dict):
                    print(f"{col} contains DICT values")

        print("\n========= DUPLICATES =========")

        temp_df = self.df.copy()
        for col in temp_df.columns:
            temp_df[col] = temp_df[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict)) else x
            )
        print("Duplicate Rows:", temp_df.duplicated().sum())

    def feature_counts(self):
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.df.select_dtypes(include=["object", "bool"]).columns.tolist()
        print(f"\nNumber of Numerical Features:   {len(numerical_features)}")
        print(f"Number of Categorical Features: {len(categorical_features)}")