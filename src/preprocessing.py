# src/preprocessing.py

import numpy as np
import pandas as pd


class DataPreprocessor:

    def __init__(self, df):
        self.df = df.copy()

    def clean_text_columns(self):
        """Strip and lowercase all object-type columns."""
        text_cols = self.df.select_dtypes(include="object").columns

        for col in text_cols:
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

        return self.df

    def normalize_salaries(self):
        """
        Convert all salary fields to USD using the exchangeRate column.
        Produces: baseSalary_USD, totalCompensation_USD,
                  avgAnnualBonusValue_USD, avgAnnualStockGrantValue_USD
        """
        exch = (
            pd.to_numeric(
                self.df.get("exchangeRate", pd.Series(1, index=self.df.index)),
                errors="coerce"
            )
            .replace(0, np.nan)
            .fillna(1)
        )

        base_currency = (
            self.df.get(
                "baseSalaryCurrency",
                pd.Series("USD", index=self.df.index, dtype=str)
            ).fillna("USD")
        )

        bonus_currency = (
            self.df.get(
                "bonusCurrency",
                pd.Series("USD", index=self.df.index, dtype=str)
            ).fillna("USD")
        )

        base_raw = pd.to_numeric(
            self.df.get("baseSalary", pd.Series(0, index=self.df.index)),
            errors="coerce"
        ).fillna(0)

        bonus_raw = pd.to_numeric(
            self.df.get("avgAnnualBonusValue", pd.Series(0, index=self.df.index)),
            errors="coerce"
        ).fillna(0)

        # baseSalary → USD
        self.df["baseSalary_USD"] = np.where(
            base_currency == "USD", base_raw, base_raw / exch
        )

        # totalCompensation is stored in USD already in this dataset
        self.df["totalCompensation_USD"] = pd.to_numeric(
            self.df.get("totalCompensation", pd.Series(0, index=self.df.index)),
            errors="coerce"
        ).fillna(0)

        # avgAnnualBonusValue → USD
        self.df["avgAnnualBonusValue_USD"] = np.where(
            bonus_currency == "USD", bonus_raw, bonus_raw / exch
        )

        # Stock is always stored in USD per dataset schema
        self.df["avgAnnualStockGrantValue_USD"] = pd.to_numeric(
            self.df.get("avgAnnualStockGrantValue", pd.Series(0, index=self.df.index)),
            errors="coerce"
        ).fillna(0)

        return self.df

    def drop_columns(self, columns_to_drop):
        """Drop unnecessary columns if they exist."""
        existing = [c for c in columns_to_drop if c in self.df.columns]
        if existing:
            self.df = self.df.drop(columns=existing)
            print(f"[preprocessing] Dropped columns: {existing}")
        return self.df