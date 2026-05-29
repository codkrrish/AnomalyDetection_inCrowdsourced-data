# src/feature_engineering.py

import numpy as np
import pandas as pd

from src.config import LEVEL_ORDINAL, COUNTRY_REGION, FAANG, TIER2


class FeatureEngineer:

    def __init__(self, df):
        self.df = df.copy()

    def create_features(self):
        """
        Full feature engineering matching the notebook:
          - level_ordinal, level_group
          - region (from countryId)
          - jobFamily_clean
          - company_tier
          - employmentType_clean
          - bonus_ratio, stock_ratio, tc_to_base
          - log_base_salary, log_tc
        """

        # ── Level ordinal + group ──
        level_key = (
            self.df.get("level", pd.Series("", index=self.df.index, dtype=str))
            .str.lower().str.strip().fillna("")
        )
        self.df["level_ordinal"] = level_key.map(LEVEL_ORDINAL).fillna(0).astype(int)

        def _level_group(o):
            if o <= 1:
                return "junior"
            if o <= 2:
                return "mid"
            if o <= 3:
                return "senior"
            return "staff_plus"

        self.df["level_group"] = self.df["level_ordinal"].apply(_level_group)

        # ── Region from countryId ──
        cid = pd.to_numeric(
            self.df.get("countryId", pd.Series(-1, index=self.df.index)),
            errors="coerce"
        ).fillna(-1).astype(int)
        self.df["region"] = cid.map(COUNTRY_REGION).fillna("Other")

        # ── Job family — cleaned ──
        self.df["jobFamily_clean"] = (
            self.df.get("jobFamily", pd.Series("Unknown", index=self.df.index, dtype=str))
            .str.strip().str.title().fillna("Unknown")
        )

        # ── Company tier ──
        def _tier(name):
            n = str(name).lower().strip()
            if n in FAANG:
                return "faang"
            if n in TIER2:
                return "tier2"
            return "other"

        self.df["company_tier"] = (
            self.df.get("company", pd.Series("other", index=self.df.index, dtype=str))
            .apply(_tier)
        )

        # ── Employment type — normalized ──
        self.df["employmentType_clean"] = (
            self.df.get("employmentType", pd.Series("unknown", index=self.df.index, dtype=str))
            .str.lower().str.strip().fillna("unknown")
        )

        # ── Derived ratio features ──
        base_safe = self.df["baseSalary_USD"].replace(0, np.nan)
        tc_safe = self.df["totalCompensation_USD"].replace(0, np.nan)

        self.df["bonus_ratio"] = (
            self.df["avgAnnualBonusValue_USD"] / base_safe
        ).fillna(0).clip(0, 5)

        self.df["stock_ratio"] = (
            self.df["avgAnnualStockGrantValue_USD"] / base_safe
        ).fillna(0).clip(0, 20)

        self.df["tc_to_base"] = (
            tc_safe / base_safe
        ).fillna(1).clip(0.5, 20)

        # ── Log transforms ──
        self.df["log_base_salary"] = np.log1p(self.df["baseSalary_USD"])
        self.df["log_tc"] = np.log1p(self.df["totalCompensation_USD"])

        return self.df