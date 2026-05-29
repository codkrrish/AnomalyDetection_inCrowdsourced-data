# src/rule_engine.py

import re
import numpy as np
import pandas as pd

from src.config import PUBLIC_COMPANIES, STARTUP_STAGES, COMPANY_SIZE_MAP


def _is_gibberish(name) -> bool:
    """Company name is numeric-only, symbol-only, or suspiciously short."""
    if not isinstance(name, str) or len(name.strip()) < 2:
        return True
    return bool(re.fullmatch(r"[\d\W]+", name.strip()))


class RuleEngine:

    def __init__(self, df):
        self.df = df.copy()

    def apply_rules(self):
        """
        Returns a boolean DataFrame where True = that rule was violated.

        Columns produced:
          flag_gibberish_company       – company name is numeric/garbage
          flag_yal_gt_yac              – yearsAtLevel > yearsAtCompany
          flag_yac_gt_yoe              – yearsAtCompany > yearsOfExperience
          flag_negative_experience     – any experience field is negative
          flag_zero_or_negative_base   – baseSalary <= 0
          flag_tc_less_than_base       – placeholder (filled after normalization)
          flag_tc_arithmetic_mismatch  – placeholder (filled after normalization)
          flag_junior_level_high_yoe   – L1/IC1 with yearsOfExperience > 8
          flag_bonus_pct_mismatch      – declared bonus% ≠ actual bonus (>25% deviation)
          flag_public_co_startup_stage – known public company tagged as startup stage
          flag_companysize_mismatch    – public company listed as tiny (<1000 employees)
        """
        flags = pd.DataFrame(index=self.df.index)

        # — Company name quality
        flags["flag_gibberish_company"] = self.df["company"].apply(_is_gibberish)

        # — Experience / tenure ordering
        yoe = pd.to_numeric(
            self.df.get("yearsOfExperience", pd.Series(0, index=self.df.index)),
            errors="coerce"
        ).fillna(0)

        yac = pd.to_numeric(
            self.df.get("yearsAtCompany", pd.Series(0, index=self.df.index)),
            errors="coerce"
        ).fillna(0)

        yal = pd.to_numeric(
            self.df.get("yearsAtLevel", pd.Series(0, index=self.df.index)),
            errors="coerce"
        ).fillna(0)

        flags["flag_yal_gt_yac"] = yal > yac
        flags["flag_yac_gt_yoe"] = yac > yoe
        flags["flag_negative_experience"] = (yoe < 0) | (yac < 0) | (yal < 0)

        # — Salary sanity (pre-normalization)
        flags["flag_zero_or_negative_base"] = pd.to_numeric(
            self.df.get("baseSalary", pd.Series(0, index=self.df.index)),
            errors="coerce"
        ).fillna(0) <= 0

        # — Placeholders filled after normalization
        flags["flag_tc_less_than_base"] = False
        flags["flag_tc_arithmetic_mismatch"] = False

        # — Junior level with high experience
        level_norm = (
            self.df.get("level", pd.Series("", index=self.df.index, dtype=str))
            .str.lower().str.strip().fillna("")
        )
        is_entry = level_norm.isin(["l1", "ic1", "e1", "junior", "swe i"])
        flags["flag_junior_level_high_yoe"] = is_entry & (yoe > 8)

        # — Bonus percentage vs actual bonus value (>25% deviation)
        base_raw = pd.to_numeric(
            self.df.get("baseSalary", pd.Series(0, index=self.df.index)),
            errors="coerce"
        ).fillna(0)

        bonus_pct = pd.to_numeric(
            self.df.get("annualTargetBonusPercentage", pd.Series(0, index=self.df.index)),
            errors="coerce"
        ).fillna(0) / 100

        bonus_actual = pd.to_numeric(
            self.df.get("annualTargetBonusValue", pd.Series(0, index=self.df.index)),
            errors="coerce"
        ).fillna(0)

        expected_bonus = base_raw * bonus_pct
        both_nonzero = (expected_bonus > 0) & (bonus_actual > 0)
        ratio = (bonus_actual / expected_bonus.replace(0, np.nan)).fillna(1.0)
        flags["flag_bonus_pct_mismatch"] = both_nonzero & ((ratio < 0.75) | (ratio > 1.25))

        # — Known public company with startup funding stage
        company_lc = (
            self.df.get("company", pd.Series("", index=self.df.index, dtype=str))
            .str.lower().str.strip().fillna("")
        )
        funding_lc = (
            self.df.get("fundingStage", pd.Series("", index=self.df.index, dtype=str))
            .str.lower().str.strip().fillna("")
        )
        flags["flag_public_co_startup_stage"] = (
            company_lc.isin(PUBLIC_COMPANIES) & funding_lc.isin(STARTUP_STAGES)
        )

        # — Large public company listed with tiny headcount
        size_numeric = (
            self.df.get("companySize", pd.Series(np.nan, index=self.df.index, dtype=str))
            .map(COMPANY_SIZE_MAP)
            .fillna(np.nan)
        )
        flags["flag_companysize_mismatch"] = (
            company_lc.isin(PUBLIC_COMPANIES) & (size_numeric < 1000)
        )

        return flags

    @staticmethod
    def fill_postnorm_flags(df, flags):
        """
        Fill flag placeholders that require USD-normalized columns.
        Must be called after normalize_salaries().
        """
        # TC < base salary
        flags["flag_tc_less_than_base"] = (
            (df["totalCompensation_USD"] < df["baseSalary_USD"])
            & (df["baseSalary_USD"] > 0)
        )

        # TC arithmetic: |TC - (base + bonus + stock)| / TC > 15%
        reconstructed = (
            df["baseSalary_USD"]
            + df["avgAnnualBonusValue_USD"]
            + df["avgAnnualStockGrantValue_USD"]
        )
        tc_safe = df["totalCompensation_USD"].replace(0, np.nan)
        deviation = ((reconstructed - tc_safe).abs() / tc_safe).fillna(0)
        flags["flag_tc_arithmetic_mismatch"] = deviation > 0.15

        return flags