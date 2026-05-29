# src/scoring.py

import numpy as np
import pandas as pd

from src.config import RULE_COLUMNS, WEIGHT_RULE, WEIGHT_LOF


class ConsistencyScorer:

    def __init__(self, df, flags):
        self.df = df.copy()
        self.flags = flags

    def compute_score(self):
        """
        Weighted consistency score:
          rule_score      = 1 - (violations / total_rules)
          lof_normalized  = 1 - ((lof_score.clip(1,3) - 1) / 2)   [1→1, 3→0]
          consistency     = WEIGHT_RULE * rule_score + WEIGHT_LOF * lof_normalized
        Falls back to rule-only for entries where LOF was not scored.
        """
        # Merge flag columns into df
        for col in RULE_COLUMNS:
            if col in self.flags.columns:
                self.df[col] = self.flags[col].astype(bool).values

        self.df["rule_violations"] = (
            self.flags[RULE_COLUMNS].astype(int).sum(axis=1).values
        )
        self.df["rule_score"] = 1.0 - (
            self.df["rule_violations"] / len(RULE_COLUMNS)
        )

        # Normalize LOF: clip raw score to [1, 3], map to [1, 0]
        lof_raw = self.df["lof_score"].copy()
        self.df["lof_normalized"] = 1.0 - (
            (lof_raw.clip(1.0, 3.0) - 1.0) / 2.0
        )

        has_lof = self.df["lof_score"].notna()

        self.df["consistency_score"] = np.where(
            has_lof,
            WEIGHT_RULE * self.df["rule_score"] + WEIGHT_LOF * self.df["lof_normalized"],
            self.df["rule_score"],  # fallback to rule-only for small segments
        ).clip(0.0, 1.0).round(4)

        return self.df