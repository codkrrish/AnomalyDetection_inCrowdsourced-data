# src/decision.py


class DecisionEngine:

    def __init__(self, df):
        self.df = df.copy()

    def final_decision(self):
        """
        3-tier decision based on consistency_score, flag_duplicate, and flag_benchmark:
          - Approve: high consistency (>0.9)
          - Reject:  very low consistency (<0.5)
          - Review:  everything else (including duplicates, benchmarks, and small unscored segments)
        """
        decisions = []

        for _, row in self.df.iterrows():
            score = row["consistency_score"]
            duplicate = row.get("flag_duplicate", False)
            benchmark = row.get("flag_benchmark", False)
            lof_label = row.get("lof_label", "inlier")

            if score < 0.5:
                decisions.append("Reject")
            elif duplicate:
                decisions.append("Review")
            elif benchmark:
                decisions.append("Review")
            elif lof_label == "unscored":
                decisions.append("Review")
            elif score > 0.9:
                decisions.append("Approve")
            else:
                decisions.append("Review")

        self.df["decision"] = decisions
        return self.df

    def explain_flags(self):
        """
        Generate a human-readable explanation string per row
        summarising which flags were triggered.
        """
        self.df["explanation"] = self.df.apply(self._explain_row, axis=1)
        return self.df

    @staticmethod
    def _explain_row(row):
        reasons = []

        if row.get("flag_duplicate", False):
            reasons.append("Duplicate")

        if row.get("flag_benchmark", False):
            reasons.append("Benchmark anomaly")

        if row.get("lof_label") == "outlier":
            reasons.append("LOF/KNN anomaly")
            
        if row.get("lof_label") == "unscored":
            reasons.append("Small segment (marked for review)")

        if row.get("flag_tc_less_than_base", False):
            reasons.append("TC<Base")

        if row.get("flag_gibberish_company", False):
            reasons.append("Gibberish company name")

        if row.get("flag_negative_experience", False):
            reasons.append("Negative experience")

        if row.get("flag_yac_gt_yoe", False):
            reasons.append("YearsAtCompany > YearsOfExperience")

        if row.get("flag_yal_gt_yac", False):
            reasons.append("YearsAtLevel > YearsAtCompany")

        if row.get("flag_tc_arithmetic_mismatch", False):
            reasons.append("TC arithmetic mismatch")

        if row.get("flag_bonus_pct_mismatch", False):
            reasons.append("Bonus percentage mismatch")

        return "; ".join(reasons) if reasons else "Clean"
