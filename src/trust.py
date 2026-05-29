# src/trust.py


from src.config import TRUST_SCORE_BOOST

class TrustScorer:

    def __init__(self, df):
        self.df = df.copy()

    def compute_trust(self, initial_trust=1.0):
        """
        User trust score: starts at initial_trust, penalized for:
          - flag_duplicate:  −0.3
          - flag_benchmark:  −0.3
          - low consistency (<0.7): −0.2
        Clipped to [0, 1].
        """
        trust = []

        for _, row in self.df.iterrows():
            score = row["consistency_score"]
            duplicate = row.get("flag_duplicate", False)
            benchmark = row.get("flag_benchmark", False)

            trust_score = initial_trust

            if duplicate:
                trust_score -= 0.3

            if benchmark:
                trust_score -= 0.3

            if score < 0.7:
                trust_score -= 0.2

            trust.append(max(0, round(trust_score, 2)))

        self.df["user_trust"] = trust

        return self.df

    def apply_trust_boost(self):
        """
        Boost trust score by TRUST_SCORE_BOOST if the final decision is 'Approve'.
        Should be called after DecisionEngine.
        """
        if "decision" not in self.df.columns or "user_trust" not in self.df.columns:
            return self.df
            
        def _boost(row):
            t = row["user_trust"]
            if row["decision"] == "Approve":
                t = min(1.0, t + TRUST_SCORE_BOOST)
            return round(t, 2)
            
        self.df["user_trust"] = self.df.apply(_boost, axis=1)
        return self.df
