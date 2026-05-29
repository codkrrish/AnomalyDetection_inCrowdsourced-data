# src/visualization.py

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PLOT_DIR, CONSISTENCY_THRESHOLD, MIN_SEGMENT_SIZE

sns.set_style("whitegrid")


class Visualizer:

    def __init__(self, df, output_dir=None):
        self.df = df
        self.output_dir = output_dir or PLOT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    # ──────────────────────────────────────────
    # Basic distribution plots
    # ──────────────────────────────────────────

    def anomaly_distribution(self):
        """Bar chart of final decision distribution."""
        if "decision" not in self.df.columns:
            return

        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.df, x="decision",
                      order=["Approve", "Review", "Reject"],
                      palette="viridis")
        plt.title("Decision Distribution")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/decision_distribution.png", dpi=150)
        plt.close()

    def consistency_distribution(self):
        if "consistency_score" not in self.df.columns:
            return

        plt.figure(figsize=(10, 6))
        sns.histplot(self.df["consistency_score"], bins=30, kde=True, color="steelblue")
        plt.axvline(CONSISTENCY_THRESHOLD, color="red", linestyle="--",
                     label=f"Threshold = {CONSISTENCY_THRESHOLD}")
        plt.title("Consistency Score Distribution")
        plt.xlabel("Consistency Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/consistency_distribution.png", dpi=150)
        plt.close()

    def salary_vs_experience(self):
        required = ["yearsOfExperience", "baseSalary_USD", "decision"]
        if not all(col in self.df.columns for col in required):
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.df,
            x="yearsOfExperience",
            y="baseSalary_USD",
            hue="decision",
            alpha=0.6,
        )
        plt.title("Salary vs Experience")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/salary_vs_experience.png", dpi=150)
        plt.close()

    def lof_score_distribution(self):
        if "lof_score" not in self.df.columns:
            return

        plt.figure(figsize=(10, 6))
        scored = self.df["lof_score"].dropna()
        sns.histplot(scored, bins=30, kde=True, color="coral")
        plt.title("LOF Score Distribution")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/lof_score_distribution.png", dpi=150)
        plt.close()

    def trust_score_analysis(self):
        if "user_trust" not in self.df.columns:
            return

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x="decision", y="user_trust",
                    order=["Approve", "Review", "Reject"],
                    palette="Set2")
        plt.title("User Trust Score by Decision")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/trust_score_analysis.png", dpi=150)
        plt.close()



    # ──────────────────────────────────────────
    # LOF segment scatter (from notebook Cell 32)
    # ──────────────────────────────────────────

    def lof_segment_scatter(self, segment_name=None, highlight_row=None):
        """
        Scatter plot for a specific jobFamily | level_group segment.
        If highlight_row is provided, it plots it as a bright square marker.
        """
        if "jobFamily_clean" not in self.df.columns or "level_group" not in self.df.columns:
            return

        # Create region-less segment definition
        self.df["jf_level"] = self.df["jobFamily_clean"] + " | " + self.df["level_group"]

        if segment_name is None:
            seg_stats = self.df.groupby("jf_level")["consistency_score"].mean()
            if seg_stats.empty:
                return
            segment_name = seg_stats.idxmin()

        df_seg = self.df[self.df["jf_level"] == segment_name].copy()
        
        # Only generate plot if eligible for LOF
        if len(df_seg) < MIN_SEGMENT_SIZE:
            return

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df_seg,
            x="yearsOfExperience",
            y="log_base_salary",
            hue="lof_label",
            style="lof_label",
            s=100, alpha=0.7,
            palette={"inlier": "blue", "outlier": "red", "unscored": "gray"},
        )

        # Highlight new entry if provided
        if highlight_row is not None:
            plt.scatter(
                highlight_row["yearsOfExperience"],
                highlight_row["log_base_salary"],
                c="gold", marker="s", s=250, edgecolors="black", linewidths=1.5,
                label="New Entry", zorder=5
            )

        # Annotate outliers
        outliers = df_seg[df_seg["lof_label"] == "outlier"]
        for _, row in outliers.iterrows():
            plt.text(
                row["yearsOfExperience"], row["log_base_salary"],
                f"LOF: {row.get('lof_score', 0):.2f}",
                fontsize=9, color="darkred", ha="right", va="bottom",
            )

        plt.title(f"LOF Outlier Visualization — {segment_name}")
        plt.xlabel("Years of Experience")
        plt.ylabel("Log(Base Salary USD)")
        plt.grid(True)
        plt.legend(title="LOF Label")
        plt.tight_layout()

        safe_name = segment_name.replace(" | ", "_").replace(" ", "_")
        plt.savefig(f"{self.output_dir}/lof_segment_{safe_name}.png", dpi=150)
        plt.close()

    # ──────────────────────────────────────────
    # LOF by job family (from notebook Cell 34)
    # ──────────────────────────────────────────

    def lof_by_job_family(self, max_families=5):
        """
        For each unique job family, generate LOF scatter plots
        for each region-less segment (jobFamily | level_group).
        """
        if "jobFamily_clean" not in self.df.columns or "level_group" not in self.df.columns:
            return

        self.df["jf_level"] = self.df["jobFamily_clean"] + " | " + self.df["level_group"]
        families = self.df["jobFamily_clean"].unique()[:max_families]

        for job_family in families:
            jf_df = self.df[self.df["jobFamily_clean"] == job_family]
            segments = sorted(jf_df["jf_level"].unique())

            for seg_name in segments:
                seg_df = jf_df[jf_df["jf_level"] == seg_name]
                
                # Only plot if eligible for LOF
                if len(seg_df) < MIN_SEGMENT_SIZE:
                    continue

                plt.figure(figsize=(12, 8))
                sns.scatterplot(
                    data=seg_df,
                    x="yearsOfExperience",
                    y="log_base_salary",
                    hue="lof_label",
                    style="lof_label",
                    s=100, alpha=0.7,
                    palette={"inlier": "blue", "outlier": "red", "unscored": "gray"},
                )

                outliers = seg_df[seg_df["lof_label"] == "outlier"]
                for _, row in outliers.iterrows():
                    plt.text(
                        row["yearsOfExperience"], row["log_base_salary"],
                        f"LOF: {row.get('lof_score', 0):.2f}",
                        fontsize=9, color="darkred", ha="right", va="bottom",
                    )

                plt.title(f"LOF Outliers — {seg_name}")
                plt.xlabel("Years of Experience")
                plt.ylabel("Log(Base Salary USD)")
                plt.grid(True)
                plt.legend(title="LOF Label")
                plt.tight_layout()

                safe = seg_name.replace(" | ", "_").replace(" ", "_")
                plt.savefig(f"{self.output_dir}/lof_jf_{safe}.png", dpi=150)
                plt.close()

    # ──────────────────────────────────────────
    # Generate all
    # ──────────────────────────────────────────

    def generate_all(self):
        print("\nGenerating visualizations...")
        self.anomaly_distribution()
        self.consistency_distribution()
        self.salary_vs_experience()
        self.lof_score_distribution()
        self.trust_score_analysis()
        self.lof_segment_scatter()
        self.lof_by_job_family()
        print(f"Saved plots to {self.output_dir}")

    def inference_notes(self):
        print("""
================ INFERENCE GUIDE ================

1. Decision Distribution
   Large Reject percentage may indicate
   poor data quality or rule thresholds.

2. Consistency Score Distribution
   Scores near 1.0 indicate reliable records.
   Scores near 0 indicate suspicious entries.

3. Salary vs Experience
   Extreme salaries at low experience
   often become anomalies.

4. LOF Distribution
   Higher LOF scores indicate stronger
   local outliers within each segment.

5. Trust Score Analysis
   If trust scores correlate with
   Approve decisions, trust mechanism works.

6. Region-wise Decisions
   Regions with many Reject/Review entries
   may require custom thresholds.

=================================================
        """)