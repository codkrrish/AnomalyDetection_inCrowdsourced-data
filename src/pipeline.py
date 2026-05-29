# src/pipeline.py

import os
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    TEST_SIZE, RANDOM_STATE, DROP_COLUMNS,
    OUTPUT_DIR, CONSISTENCY_THRESHOLD,
)
from src.data_analysis import DatasetAnalyzer
from src.preprocessing import DataPreprocessor
from src.rule_engine import RuleEngine
from src.feature_engineering import FeatureEngineer
from src.duplicate_detection import DuplicateDetector
from src.benchmark import BenchmarkValidator
from src.lof_model import SegmentLOF
from src.scoring import ConsistencyScorer
from src.trust import TrustScorer
from src.decision import DecisionEngine
from src.evaluation import Evaluator
from src.visualization import Visualizer


def process_dataframe(df, split_name="train"):
    """
    Full anomaly pipeline for a single dataframe (train or test split).

    Order matches the notebook:
      1. flag_rules (pre-normalization)
      2. normalize_salaries
      3. fill_postnorm_flags
      4. engineer_features
      5. detect_duplicates
      6. benchmark_validation
      7. run_lof
      8. compute_scores
      9. user_trust_score
     10. final_decision
     11. explain_flags
    """

    print(f"\n{'='*60}")
    print(f"PROCESSING {split_name.upper()} DATA ({len(df)} records)")
    print(f"{'='*60}")

    # ── 1. Rule-based flags (pre-normalization) ──
    print("\n[1/11] Applying rule-based flags...")
    rules = RuleEngine(df)
    flags = rules.apply_rules()

    # ── 2. Salary normalization ──
    print("[2/11] Normalizing salaries to USD...")
    prep = DataPreprocessor(df)
    df = prep.normalize_salaries()

    # ── 3. Fill post-normalization flags ──
    print("[3/11] Filling post-normalization flags...")
    flags = RuleEngine.fill_postnorm_flags(df, flags)

    # ── 4. Feature engineering ──
    print("[4/11] Engineering features...")
    fe = FeatureEngineer(df)
    df = fe.create_features()

    # ── 5. Duplicate detection ──
    print("[5/11] Detecting duplicates...")
    dup = DuplicateDetector(df)
    df = dup.detect_duplicates()

    # ── 6. Benchmark validation ──
    print("[6/11] Validating against benchmarks...")
    bench = BenchmarkValidator(df)
    df = bench.validate()

    # ── 7. LOF detection ──
    print("[7/11] Running segmented LOF...")
    lof = SegmentLOF(df)
    df = lof.run_lof()

    if split_name == "train":
        lof.save_models()

    # ── 8. Consistency scoring ──
    print("[8/11] Computing consistency scores...")
    scorer = ConsistencyScorer(df, flags)
    df = scorer.compute_score()

    # ── 9. User trust score ──
    print("[9/11] Computing user trust scores...")
    trust = TrustScorer(df)
    df = trust.compute_trust()

    # ── 10. Final decision ──
    print("[10/11] Making final decisions...")
    dec = DecisionEngine(df)
    df = dec.final_decision()

    # ── 11. Explain flags ──
    print("[11/11] Generating explanations...")
    dec2 = DecisionEngine(df)
    df = dec2.explain_flags()

    # ── 12. Apply trust boost ──
    print("[12/12] Applying trust score boost...")
    trust2 = TrustScorer(df)
    df = trust2.apply_trust_boost()

    return df


def run_pipeline(df):
    """
    Full end-to-end pipeline:
      - Analyse raw data
      - Separate labels
      - Drop unnecessary columns
      - Train/test split
      - Process each split
      - Evaluate
      - Save outputs
      - Generate visualizations
    """

    print("\n" + "=" * 60)
    print("STARTING PIPELINE")
    print("=" * 60)

    # ══════════════════════════════════════════
    # DATA ANALYSIS
    # ══════════════════════════════════════════

    analyzer = DatasetAnalyzer(df)
    analyzer.basic_info()
    analyzer.statistical_summary()
    analyzer.missing_values()
    analyzer.duplicate_statistics()
    analyzer.feature_counts()

    # ══════════════════════════════════════════
    # SEPARATE LABELS
    # ══════════════════════════════════════════

    has_labels = "is_anomaly" in df.columns

    if has_labels:
        y = df["is_anomaly"].astype(int)
        print(f"\nLabel distribution:\n{y.value_counts().to_string()}")
        print(f"Percentage:\n{(y.value_counts(normalize=True) * 100).round(2).to_string()}")

    # ══════════════════════════════════════════
    # DROP UNNECESSARY COLUMNS
    # ══════════════════════════════════════════

    existing_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)
        print(f"\nDropped columns: {existing_drop}")

    # ══════════════════════════════════════════
    # TRAIN / TEST SPLIT
    # ══════════════════════════════════════════

    print("\nCreating Train/Test Split...")

    if has_labels:
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=df["is_anomaly"],
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    print(f"Train Size : {len(train_df)}")
    print(f"Test Size  : {len(test_df)}")

    # ══════════════════════════════════════════
    # PROCESS TRAIN
    # ══════════════════════════════════════════

    train_processed = process_dataframe(train_df.copy(), split_name="train")

    # ══════════════════════════════════════════
    # PROCESS TEST
    # ══════════════════════════════════════════

    test_processed = process_dataframe(test_df.copy(), split_name="test")

    # ══════════════════════════════════════════
    # EVALUATION
    # ══════════════════════════════════════════

    if has_labels:
        print("\n" + "=" * 60)
        print("TRAIN EVALUATION")
        print("=" * 60)
        eval_train = Evaluator(train_processed)
        train_metrics = eval_train.evaluate()
        eval_train.decision_summary()
        eval_train.segment_summary()

        if train_metrics:
            eval_train.save_metrics(train_metrics, "reports/statistics/train_metrics.json")

        print("\n" + "=" * 60)
        print("TEST EVALUATION")
        print("=" * 60)
        eval_test = Evaluator(test_processed)
        test_metrics = eval_test.evaluate()
        eval_test.decision_summary()
        eval_test.segment_summary()

        if test_metrics:
            eval_test.save_metrics(test_metrics, "reports/statistics/test_metrics.json")

    # ══════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════

    for name, processed in [("TRAIN", train_processed), ("TEST", test_processed)]:
        df_flagged = processed[processed["rule_violations"] > 0]
        df_lof_outliers = processed[processed["lof_label"] == "outlier"]
        df_low = processed[processed["consistency_score"] < CONSISTENCY_THRESHOLD]

        print(f"\n{'='*54}")
        print(f"  {name} CONSISTENCY VALIDATION SUMMARY")
        print(f"{'='*54}")
        print(f"  Total entries      : {len(processed)}")
        print(f"  Rule violations    : {len(df_flagged)}")
        print(f"  LOF outliers       : {len(df_lof_outliers)}")
        print(f"  Low consistency    : {len(df_low)}")

    # ══════════════════════════════════════════
    # SAVE OUTPUTS
    # ══════════════════════════════════════════

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_processed.to_csv(f"{OUTPUT_DIR}/train_processed.csv", index=False)
    test_processed.to_csv(f"{OUTPUT_DIR}/test_processed.csv", index=False)

    # Combined output
    combined = pd.concat([train_processed, test_processed], ignore_index=True)
    combined.to_csv(f"{OUTPUT_DIR}/validated_output.csv", index=False)

    # Separate anomalies / clean for each split
    train_processed[train_processed["decision"] == "Reject"].to_csv(
        f"{OUTPUT_DIR}/train_rejected.csv", index=False
    )
    test_processed[test_processed["decision"] == "Reject"].to_csv(
        f"{OUTPUT_DIR}/test_rejected.csv", index=False
    )

    print(f"\nOutput files saved to {OUTPUT_DIR}")

    # ══════════════════════════════════════════
    # VISUALIZATION
    # ══════════════════════════════════════════

    print("\nGenerating plots...")
    visualizer = Visualizer(train_processed)
    visualizer.generate_all()
    visualizer.inference_notes()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return {
        "train": train_processed,
        "test": test_processed,
    }