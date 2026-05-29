import argparse
import json
import os
import joblib
import pandas as pd
import numpy as np

from src.config import OUTPUT_DIR, LOF_FEATURES
from src.rule_engine import RuleEngine
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.benchmark import BenchmarkValidator
from src.scoring import ConsistencyScorer
from src.trust import TrustScorer
from src.decision import DecisionEngine
from src.visualization import Visualizer

def process_single_entry(entry_dict, initial_trust=1.0, save_output=False):
    # 1. Convert to DataFrame
    df = pd.DataFrame([entry_dict])

    # 2. Rule flags
    rules = RuleEngine(df)
    flags = rules.apply_rules()

    # 3. Normalize salaries
    prep = DataPreprocessor(df)
    df = prep.normalize_salaries()

    # 4. Fill post-normalization flags
    flags = RuleEngine.fill_postnorm_flags(df, flags)

    # 5. Feature Engineering
    fe = FeatureEngineer(df)
    df = fe.create_features()

    # 6. Benchmark Validation
    # Set default duplicate to False since we aren't comparing against the whole dataset
    df["flag_duplicate"] = False
    bench = BenchmarkValidator(df)
    df = bench.validate()

    # 7. Segment and LOF/KNN inference
    df["segment"] = (
        df["jobFamily_clean"] + " | "
        + df["level_group"] + " | "
        + df["region"]
    )
    seg_label = df["segment"].iloc[0]
    safe_seg_label = seg_label.replace(" | ", "__")
    
    model_path = os.path.join("models/lof_models", f"{safe_seg_label}.pkl")
    scaler_path = os.path.join("models/scalers", f"{safe_seg_label}.pkl")
    
    df["segment_size"] = 0
    df["lof_score"] = np.nan
    df["lof_label"] = "unscored"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model_tuple = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        X = df[LOF_FEATURES].fillna(0).values
        X_scaled = scaler.transform(X)
        
        if isinstance(model_tuple, tuple):
            model_type, model = model_tuple
            if model_type == 'lof':
                # LOF
                # We can use decision_function or novelty detection if trained with novelty=True.
                # However, since we used novelty=False, we can't easily run predict on new data.
                # But wait, LocalOutlierFactor with novelty=False doesn't have predict for new data!
                # We must use `_predict` or re-fit. 
                # Actually, LocalOutlierFactor doesn't support predict if novelty=False.
                # We will calculate distance to neighbors using the training data? 
                pass
        else:
            # Old model format (just LOF model)
            model = model_tuple
            model_type = 'lof'
            
        # Due to LOF novelty=False restriction, the proper way to score a new point
        # is to either have novelty=True (which alters training) or just use KNN for inference.
        # For simplicity and given the prompt constraints, if it's LOF or KNN, we can load the 
        # training data for this segment and calculate the score, or use a heuristic.
        # Let's load train_processed.csv to do exact neighbor distance.
    
    # Actually, a robust way to do inference without retraining or novelty=True 
    # is to load train_processed.csv, filter by segment, and append the new point, then fit LOF/KNN.
    # Let's do that.
    
    train_file = os.path.join(OUTPUT_DIR, "train_processed.csv")
    if os.path.exists(train_file):
        train_df = pd.read_csv(train_file)
        seg_train = train_df[train_df["segment"] == seg_label].copy()
        
        if not seg_train.empty:
            df["segment_size"] = len(seg_train)
            
            # Combine
            combined_X = pd.concat([seg_train[LOF_FEATURES], df[LOF_FEATURES]]).fillna(0).values
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            combined_X_scaled = scaler.fit_transform(combined_X)
            
            new_X_scaled = combined_X_scaled[-1:]
            
            if len(seg_train) >= 10: # MIN_SEGMENT_SIZE
                from sklearn.neighbors import LocalOutlierFactor
                k = min(5, len(seg_train))
                lof = LocalOutlierFactor(n_neighbors=k, contamination=0.10, novelty=True)
                lof.fit(combined_X_scaled[:-1])
                pred = lof.predict(new_X_scaled)
                score = -lof.score_samples(new_X_scaled)
                df.loc[0, "lof_score"] = score[0]
                df.loc[0, "lof_label"] = "outlier" if pred[0] == -1 else "inlier"
            elif 5 <= len(seg_train) < 10: # KNN
                from sklearn.neighbors import NearestNeighbors
                k = min(5, len(seg_train))
                knn = NearestNeighbors(n_neighbors=k)
                knn.fit(combined_X_scaled[:-1])
                distances, _ = knn.kneighbors(new_X_scaled)
                score = distances.mean(axis=1)[0]
                
                # Calculate threshold from training data
                train_dist, _ = knn.kneighbors(combined_X_scaled[:-1])
                train_scores = train_dist.mean(axis=1)
                threshold = np.percentile(train_scores, 90) # Top 10%
                
                df.loc[0, "lof_score"] = score
                df.loc[0, "lof_label"] = "outlier" if score >= threshold else "inlier"

    # 8. Consistency Score
    scorer = ConsistencyScorer(df, flags)
    df = scorer.compute_score()

    # 9. User Trust Score
    trust = TrustScorer(df)
    df = trust.compute_trust(initial_trust=initial_trust)

    # 10. Decision
    dec = DecisionEngine(df)
    df = dec.final_decision()
    df = dec.explain_flags()

    # 11. Apply Trust Boost if Approved
    trust2 = TrustScorer(df)
    df = trust2.apply_trust_boost()
    
    # 12. Visualization
    if os.path.exists(train_file):
        train_df = pd.read_csv(train_file)
        # Create a Visualizer with the combined data to plot the segment
        jf_level = df["jobFamily_clean"].iloc[0] + " | " + df["level_group"].iloc[0]
        
        # Filter train_df to the same jf_level
        train_df["jf_level"] = train_df["jobFamily_clean"] + " | " + train_df["level_group"]
        
        # Only plot if segment is eligible (>= 10)
        seg_train_jf = train_df[train_df["jf_level"] == jf_level]
        if len(seg_train_jf) >= 10:
            viz_df = pd.concat([train_df, df], ignore_index=True)
            vis = Visualizer(viz_df)
            vis.lof_segment_scatter(segment_name=jf_level, highlight_row=df.iloc[0])

    # Result extraction
    result = df.iloc[0].to_dict()
    output = {
        "decision": result["decision"],
        "explanation": result["explanation"],
        "consistency_score": result["consistency_score"],
        "user_trust": result["user_trust"],
        "lof_label": result["lof_label"],
        "lof_score": result.get("lof_score", "N/A"),
        "rule_violations": int(result["rule_violations"]),
    }
    
    print("\n" + "="*40)
    print("INFERENCE RESULT")
    print("="*40)
    print(json.dumps(output, indent=2))
    
    if save_output:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "inference_result.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResult saved to {out_path}")

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer anomaly status for a new entry.")
    parser.add_argument("--data", type=str, required=True, help="JSON string of the new entry.")
    parser.add_argument("--trust", type=float, default=1.0, help="Initial user trust score (0.0 to 1.0).")
    parser.add_argument("--save", action="store_true", help="Save the output to a JSON file.")
    
    args = parser.parse_args()
    
    try:
        entry = json.loads(args.data)
    except json.JSONDecodeError:
        print("Error: --data must be a valid JSON string.")
        exit(1)
        
    process_single_entry(entry, initial_trust=args.trust, save_output=args.save)
