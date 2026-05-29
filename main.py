# main.py

import json
import pandas as pd

from src.pipeline import run_pipeline
from src.config import DATA_PATH


if __name__ == "__main__":

    print(f"Loading data from: {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    print(f"Loaded {len(df)} records, {len(df.columns)} columns")

    results = run_pipeline(df)

    print("\nPipeline completed successfully.")