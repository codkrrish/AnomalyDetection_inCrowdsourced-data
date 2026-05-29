# src/config.py

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "compensation_dataset_1250_entries_labeled.json"

)

MODEL_SAVE_DIR = "models/"

PLOT_DIR = "reports/plots/"
OUTPUT_DIR = "data/outputs/"
OUTPUT_PATH = "data/outputs/validated_output.csv"

# ──────────────────────────────────────────────
# Pipeline Parameters
# ──────────────────────────────────────────────

TEST_SIZE = 0.2
RANDOM_STATE = 42

MIN_SEGMENT_SIZE = 10
KNN_MIN_SEGMENT_SIZE = 5
LOF_N_NEIGHBORS = 5
LOF_CONTAMINATION = 0.10
CONSISTENCY_THRESHOLD = 0.50
TRUST_SCORE_BOOST = 0.03

# Weights for final consistency score (must sum to 1.0)
WEIGHT_RULE = 0.30
WEIGHT_LOF = 0.70

# ──────────────────────────────────────────────
# Columns to drop before processing
# ──────────────────────────────────────────────

DROP_COLUMNS = ["uuid", "_anomaly_type", "commentsAndRepliesCount", "threadId"]

# ──────────────────────────────────────────────
# LOF features — all must be numeric and USD-normalized
# ──────────────────────────────────────────────

LOF_FEATURES = [
    "log_base_salary",
    "log_tc",
    "avgAnnualBonusValue_USD",
    "avgAnnualStockGrantValue_USD",
    "yearsOfExperience",
    "bonus_ratio",
    "stock_ratio",
    "tc_to_base",
]

# Segmentation keys for LOF
SEGMENT_KEYS = ["jobFamily_clean", "region"]

# ──────────────────────────────────────────────
# Level ordinal mapping (IC + Manager tracks)
# ──────────────────────────────────────────────

LEVEL_ORDINAL = {
    # IC track
    "l1": 1, "l2": 2, "l3": 3, "l4": 4, "l5": 5, "l6": 6, "l7": 7,
    "ic1": 1, "ic2": 2, "ic3": 3, "ic4": 4, "ic5": 5, "ic6": 6,
    "e1": 1, "e2": 2, "e3": 3, "e4": 4, "e5": 5, "e6": 6, "e7": 7,
    "junior": 1, "mid": 2, "mid-level": 2, "senior": 3,
    "staff": 4, "principal": 5, "fellow": 6, "distinguished": 7,
    "swe i": 1, "swe ii": 2, "senior swe": 3, "staff swe": 4,
    # Manager track
    "m1": 1, "m2": 2, "m3": 3, "m4": 4, "m5": 5,
    "manager": 1, "senior manager": 2, "director": 3,
    "senior director": 4, "vp": 5, "svp": 6, "evp": 7, "cto": 8,
    "partner": 3,
}

# ──────────────────────────────────────────────
# Country → Region mapping
# ──────────────────────────────────────────────

COUNTRY_REGION = {
    1: "US", 113: "India", 826: "UK", 276: "Germany",
    250: "France", 380: "Italy", 724: "Spain", 124: "Canada",
    36: "Australia", 392: "Japan", 156: "China", 76: "Brazil",
}

# ──────────────────────────────────────────────
# Company classification
# ──────────────────────────────────────────────

FAANG = {"google", "meta", "amazon", "apple", "microsoft", "netflix"}

TIER2 = {
    "oracle", "salesforce", "servicenow", "snowflake", "adobe", "nvidia",
    "uber", "lyft", "airbnb", "stripe", "linkedin", "twitter", "x",
    "databricks", "confluent", "figma", "notion", "openai", "anthropic",
}

PUBLIC_COMPANIES = {
    "oracle", "microsoft", "google", "amazon", "apple", "meta", "netflix",
    "salesforce", "servicenow", "snowflake", "adobe", "ibm", "intel",
    "deere & company", "cigna", "hca healthcare", "jpmorgan", "goldman sachs",
    "morgan stanley", "bank of america", "wells fargo", "visa", "mastercard",
    "unitedhealth", "cvs", "walmart", "target", "boeing", "lockheed martin",
    "accenture", "infosys", "wipro", "tcs", "capgemini",
}

STARTUP_STAGES = {"seed", "series a", "series b", "series c", "series d"}

# ──────────────────────────────────────────────
# Salary benchmarks by (jobFamily, level_group, region)
# ──────────────────────────────────────────────

BENCHMARKS = {
    ("Software Engineer", "junior", "US"): {"min": 90000, "max": 220000},
    ("Software Engineer", "senior", "US"): {"min": 180000, "max": 500000},
}

# ──────────────────────────────────────────────
# Company size mapping
# ──────────────────────────────────────────────

COMPANY_SIZE_MAP = {
    "1-10": 5, "11-50": 30, "51-200": 125, "201-500": 350,
    "501-1000": 750, "1001-5000": 3000, "5000+": 10000,
}

# ──────────────────────────────────────────────
# Rule columns used for scoring
# ──────────────────────────────────────────────

RULE_COLUMNS = [
    "flag_gibberish_company",
    "flag_yal_gt_yac",
    "flag_yac_gt_yoe",
    "flag_negative_experience",
    "flag_zero_or_negative_base",
    "flag_tc_less_than_base",
    "flag_junior_level_high_yoe",
    "flag_bonus_pct_mismatch",
    "flag_public_co_startup_stage",
    "flag_tc_arithmetic_mismatch",
    "flag_companysize_mismatch",
]