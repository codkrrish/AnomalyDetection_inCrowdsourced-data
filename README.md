# Crowdsourced Anomaly Detection

This project is an anomaly detection pipeline designed to identify fake, invalid, or anomalous entries in crowdsourced salary data. It is built to ensure high data quality by filtering out malicious or incorrect submissions before they can degrade business insights.

## Dataset

I have used a synthetic dataset for this project, it is modeled based on the input features of the submission forms provided by [levels.fyi](https://www.levels.fyi/), a platform that provides valuable business insights and compensation transparency based on crowdsourced data. The synthetic data attempts to mimic real-world compensation structures, including base salaries, bonuses, stock grants, years of experience, job families, and geographic locations. I have also added a few anomalies to the dataset to test the model.

## How It Works

The current anomaly detection model uses a multi-layered approach combining rule-based heuristics with unsupervised machine learning to classify entries as `Approve`, `Review`, or `Reject`.

### 1. Rule-Based Validation
The pipeline first applies strict mathematical logic and common-sense rules to flag obvious fakes. Examples include:
- Negative experience or salary values.
- "Years at Company" exceeding "Total Years of Experience".
- Total compensation arithmetic mismatches (e.g. TC is significantly less than Base Salary).
- Gibberish company names.
- Known large public companies incorrectly flagged with early startup funding stages.

### 2. Segmentation and Statistical Outlier Detection
Entries are strictly segregated based on their **Job Family** and **Job Level** to ensure fair, contextual comparisons.
- **Large Segments ($\ge 10$ entries)**: Evaluated using **Local Outlier Factor (LOF)** to detect local anomalies in compensation compared to immediate peers.
- **Medium Segments ($5 - 9$ entries)**: Evaluated using a **K-Nearest Neighbors (KNN)** distance-based approach to identify outliers where LOF lacks sufficient density.
- **Small Segments ($< 5$ entries)**: Skips statistical scoring due to lack of data and is automatically flagged for manual `Review`.

### 3. Consistency and Trust Scoring
- **Consistency Score**: A weighted score combining rule violations and statistical anomaly scores to determine overall data reliability.
- **User Trust Score**: Users are assigned a trust score. Submitting high-consistency genuine entries rewards the user with a trust score boost (+3%), encouraging reliable crowdsourcing over time. Importantly, the trust score does not override statistical facts—classifications remain purely data-driven.

### 4. Final Decision Engine
The pipeline synthesizes the flags and scores into a final classification:
- **Approve**: High consistency score with no hard flags.
- **Reject**: Extremely low consistency or mathematically impossible values.
- **Review**: Borderline consistency, duplicates, benchmark violations, or statistically anomalous LOF/KNN scores.

## Inference Pipeline
A standalone script (`infer.py`) allows for the evaluation of new, incoming single entries. It routes the entry through the same pipeline, evaluates it against the pre-trained segment models, and generates an isolated scatter plot visualizing exactly where the new entry lands among the historical dataset.

------------------------


**Note:** I am continuously working on this project and actively experimenting with the model. My goal is to find and try better and more advanced classification methods to handle the complex nuances of crowdsourced data. Future updates may introduce new algorithms and improved validation techniques.