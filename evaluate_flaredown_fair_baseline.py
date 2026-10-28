
"""
Fair grouped-fold comparison of ML models vs. the Flaredown naive baseline.

Run from the ra/ project root after data/ra_processed_v3.csv exists:
    python evaluate_flaredown_fair_baseline.py

Output:
    outputs/ra_v3_fair_baseline_results.json
"""

import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA = "data/ra_processed_v3.csv"
OUT = "outputs/ra_v3_fair_baseline_results.json"

FEATS = [
    "past_mean_severity",
    "past_std_severity",
    "past_trend",
    "past_last_severity",
    "past_engagement",
    "personal_baseline",
    "age",
    "sex",
]

RANDOM_STATE = 42
N_REPEATS = 6
N_SPLITS = 5


def models():
    """Create fresh model instances for each evaluation fold."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def safe_metrics(y_true, probabilities):
    """Calculate classification metrics at a fixed 0.5 threshold."""
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "recall_at_0_5": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "precision_at_0_5": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
    }


def summarize(metric_rows):
    """Summarize metrics across valid held-out folds."""
    summary = {}

    if not metric_rows:
        return summary

    for metric_name in metric_rows[0]:
        values = np.asarray(
            [row[metric_name] for row in metric_rows],
            dtype=float,
        )
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "n_folds": int(len(values)),
        }

    return summary


def main():
    if not os.path.exists(DATA):
        raise FileNotFoundError(
            f"Missing {DATA}. Run the Flaredown preparation script first."
        )

    df = pd.read_csv(DATA)

    required_columns = FEATS + ["user_id", "flare_next_3d"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The processed dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Convert feature and target columns to numeric.
    for column in FEATS + ["flare_next_3d"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns).copy()

    # Reject invalid binary labels.
    df = df[df["flare_next_3d"].isin([0, 1])].copy()
    df["flare_next_3d"] = df["flare_next_3d"].astype(int)

    if df.empty:
        raise ValueError("No valid rows remain after data cleaning.")

    if df["user_id"].nunique() < N_SPLITS:
        raise ValueError(
            f"At least {N_SPLITS} unique patients are required for "
            "grouped cross-validation."
        )

    X = df[FEATS].astype(float).to_numpy()
    y = df["flare_next_3d"].to_numpy()
    groups = df["user_id"].to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError(
            "The dataset must contain both positive and negative labels."
        )

    # The baseline is a simple historical severity ratio.
    # Zero or invalid personal baselines cannot produce a valid ratio.
    baseline = (
        df["past_mean_severity"]
        / df["personal_baseline"].replace(0, np.nan)
    )
    baseline = baseline.replace([np.inf, -np.inf], np.nan)

    zero_baseline = df["personal_baseline"].eq(0)

    results = {
        "dataset": {
            "n_rows": int(len(df)),
            "n_patients": int(df["user_id"].nunique()),
            "positive_rate": float(y.mean()),
            "zero_baseline_rows": int(zero_baseline.sum()),
            "zero_baseline_rate": float(zero_baseline.mean()),
        },
        "evaluation": (
            f"{N_REPEATS} repeats x {N_SPLITS}-fold "
            "StratifiedGroupKFold, grouped by user_id. "
            "All reported model metrics use held-out patients."
        ),
        "models": {},
        "naive_baseline": {},
        "notes": [
            (
                "Naive score = past_mean_severity / personal_baseline. "
                "Rows with zero or invalid baselines are excluded from "
                "baseline metrics only."
            ),
            (
                "Models and baseline use the same held-out folds, but "
                "baseline metrics use only rows with valid baseline scores. "
                "Their evaluation populations can therefore differ."
            ),
            (
                "Folds containing only one target class are skipped for "
                "all metrics in that fold."
            ),
            (
                "Balanced accuracy, recall, and precision use a fixed "
                "probability threshold of 0.5 for ML models."
            ),
            (
                "The target represents future symptom worsening under "
                "the project's operational definition, not a "
                "clinician-adjudicated rheumatoid arthritis flare."
            ),
            (
                "Fold standard deviations describe variation across "
                "overlapping repeated cross-validation folds; they are "
                "not independent-sample confidence intervals."
            ),
        ],
    }

    model_scores = {
        name: [] for name in models()
    }
    baseline_scores = []
    valid_fold_count = 0
    skipped_fold_count = 0

    for repeat in range(N_REPEATS):
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=repeat,
        )

        for train_indices, test_indices in cv.split(X, y, groups):
            y_test = y[test_indices]

            # AUC and average precision are not comparable in a
            # single-class test fold, so skip that fold consistently.
            if len(np.unique(y_test)) < 2:
                skipped_fold_count += 1
                continue

            valid_fold_count += 1

            # Evaluate both ML models on exactly the same held-out fold.
            for name, model in models().items():
                model.fit(X[train_indices], y[train_indices])
                probabilities = model.predict_proba(
                    X[test_indices]
                )[:, 1]

                model_scores[name].append(
                    safe_metrics(y_test, probabilities)
                )

            # Evaluate the naive baseline on valid held-out rows only.
            fold_baseline = baseline.iloc[test_indices].to_numpy()
            valid_baseline = np.isfinite(fold_baseline)

            if (
                valid_baseline.sum() > 0
                and len(np.unique(y_test[valid_baseline])) == 2
            ):
                y_baseline = y_test[valid_baseline]
                score_baseline = fold_baseline[valid_baseline]

                baseline_scores.append(
                    {
                        "roc_auc": float(
                            roc_auc_score(y_baseline, score_baseline)
                        ),
                        "pr_auc": float(
                            average_precision_score(
                                y_baseline, score_baseline
                            )
                        ),
                    }
                )

    results["models"] = {
        name: summarize(scores)
        for name, scores in model_scores.items()
    }

    results["naive_baseline"] = summarize(baseline_scores)
    results["n_valid_test_folds"] = int(valid_fold_count)
    results["n_skipped_test_folds"] = int(skipped_fold_count)
    results["n_baseline_evaluated_folds"] = int(len(baseline_scores))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    with open(OUT, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)

    print(json.dumps(results, indent=2, allow_nan=False))
    print(f"\nResults saved to: {OUT}")


if __name__ == "__main__":
    main()
