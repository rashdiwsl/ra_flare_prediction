import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CANDIDATE_DATA = "data/ra_processed_v3_candidate.csv"
RAW_DATA = "data/ra.csv"
OUTPUT = "outputs/flaredown_weather_enhancement_results.json"

BASE_FEATURES = [
    "past_mean_severity",
    "past_std_severity",
    "past_trend",
    "past_last_severity",
    "past_engagement",
    "personal_baseline",
    "age",
    "sex",
]

WEATHER_VARIABLES = [
    "humidity",
    "precip_intensity",
    "pressure",
    "temperature_max",
    "temperature_min",
]

TARGET = "flare_next_3d"
N_REPEATS = 6
N_SPLITS = 5


def load_weather():
    """Build daily weather summaries from raw records in chunks."""
    partial_tables = []

    columns = [
        "user_id", "checkin_date", "trackable_type",
        "trackable_name", "trackable_value",
    ]

    for chunk in pd.read_csv(
        RAW_DATA,
        usecols=columns,
        dtype={"user_id": str},
        chunksize=200000,
    ):
        chunk = chunk[chunk["trackable_type"] == "Weather"].copy()
        chunk = chunk[
            chunk["trackable_name"].isin(WEATHER_VARIABLES)
        ].copy()

        if chunk.empty:
            continue

        chunk["checkin_date"] = pd.to_datetime(
            chunk["checkin_date"], errors="coerce"
        ).dt.normalize()
        chunk["trackable_value"] = pd.to_numeric(
            chunk["trackable_value"], errors="coerce"
        )

        chunk = chunk.dropna(
            subset=["user_id", "checkin_date", "trackable_value"]
        )

        if chunk.empty:
            continue

        # Average duplicate observations of a variable within a day.
        daily = (
            chunk.groupby(
                ["user_id", "checkin_date", "trackable_name"],
                as_index=False,
            )["trackable_value"].mean()
        )

        wide = daily.pivot(
            index=["user_id", "checkin_date"],
            columns="trackable_name",
            values="trackable_value",
        ).reset_index()

        partial_tables.append(wide)

        del chunk, daily, wide

    if not partial_tables:
        raise ValueError("No usable numeric weather records were found.")

    # A user-date may span two CSV chunks; combine partial summaries.
    weather = pd.concat(partial_tables, ignore_index=True)
    weather = weather.groupby(
        ["user_id", "checkin_date"], as_index=False
    )[WEATHER_VARIABLES].mean()

    return weather


def build_dataset():
    df = pd.read_csv(CANDIDATE_DATA, dtype={"user_id": str})

    required = BASE_FEATURES + [
        "user_id", "checkpoint_date", TARGET
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Missing candidate columns: " + ", ".join(missing))

    df["checkpoint_date"] = pd.to_datetime(
        df["checkpoint_date"], errors="coerce"
    ).dt.normalize()

    for col in BASE_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["user_id", "checkpoint_date", TARGET]
    ).copy()
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    if df.duplicated(["user_id", "checkpoint_date"]).any():
        raise ValueError("Duplicate patient/checkpoint-date rows found.")

    weather = load_weather()

    # Expand each checkpoint to the seven strictly prior calendar days.
    expanded = df[["user_id", "checkpoint_date"]].copy()
    expanded["checkpoint_id"] = np.arange(len(expanded))

    expanded = expanded.loc[
        expanded.index.repeat(7)
    ].copy()
    expanded["day_offset"] = np.tile(
        np.arange(1, 8), len(df)
    )
    expanded["weather_date"] = (
        expanded["checkpoint_date"]
        - pd.to_timedelta(expanded["day_offset"], unit="D")
    )

    matched = expanded.merge(
        weather,
        left_on=["user_id", "weather_date"],
        right_on=["user_id", "checkin_date"],
        how="left",
        validate="many_to_one",
    )

    # Each weather feature uses only the previous seven calendar days.
    aggregations = {}
    for variable in WEATHER_VARIABLES:
        aggregations[variable] = ["mean", "std"]

    summary = matched.groupby("checkpoint_id").agg(aggregations)
    summary.columns = [
        f"weather_{variable}_{stat}"
        for variable, stat in summary.columns
    ]
    summary = summary.reset_index()

    df = df.reset_index(drop=True)
    df["checkpoint_id"] = np.arange(len(df))
    df = df.merge(summary, on="checkpoint_id", how="left")

    weather_features = [
        f"weather_{variable}_{stat}"
        for variable in WEATHER_VARIABLES
        for stat in ("mean", "std")
    ]

    # Both models use exactly the same weather-covered checkpoints.
    df[weather_features] = df[weather_features].replace(
        [np.inf, -np.inf], np.nan
    )
    covered = df[weather_features].notna().any(axis=1)
    evaluated = df.loc[covered].copy()

    if evaluated["user_id"].nunique() < N_SPLITS:
        raise ValueError("Too few patients for grouped cross-validation.")
    if evaluated[TARGET].nunique() < 2:
        raise ValueError("Both target classes are required.")

    return df, evaluated, weather_features


def make_models(features):
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }


def summarize(rows):
    output = {}
    for metric in ("roc_auc", "pr_auc"):
        values = np.asarray(
            [row[metric] for row in rows], dtype=float
        )
        values = values[np.isfinite(values)]
        output[metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std()) if len(values) else None,
            "n_folds": int(len(values)),
        }
    return output


def evaluate(df, weather_features):
    y = df[TARGET].to_numpy()
    groups = df["user_id"].to_numpy()

    feature_sets = {
        "baseline": BASE_FEATURES,
        "baseline_plus_weather": BASE_FEATURES + weather_features,
    }

    scores = {
        f"{model_name} | {set_name}": []
        for set_name in feature_sets
        for model_name in ("Logistic Regression", "Random Forest")
    }
    fold_rows = []
    skipped = 0

    for repeat in range(N_REPEATS):
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=repeat,
        )

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(np.zeros(len(df)), y, groups), start=1
        ):
            y_test = y[test_idx]
            if len(np.unique(y_test)) < 2:
                skipped += 1
                continue

            row = {"repeat": repeat, "fold": fold}

            for set_name, features in feature_sets.items():
                X = df[features].replace(
                    [np.inf, -np.inf], np.nan
                )

                for model_name, model in make_models(features).items():
                    model.fit(X.iloc[train_idx], y[train_idx])
                    probabilities = model.predict_proba(
                        X.iloc[test_idx]
                    )[:, 1]

                    metrics = {
                        "roc_auc": float(
                            roc_auc_score(y_test, probabilities)
                        ),
                        "pr_auc": float(
                            average_precision_score(y_test, probabilities)
                        ),
                    }

                    key = f"{model_name} | {set_name}"
                    scores[key].append(metrics)
                    row[key] = metrics

            fold_rows.append(row)

    return {
        "summary": {
            key: summarize(value) for key, value in scores.items()
        },
        "fold_results": fold_rows,
        "valid_folds": len(fold_rows),
        "skipped_folds": skipped,
    }


def main():
    for path in (CANDIDATE_DATA, RAW_DATA):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    all_rows, evaluated, weather_features = build_dataset()

    result = evaluate(evaluated, weather_features)

    result["dataset"] = {
        "candidate_checkpoints": int(len(all_rows)),
        "candidate_patients": int(all_rows["user_id"].nunique()),
        "weather_covered_checkpoints": int(len(evaluated)),
        "weather_covered_patients": int(evaluated["user_id"].nunique()),
        "weather_covered_positive_outcomes": int(
            evaluated[TARGET].sum()
        ),
        "weather_covered_positive_rate": float(evaluated[TARGET].mean()),
        "weather_coverage_rate": float(len(evaluated) / len(all_rows)),
        "target": (
            "flare_next_3d operational proxy for future symptom worsening; "
            "not a clinician-adjudicated RA flare"
        ),
    }

    result["method"] = {
        "comparison": (
            "Baseline and baseline-plus-weather use the same weather-covered "
            "checkpoints and the same patient-grouped folds."
        ),
        "weather_window": (
            "Seven calendar days strictly before checkpoint_date; "
            "checkpoint-day and future weather are excluded."
        ),
        "weather_features": weather_features,
        "models": ["Logistic Regression", "Random Forest"],
        "validation": (
            "6 repeats x 5-fold StratifiedGroupKFold grouped by user_id"
        ),
        "preprocessing": (
            "Median imputation and scaling for Logistic Regression are "
            "fitted within each training fold. Random Forest imputation is "
            "also fitted within each training fold."
        ),
        "metrics": ["ROC-AUC", "PR-AUC"],
        "limitations": [
            "Exploratory retrospective analysis, not clinical validation.",
            "Weather coverage may select a non-representative patient subset.",
            "Weather metadata may be observational or user-entered; its "
            "provenance and timing should be verified.",
            "Fold standard deviations are not independent-sample confidence "
            "intervals.",
            "A weather improvement does not establish causal weather effects."
        ],
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)

    print(json.dumps(result["dataset"], indent=2))
    print("\nModel summary:")
    print(json.dumps(result["summary"], indent=2))
    print("\nValid folds:", result["valid_folds"])
    print("Skipped folds:", result["skipped_folds"])
    print("\nFull results saved to:", OUTPUT)


if __name__ == "__main__":
    main()
