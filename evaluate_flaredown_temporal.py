"""Patient-disjoint, time-aware evaluation of the Flaredown symptom proxy.

This script rebuilds dated checkpoints in a temporary directory using the
existing preparation implementation, then evaluates fixed models without
writing model artifacts. The target is an operational symptom-worsening proxy,
not a clinician-confirmed RA flare.
"""

import json
import math
import os
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "ra.csv"
OUTPUT_PATH = ROOT / "outputs" / "flaredown_temporal_evaluation.json"
FEATURES = (
    "past_mean_severity",
    "past_std_severity",
    "past_trend",
    "past_last_severity",
    "past_engagement",
    "personal_baseline",
    "age",
    "sex",
)
TARGET = "flare_next_3d"
PATIENT = "user_id"
CHECKPOINT_DATE = "checkpoint_date"
LABEL_HORIZON_DAYS = 3
TEST_DATE_FRACTION = 0.20
RANDOM_STATE = 42


def make_temporal_patient_split(
    frame: pd.DataFrame,
    test_date_fraction: float = TEST_DATE_FRACTION,
    label_horizon_days: int = LABEL_HORIZON_DAYS,
) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    """Return patient-disjoint train/test masks separated by a date cutoff.

    The cutoff is selected from distinct checkpoint dates only. Patients with
    checkpoints on both sides of the cutoff are excluded. The training
    patient's final checkpoint plus the full label horizon must precede the
    test period.
    """
    required = {PATIENT, CHECKPOINT_DATE}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing temporal split columns: {sorted(missing)}")
    if not 0 < test_date_fraction < 1:
        raise ValueError("test_date_fraction must be between 0 and 1.")
    if label_horizon_days < 0:
        raise ValueError("label_horizon_days cannot be negative.")

    dates = pd.to_datetime(frame[CHECKPOINT_DATE], errors="coerce")
    if dates.isna().any():
        raise ValueError("Checkpoint dates must all be valid before splitting.")
    if frame[PATIENT].isna().any():
        raise ValueError("Patient IDs must be present before splitting.")

    unique_dates = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
    cutoff_index = math.floor((1 - test_date_fraction) * len(unique_dates))
    if cutoff_index <= 0 or cutoff_index >= len(unique_dates):
        raise ValueError("Not enough distinct checkpoint dates for this split.")
    cutoff = unique_dates[cutoff_index]

    dated = frame[[PATIENT]].copy()
    dated[CHECKPOINT_DATE] = dates.to_numpy()
    patient_dates = dated.groupby(PATIENT, sort=False)[CHECKPOINT_DATE].agg(
        first="min", last="max"
    )
    train_patients = patient_dates.index[
        patient_dates["last"] + pd.Timedelta(days=label_horizon_days) < cutoff
    ]
    test_patients = patient_dates.index[patient_dates["first"] >= cutoff]

    train_mask = frame[PATIENT].isin(train_patients)
    test_mask = frame[PATIENT].isin(test_patients)
    if (train_mask & test_mask).any():
        raise AssertionError("Temporal split assigned a patient to both sets.")
    if train_mask.sum() and test_mask.sum():
        train_latest_label_end = (
            dates.loc[train_mask].max()
            + pd.Timedelta(days=label_horizon_days)
        )
        test_earliest_checkpoint = dates.loc[test_mask].min()
        if train_latest_label_end >= test_earliest_checkpoint:
            raise AssertionError("Training label horizon overlaps test time.")
    return train_mask, test_mask, cutoff


def _pipeline(model, features: pd.DataFrame, scale_numeric: bool) -> Pipeline:
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    transformers = []
    if numeric:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(
            ("numeric", Pipeline(numeric_steps), numeric)
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(transformers, remainder="drop"),
            ),
            ("model", model),
        ]
    )


def _metric_summary(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    inclusive: bool = True,
):
    predictions = (
        scores >= threshold if inclusive else scores > threshold
    ).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }


def _class_counts(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts().to_dict()
    return {str(label): int(counts.get(label, 0)) for label in (0, 1)}


def _prepare_checkpoints(output_path: Path, temporary_directory: str):
    from prepare_flaredown_v2 import CHUNK, process_flaredown_chunks

    def chunks_factory():
        return pd.read_csv(RAW_PATH, chunksize=CHUNK, dtype=str)

    return process_flaredown_chunks(
        chunks_factory,
        output_path=output_path,
        staging_directory=temporary_directory,
    )


def run_evaluation(output_path: Path = OUTPUT_PATH) -> dict:
    if not RAW_PATH.is_file():
        raise FileNotFoundError(f"Required source file is missing: {RAW_PATH}")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing evaluation output: {output_path}"
        )

    with tempfile.TemporaryDirectory(prefix="flaredown_temporal_eval_") as tmp:
        checkpoint_path = Path(tmp) / "dated_checkpoints.csv"
        preparation = _prepare_checkpoints(checkpoint_path, tmp)
        frame = pd.read_csv(checkpoint_path, dtype={PATIENT: "string"})

    missing = set(FEATURES + (TARGET, PATIENT, CHECKPOINT_DATE)).difference(
        frame.columns
    )
    if missing:
        raise ValueError(f"Prepared checkpoints lack required fields: {sorted(missing)}")
    frame[CHECKPOINT_DATE] = pd.to_datetime(
        frame[CHECKPOINT_DATE], errors="raise"
    )
    for column in FEATURES:
        if column != "sex":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["sex"] = frame["sex"].astype("string").replace("", pd.NA)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    if not frame[TARGET].isin([0, 1]).all():
        raise ValueError("The prepared target must contain only binary labels.")

    train_mask, test_mask, cutoff = make_temporal_patient_split(frame)
    train = frame.loc[train_mask].copy()
    test = frame.loc[test_mask].copy()
    if train.empty or test.empty:
        raise ValueError(
            "The chronological patient-disjoint split produced an empty "
            "train or test set; no metrics were calculated."
        )
    if train[TARGET].nunique() != 2 or test[TARGET].nunique() != 2:
        raise ValueError(
            "Both train and test must contain both target classes for ROC-AUC; "
            "no model was fitted."
        )

    x_train, y_train = train.loc[:, FEATURES], train[TARGET]
    x_test, y_test = test.loc[:, FEATURES], test[TARGET]
    candidates = {
        "Logistic Regression": _pipeline(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            x_train,
            scale_numeric=True,
        ),
        "Random Forest": _pipeline(
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            x_train,
            scale_numeric=False,
        ),
    }

    y_test_array = y_test.to_numpy()
    metrics = {}
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics[name] = _metric_summary(y_test_array, probabilities, 0.5)

    baseline_score = (
        test["past_mean_severity"] - 1.15 * test["personal_baseline"]
    ).to_numpy(dtype=float)
    if not np.isfinite(baseline_score).all():
        raise ValueError("The fixed baseline score contains non-finite values.")
    metrics["Past severity vs personal baseline"] = _metric_summary(
        y_test_array, baseline_score, 0.0, inclusive=False
    )

    train_patients = set(train[PATIENT].dropna().astype(str))
    test_patients = set(test[PATIENT].dropna().astype(str))
    if train_patients.intersection(test_patients):
        raise AssertionError("A patient appears in both evaluation partitions.")

    result = {
        "task": (
            "Operational future symptom-worsening proxy; not a "
            "clinician-confirmed RA flare."
        ),
        "input_source": str(RAW_PATH.relative_to(ROOT)),
        "temporary_checkpoint_rows": int(preparation["rows"]),
        "split": {
            "method": (
                "Distinct checkpoint dates sorted chronologically; cutoff "
                "at the 80th-percentile date. Train patients have every "
                "checkpoint plus the full 3-day label horizon before cutoff. "
                "Test patients have every checkpoint on/after cutoff. Patients "
                "crossing the boundary are excluded."
            ),
            "cutoff": cutoff.date().isoformat(),
            "training_checkpoint_date_range": [
                train[CHECKPOINT_DATE].min().date().isoformat(),
                train[CHECKPOINT_DATE].max().date().isoformat(),
            ],
            "training_latest_label_horizon_end": (
                train[CHECKPOINT_DATE].max()
                + pd.Timedelta(days=LABEL_HORIZON_DAYS)
            ).date().isoformat(),
            "test_checkpoint_date_range": [
                test[CHECKPOINT_DATE].min().date().isoformat(),
                test[CHECKPOINT_DATE].max().date().isoformat(),
            ],
            "excluded_boundary_patients": int(
                frame[PATIENT].nunique()
                - len(train_patients)
                - len(test_patients)
            ),
            "patient_overlap": 0,
        },
        "training": {
            "rows": int(len(train)),
            "patients": int(len(train_patients)),
            "class_counts": _class_counts(y_train),
        },
        "test": {
            "rows": int(len(test)),
            "patients": int(len(test_patients)),
            "class_counts": _class_counts(y_test),
        },
        "features": list(FEATURES),
        "preprocessing": (
            "Imputation, numeric scaling for Logistic Regression, and "
            "categorical one-hot encoding are fitted within each training "
            "pipeline only."
        ),
        "selection_and_thresholds": (
            "Both candidate models were fixed before evaluation; no test-set "
            "model selection or threshold tuning. Model class threshold=0.5. "
            "Baseline score is past_mean_severity - 1.15 * personal_baseline; "
            "its prespecified decision threshold is 0."
        ),
        "metrics": metrics,
        "limitations": [
            "Outcomes are self-reported symptom-worsening proxies, not adjudicated RA flares.",
            "This is one temporal holdout from a small cohort; temporal external validation is absent.",
            "The split excludes patients with checkpoints spanning the cutoff and may select a nonrepresentative subset.",
            "Metrics are checkpoint-weighted, so participants with more checkpoints contribute more observations.",
            "The target may use fewer than all three future calendar days when some days have no symptom report.",
        ],
    }
    _write_new_json(output_path, result)
    return result


def _write_new_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="New JSON output path; existing files are never overwritten.",
    )
    args = parser.parse_args()
    print(json.dumps(run_evaluation(args.output), indent=2, allow_nan=False))
