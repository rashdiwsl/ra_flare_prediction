
"""Isolated evaluation of the Flaredown candidate dataset.

Does not modify the original training script or existing model/results files.
The target is a dataset-defined symptom-worsening proxy, not a clinical diagnosis.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from xgboost import XGBClassifier

# Reuse the established preprocessing/model functions without running main().
from importlib.util import spec_from_file_location, module_from_spec

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "1_train_models_v2.py"
DATA = BASE / "data" / "ra_processed_v3_candidate.csv"
OUTPUT = BASE / "outputs"
MODEL_DIR = BASE / "models"

RESULTS_PATH = OUTPUT / "flaredown_candidate_grouped_results.json"
MODEL_PATH = MODEL_DIR / "flaredown_candidate_model.pkl"
CHART_PATH = OUTPUT / "flaredown_candidate_comparison.png"

TARGET = "flare_next_3d"
GROUP = "user_id"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    for path in (DATA, SOURCE):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    for path in (RESULTS_PATH, MODEL_PATH, CHART_PATH):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}\n"
                "Choose new output filenames before continuing."
            )

    spec = spec_from_file_location("existing_training_helpers", SOURCE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    df = pd.read_csv(DATA)

    required = {
        GROUP,
        "checkpoint_date",
        TARGET,
        *module.RA_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Candidate dataset is missing columns: {missing}")

    if df[GROUP].isna().any() or df[TARGET].isna().any():
        raise ValueError("Patient IDs and target labels must not be missing.")

    X = module.select_ra_features(df)
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    groups = df[GROUP].astype(str)

    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected binary target labels 0 and 1.")

    # Split by patient: no patient's checkpoints can cross partitions.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]

    overlap = set(g_train) & set(g_test)
    if overlap:
        raise RuntimeError("Patient leakage detected between train and test.")

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "This fixed patient split does not contain both classes in train "
            "and test. Do not use row-level splitting to work around this."
        )

    candidates = module.make_candidates(X_train, y_train, balanced=True)
    n_splits = min(5, g_train.nunique())
    cv_iterator = GroupKFold(n_splits=n_splits).split(
        X_train, y_train, groups=g_train
    )

    # Keep only folds whose training and validation portions contain both classes.
    folds = [
        (tr, va)
        for tr, va in cv_iterator
        if y_train.iloc[tr].nunique() == 2
        and y_train.iloc[va].nunique() == 2
    ]
    if not folds:
        raise ValueError("No valid patient-grouped CV folds with both classes.")

    cv_scores = {}
    for name, pipeline in candidates.items():
        aucs = []
        for tr, va in folds:
            fitted = clone(pipeline)
            fitted.fit(X_train.iloc[tr], y_train.iloc[tr])
            probabilities = fitted.predict_proba(X_train.iloc[va])[:, 1]
            aucs.append(roc_auc_score(y_train.iloc[va], probabilities))
        cv_scores[name] = float(np.mean(aucs))

    best_name = max(cv_scores, key=cv_scores.get)

    # Evaluate each candidate on the same untouched patient holdout.
    test_results = {}
    fitted_models = {}
    for name, pipeline in candidates.items():
        fitted = clone(pipeline)
        fitted.fit(X_train, y_train)
        probabilities = fitted.predict_proba(X_test)[:, 1]
        predictions = fitted.predict(X_test)

        test_results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "weighted_f1": float(
                f1_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
        }
        fitted_models[name] = fitted

    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)

    report = {
        "experiment": "Flaredown candidate; patient-grouped holdout",
        "dataset": str(DATA.relative_to(BASE)),
        "target": (
            "Dataset-defined next-3-day symptom-worsening proxy; "
            "not a clinically validated RA flare label."
        ),
        "rows": int(len(df)),
        "unique_patients": int(groups.nunique()),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_patients": int(g_train.nunique()),
        "test_patients": int(g_test.nunique()),
        "patient_overlap": len(overlap),
        "positive_checkpoints": int(y.sum()),
        "negative_checkpoints": int((y == 0).sum()),
        "grouped_cv_folds_used": int(len(folds)),
        "cv_roc_auc": cv_scores,
        "selected_model": best_name,
        "held_out_test_metrics": test_results,
        "selected_model_test_classification_report": classification_report(
            y_test, best_predictions, zero_division=0, output_dict=True
        ),
        "selected_model_test_confusion_matrix": confusion_matrix(
            y_test, best_predictions
        ).tolist(),
        "limitations": [
            "Checkpoints are repeated observations nested within patients.",
            "Only 71 patients are available; estimates may be unstable.",
            "The target is a symptom-worsening proxy, not a clinical flare diagnosis.",
            "This experiment evaluates one dataset, not multimodal fusion.",
        ],
    }

    # Write results and selected model to new paths only.
    with RESULTS_PATH.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    joblib.dump(
        {
            "model": best_model,
            "cols": X.columns.tolist(),
            "best": best_name,
            "target_description": report["target"],
            "split": "GroupShuffleSplit by user_id",
        },
        MODEL_PATH,
    )

    # Create a new chart; never reuse the original comparison chart name.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(test_results)
    scores = [test_results[name]["roc_auc"] for name in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, scores)
    ax.set_title("Flaredown Candidate: Patient-Held-Out ROC-AUC")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=140)
    plt.close(fig)

    print("\nExperiment completed.")
    print(f"Patients: {groups.nunique()} | Train: {g_train.nunique()} | Test: {g_test.nunique()}")
    print(f"Patient overlap: {len(overlap)}")
    print(f"Selected by grouped CV ROC-AUC: {best_name}")
    print("\nHeld-out test metrics:")
    for name, metrics in test_results.items():
        print(name, {key: round(value, 4) for key, value in metrics.items()})
    print(f"\nResults: {RESULTS_PATH}")
    print(f"Model:   {MODEL_PATH}")
    print(f"Chart:   {CHART_PATH}")


if __name__ == "__main__":
    main()
