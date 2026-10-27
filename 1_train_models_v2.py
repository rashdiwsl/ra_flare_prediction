"""
1_train_models_v2.py
====================
Leakage-aware training and evaluation for four separate datasets.

Targets are different and evaluated separately:
1. RA Clinical: dataset-defined future symptom-worsening proxy (flare_next_3d).
2. Sleep Health: sleep disorder present vs none.
3. HRV: stress label from condition; no verified subject/session IDs available.
4. Sri Lankan RA: high vs low month-9 cDAI using baseline predictors only.

This script does NOT validate a combined multimodal RA-flare model.
Preprocessing is fitted inside training folds and the final training set only.
"""

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
RA_FEATURE_COLUMNS = (
    "past_mean_severity",
    "past_std_severity",
    "past_trend",
    "past_last_severity",
    "past_engagement",
    "personal_baseline",
    "age",
    "sex",
)


def fix_duplicate_cols(df):
    """Make duplicate column names unique."""
    cols, seen = [], {}
    for col in df.columns:
        col = str(col).strip()
        if col in seen:
            seen[col] += 1
            cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            cols.append(col)
    out = df.copy()
    out.columns = cols
    return out


def normalize_text(series):
    return series.astype("string").str.strip().str.lower()


def select_ra_features(df):
    """Select only the ordered RA predictors, excluding IDs, dates, and target."""
    missing = [column for column in RA_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"RA dataset is missing predictor columns: {missing}")

    features = df.loc[:, RA_FEATURE_COLUMNS].copy()
    # Keep sex categorical, while converting pandas' nullable missing marker
    # to the NaN representation expected by scikit-learn imputers.
    features["sex"] = features["sex"].astype(object).where(
        features["sex"].notna(), np.nan
    )
    if tuple(features.columns) != RA_FEATURE_COLUMNS:
        raise ValueError("RA feature schema does not match the required order.")
    return features


def make_preprocessor(X, scale_numeric):
    """Numeric median imputation; categorical mode imputation + one-hot encoding."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    transformers = []

    if numeric_cols:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric_cols))

    if categorical_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_candidates(X, y, balanced=False):
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    pos_weight = neg / pos if balanced and pos else 1.0

    return {
        "Logistic Regression": Pipeline([
            ("preprocess", make_preprocessor(X, scale_numeric=True)),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced" if balanced else None,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", make_preprocessor(X, scale_numeric=False)),
            ("model", RandomForestClassifier(
                n_estimators=250, random_state=RANDOM_STATE,
                class_weight="balanced" if balanced else None, n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocess", make_preprocessor(X, scale_numeric=False)),
            ("model", XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                random_state=RANDOM_STATE, eval_metric="logloss",
                verbosity=0, n_jobs=-1, scale_pos_weight=pos_weight,
            )),
        ]),
    }


def choose_split(X, y, groups=None):
    """Create a single held-out test split."""
    if groups is not None and pd.Series(groups).nunique() >= 3:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        return train_idx, test_idx, "grouped holdout by supplied ID"

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return train_idx, test_idx, "stratified random holdout"


def make_cv(X_train, y_train, groups_train=None):
    """Create CV folds with both classes in train and validation portions."""
    splits = []
    if groups_train is not None and pd.Series(groups_train).nunique() >= 3:
        n_splits = min(5, pd.Series(groups_train).nunique())
        iterator = GroupKFold(n_splits=n_splits).split(
            X_train, y_train, groups=groups_train
        )
    else:
        min_class = int(pd.Series(y_train).value_counts().min())
        n_splits = min(5, min_class)
        if n_splits < 2:
            raise ValueError("Too few examples in the minority class for CV.")
        iterator = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        ).split(X_train, y_train)

    for tr_idx, va_idx in iterator:
        if pd.Series(y_train).iloc[tr_idx].nunique() == 2 and \
                pd.Series(y_train).iloc[va_idx].nunique() == 2:
            splits.append((tr_idx, va_idx))
    if not splits:
        raise ValueError("No valid CV folds containing both target classes.")
    return splits


def safe_auc(y_true, scores):
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def evaluate_dataset(X, y, label, save_path, groups=None, balanced=False,
                     target_description=""):
    """Select by training-only CV ROC-AUC, evaluate once on held-out test data."""
    X = X.reset_index(drop=True).copy()
    y = pd.Series(y).reset_index(drop=True).astype(int)

    # Drop only rows with no usable predictors; impute partial missingness in
    # pipelines fitted separately within each fold.
    usable = X.notna().any(axis=1)
    X, y = X.loc[usable].reset_index(drop=True), y.loc[usable].reset_index(drop=True)
    if groups is not None:
        groups = pd.Series(groups).reset_index(drop=True).loc[usable].reset_index(drop=True)

    if len(y) < 20 or y.nunique() != 2:
        raise ValueError(f"{label}: insufficient usable rows or only one class.")
    if int(y.value_counts().min()) < 3:
        raise ValueError(f"{label}: fewer than three examples in a class.")

    train_idx, test_idx, split_note = choose_split(X, y, groups)
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    groups_train = groups.iloc[train_idx] if groups is not None else None

    print("\n" + "=" * 70)
    print(f"Training: {label}")
    print(f"Target: {target_description}")
    print(f"Rows={len(y)} | train={len(train_idx)} | test={len(test_idx)}")
    print(f"Split: {split_note}")
    print(f"Class counts: {y.value_counts().sort_index().to_dict()}")
    print("=" * 70)

    candidates = make_candidates(X_train, y_train, balanced)
    folds = make_cv(X_train, y_train, groups_train)
    cv_scores = {}

    for name, pipeline in candidates.items():
        aucs = []
        for tr_i, va_i in folds:
            fitted = clone(pipeline)
            fitted.fit(X_train.iloc[tr_i], y_train.iloc[tr_i])
            probs = fitted.predict_proba(X_train.iloc[va_i])[:, 1]
            auc = safe_auc(y_train.iloc[va_i], probs)
            if auc is not None:
                aucs.append(auc)
        cv_scores[name] = float(np.mean(aucs)) if aucs else float("-inf")

    best_name = max(cv_scores, key=cv_scores.get)
    if not np.isfinite(cv_scores[best_name]):
        raise ValueError(f"{label}: no valid CV ROC-AUC for model selection.")
    print("Training-only CV ROC-AUC:", {
        k: round(v, 4) if np.isfinite(v) else None for k, v in cv_scores.items()
    })
    print("Selected by CV:", best_name)

    results, fitted_models = {}, {}
    for name, pipeline in candidates.items():
        fitted = clone(pipeline)
        fitted.fit(X_train, y_train)
        pred = fitted.predict(X_test)
        probs = fitted.predict_proba(X_test)[:, 1]
        auc = safe_auc(y_test, probs)
        try:
            pr_auc = float(average_precision_score(y_test, probs))
        except ValueError:
            pr_auc = None

        metrics = {
            "acc": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "auc": auc,
            "pr_auc": pr_auc,
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "test_rows": int(len(y_test)),
            "target": target_description,
            "split": split_note,
        }
        results[name] = metrics
        fitted_models[name] = fitted
        auc_text = f"{auc:.3f}" if auc is not None else "N/A"
        pr_text = f"{pr_auc:.3f}" if pr_auc is not None else "N/A"
        print(f"{name:22s} | Acc={metrics['acc']:.3f} "
              f"| F1w={metrics['f1']:.3f} | ROC-AUC={auc_text} | PR-AUC={pr_text}")

    best_model = fitted_models[best_name]
    best_pred = best_model.predict(X_test)
    print("\nHeld-out classification report for selected model:")
    print(classification_report(y_test, best_pred, zero_division=0))
    print("Held-out confusion matrix:", confusion_matrix(y_test, best_pred).tolist())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    scores = [results[name]["acc"] for name in names]
    colors = ["#4CAF50" if name == best_name else "#B0BEC5" for name in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, scores, color=colors, edgecolor="#555")
    ax.set_title(f"Held-out Test Accuracy — {label}")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)
    for i, score in enumerate(scores):
        ax.text(i, score + 0.015, f"{score:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    safe_label = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in label.replace(" ", "_")
    )
    fig.savefig(OUTPUT_DIR / f"{safe_label}_comparison.png", dpi=140)
    plt.close(fig)

    joblib.dump({
        "model": best_model,
        "cols": X.columns.tolist(),
        "best": best_name,
        "target_description": target_description,
        "selected_by": "training-only cross-validation ROC-AUC",
        "test_split": split_note,
    }, save_path)
    print(f"Saved selected pipeline -> {save_path}")
    return best_model, X.columns.tolist(), results[best_name], results


def main():
    all_results, summary = {}, {}

    # 1. RA clinical dataset: future symptom-worsening proxy.
    print("\n[1/4] RA Clinical symptom-worsening proxy...")
    ra_path = DATA_DIR / "ra_processed_v3.csv"
    ra_df = pd.read_csv(ra_path)
    required = {
        "user_id",
        "checkpoint_date",
        "flare_next_3d",
        *RA_FEATURE_COLUMNS,
    }
    if not required.issubset(ra_df.columns):
        raise ValueError(f"{ra_path} must contain columns {sorted(required)}")
    groups = ra_df["user_id"].copy()
    X = select_ra_features(ra_df)
    y = pd.to_numeric(ra_df["flare_next_3d"], errors="coerce")
    valid = y.notna()
    X, y, groups = (
        X.loc[valid].reset_index(drop=True),
        y.loc[valid].astype(int).reset_index(drop=True),
        groups.loc[valid].reset_index(drop=True),
    )
    _, _, selected, results = evaluate_dataset(
        X, y, "RA_Clinical_v3", MODEL_DIR / "ra_model_v3.pkl",
        groups=groups, balanced=True,
        target_description="dataset-defined future 3-day symptom-worsening proxy (flare_next_3d)",
    )
    summary["RA Clinical"], all_results["RA Clinical"] = selected, results

    # 2. Sleep Health: sleep disorder, not RA flare.
    print("\n[2/4] Sleep Health...")
    sleep_path = DATA_DIR / "sleep.csv"
    sleep_df = pd.read_csv(sleep_path, keep_default_na=False)
    if not {"Sleep Disorder", "Person ID"}.issubset(sleep_df.columns):
        raise ValueError(f"{sleep_path} must contain 'Sleep Disorder' and 'Person ID'")
    labels = normalize_text(sleep_df["Sleep Disorder"])
    y = (labels.notna() & labels.ne("") & labels.ne("none")).astype(int)
    X = sleep_df.drop(columns=["Sleep Disorder", "Person ID"], errors="ignore")
    X = X.replace(r"^\s*$", np.nan, regex=True)
    _, _, selected, results = evaluate_dataset(
        X, y, "Sleep_Health_v2", MODEL_DIR / "sleep_model_v2.pkl",
        target_description="sleep disorder present vs no disorder",
    )
    summary["Sleep Health"], all_results["Sleep Health"] = selected, results

    # 3. HRV: explicit stress label; no verified participant/session ID.
    print("\n[3/4] HRV stress classification...")
    hrv_path = DATA_DIR / "hrv.csv"
    hrv_df = fix_duplicate_cols(pd.read_csv(hrv_path))
    if "condition" not in hrv_df.columns:
        raise ValueError(f"{hrv_path} must contain a 'condition' column")
    cap = int(os.environ.get("HRV_MAX_ROWS", "0"))
    if cap > 0 and cap < len(hrv_df):
        hrv_df = hrv_df.sample(cap, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[WARNING] HRV_MAX_ROWS={cap}: sample run only; do not report as final metrics.")

    condition = normalize_text(hrv_df["condition"])
    y = (condition != "no stress").astype(int)
    X = hrv_df.drop(columns=["condition", "datasetId"], errors="ignore")
    X = X.select_dtypes(include=["number", "bool"]).replace([np.inf, -np.inf], np.nan)
    valid = X.notna().any(axis=1)
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    _, _, selected, results = evaluate_dataset(
        X, y, "HRV_Stress_v2", MODEL_DIR / "hrv_model_v2.pkl",
        target_description="condition != 'no stress'; row-level split, no verified subject/session IDs",
    )
    summary["HRV Stress"], all_results["HRV Stress"] = selected, results

    # 4. Sri Lankan RA: baseline predictors -> high/low month-9 cDAI.
    print("\n[4/4] Sri Lankan RA baseline-only outcome...")
    sl_path = DATA_DIR / "sri_lankan_ra.csv"
    sl_df = pd.read_csv(sl_path, header=1)
    sl_df.columns = sl_df.columns.astype(str).str.strip()
    sl_df = sl_df.dropna(how="all").dropna(axis=1, how="all")

    id_col = "Index No"
    if id_col not in sl_df.columns:
        raise ValueError(f"{sl_path} missing '{id_col}'. Columns: {sl_df.columns.tolist()}")
    ids = sl_df[id_col].astype("string").str.strip()
    valid_id = ids.notna() & ids.ne("")
    sl_df = sl_df.loc[valid_id].copy()

    # CSV header whitespace was stripped above; "Age " is now "Age".
    baseline_features = [
        "Age", "Edu (upto grade)", "Dis Dur.(months)", "TJ", "SJ", "ESR",
        "DAS 28(0)", "cDAI(0)", "Pain-VAS(0)",
    ]
    target_col = "cDAI(9)"
    required_cols = baseline_features + [target_col]
    missing = [col for col in required_cols if col not in sl_df.columns]
    if missing:
        raise ValueError(f"Missing required Sri Lankan RA columns: {missing}")

    for col in required_cols:
        sl_df[col] = pd.to_numeric(sl_df[col], errors="coerce")
    sl_df = sl_df.loc[sl_df[target_col].notna()].copy()

    threshold = float(sl_df[target_col].median())
    y = (sl_df[target_col] > threshold).astype(int).reset_index(drop=True)
    X = sl_df[baseline_features].reset_index(drop=True)
    ids = sl_df[id_col].astype("string").str.strip().reset_index(drop=True)
    groups = ids if ids.duplicated().any() else None

    print(f"Patients with month-9 cDAI: {len(y)}")
    print(f"Baseline predictors: {baseline_features}")
    print(f"Target: cDAI(9) > cohort median ({threshold:.3f})")
    print(f"Class counts: {y.value_counts().sort_index().to_dict()}")
    print("Interpretation: high vs low month-9 cDAI; not a confirmed RA-flare label.")

    _, _, selected, results = evaluate_dataset(
        X, y, "Sri_Lankan_RA_v2", MODEL_DIR / "sl_ra_model_v2.pkl",
        groups=groups,
        target_description=f"cDAI(9) > cohort median ({threshold:.3f}); not confirmed flare",
    )
    summary["Sri Lankan RA"], all_results["Sri Lankan RA"] = selected, results

    # JSON-safe held-out metrics for the UI and thesis audit trail.
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, allow_nan=False)

    print("\n" + "=" * 70)
    print("SUMMARY — separate target tasks, not a fusion-model evaluation")
    print("=" * 70)
    for dataset, metrics in summary.items():
        auc = metrics.get("auc")
        auc_text = f"{auc:.3f}" if auc is not None else "N/A"
        print(f"{dataset:18s} | Acc={metrics['acc']:.3f} "
              f"| F1w={metrics['f1']:.3f} | ROC-AUC={auc_text}")
    print(f"\nSaved metrics -> {MODEL_DIR / 'metrics.json'}")
    print(f"Saved comparison charts -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
