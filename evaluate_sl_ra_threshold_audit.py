from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "sri_lankan_ra.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 42
TEST_SIZE = 0.20

FEATURES = [
    "Age",
    "Edu (upto grade)",
    "Dis Dur.(months)",
    "TJ",
    "SJ",
    "ESR",
    "DAS 28(0)",
    "cDAI(0)",
    "Pain-VAS(0)",
]
TARGET = "cDAI(9)"

# Read and clean the dataset without modifying the original file.
df = pd.read_csv(DATA, header=1)
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all").dropna(axis=1, how="all")

required = ["Index No"] + FEATURES + [TARGET]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Index No"] = df["Index No"].astype("string").str.strip()
df = df.loc[
    df["Index No"].notna()
    & df["Index No"].ne("")
].copy()

for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Each row should represent a unique patient for this evaluation.
if df["Index No"].duplicated().any():
    duplicates = df.loc[
        df["Index No"].duplicated(keep=False), "Index No"
    ].tolist()
    raise ValueError(
        "Duplicate patient IDs found; resolve them before evaluation: "
        + str(duplicates)
    )

# Only patients with a month-9 outcome can be evaluated.
df = df.loc[df[TARGET].notna()].reset_index(drop=True)

# Split rows BEFORE defining the outcome threshold.
# The split does not use month-9 outcomes for stratification.
train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=TEST_SIZE,
    random_state=SEED,
    shuffle=True,
)

train = df.iloc[train_idx].copy().reset_index(drop=True)
test = df.iloc[test_idx].copy().reset_index(drop=True)

# Threshold is calculated from training outcomes ONLY.
threshold = float(train[TARGET].median())
y_train = (train[TARGET] > threshold).astype(int)
y_test = (test[TARGET] > threshold).astype(int)

X_train = train[FEATURES].copy()
X_test = test[FEATURES].copy()

# Drop rows with no usable predictors, keeping labels aligned.
train_usable = X_train.notna().any(axis=1)
test_usable = X_test.notna().any(axis=1)

X_train = X_train.loc[train_usable].reset_index(drop=True)
y_train = y_train.loc[train_usable].reset_index(drop=True)
X_test = X_test.loc[test_usable].reset_index(drop=True)
y_test = y_test.loc[test_usable].reset_index(drop=True)

print("=" * 68)
print("SRI LANKAN RA — TRAINING-THRESHOLD AUDIT")
print("=" * 68)
print("Target: month-9 cDAI greater than training-set median")
print(f"Training-derived threshold: {threshold:.3f}")
print(f"Training rows: {len(y_train)}")
print(f"Test rows: {len(y_test)}")
print(f"Training classes: {y_train.value_counts().sort_index().to_dict()}")
print(f"Test classes: {y_test.value_counts().sort_index().to_dict()}")
print("Split: random 80/20 split, seed 42; split before threshold creation")
print("Preprocessing: fitted within each model pipeline on training data")

if len(y_train) < 20 or y_train.nunique() != 2:
    raise ValueError(
        "Training set is too small or contains only one class. "
        "Do not report model metrics."
    )
if len(y_test) < 5 or y_test.nunique() != 2:
    raise ValueError(
        "Test set is too small or contains only one class. "
        "A reliable ROC-AUC cannot be calculated for this split. "
        "Report the split limitation; do not silently change the seed."
    )

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": name,
        "target": "cDAI(9) > training-set median",
        "threshold": threshold,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_positive": int(y_train.sum()),
        "train_negative": int((y_train == 0).sum()),
        "test_positive": int(y_test.sum()),
        "test_negative": int((y_test == 0).sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
    }
    results.append(metrics)

    print(f"\n{name}")
    for key in [
        "accuracy", "balanced_accuracy", "precision",
        "recall", "f1", "roc_auc", "pr_auc"
    ]:
        print(f"  {key}: {metrics[key]:.4f}")

csv_path = OUT / "sl_ra_threshold_audit_results.csv"
json_path = OUT / "sl_ra_threshold_audit_results.json"

pd.DataFrame(results).to_csv(csv_path, index=False)
with json_path.open("w", encoding="utf-8") as f:
    json.dump(
        {
            "purpose": "Exploratory target-threshold leakage audit",
            "target": "month-9 cDAI greater than training-set median",
            "clinical_flare_label": False,
            "threshold": threshold,
            "split_seed": SEED,
            "test_size": TEST_SIZE,
            "threshold_source": "training outcomes only",
            "results": results,
            "limitations": [
                "Small cohort with missing month-9 outcomes.",
                "Single random holdout; results may vary with the split.",
                "Target is relative high/low month-9 cDAI, not confirmed RA flare.",
                "Training-median threshold is exploratory, not a clinical cutoff.",
                "No external validation.",
            ],
        },
        f,
        indent=2,
        allow_nan=False,
    )

print("\nSaved separate audit outputs:")
print(csv_path)
print(json_path)
print("\nOriginal training script and models/metrics.json were not changed.")
