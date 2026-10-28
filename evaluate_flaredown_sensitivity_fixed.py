from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ra_processed_v3_candidate.csv"
OUTPUT = ROOT / "outputs" / "flaredown_sensitivity_fixed_results.csv"

FEATURES = [
    "past_mean_severity",
    "past_std_severity",
    "past_trend",
    "past_last_severity",
    "past_engagement",
    "personal_baseline",
    "age",
    "sex",
]
TARGET = "flare_next_3d"
GROUP = "user_id"

N_SPLITS = 5
N_REPEATS = 6
RANDOM_STATE = 42

def make_preprocessor(scale_numeric):
    numeric = FEATURES[:-1]
    categorical = ["sex"]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {OUTPUT}"
        )

    df = pd.read_csv(DATA)

    required = FEATURES + [TARGET, GROUP]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TARGET, GROUP]).copy()
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    X = df[FEATURES].copy()
    y = df[TARGET]
    groups = df[GROUP]

    if X["sex"].dropna().empty:
        raise ValueError("No non-missing sex values available.")

    models = {
        "Logistic Regression": Pipeline([
            ("preprocess", make_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", make_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

    rows = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )

        for model_name, model in models.items():
            for fold, (train_idx, test_idx) in enumerate(
                splitter.split(X, y, groups), start=1
            ):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                train_groups = set(groups.iloc[train_idx])
                test_groups = set(groups.iloc[test_idx])

                if train_groups & test_groups:
                    raise RuntimeError("Patient overlap detected.")
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    print(
                        f"Skipping {model_name}, repeat {repeat + 1}, "
                        f"fold {fold}: only one target class."
                    )
                    continue

                fitted = clone(model)
                fitted.fit(X_train, y_train)
                probabilities = fitted.predict_proba(X_test)[:, 1]
                predictions = (probabilities >= 0.5).astype(int)

                rows.append({
                    "model": model_name,
                    "repeat": repeat + 1,
                    "fold": fold,
                    "train_patients": len(train_groups),
                    "test_patients": len(test_groups),
                    "rows": len(test_idx),
                    "positive_rate": float(y_test.mean()),
                    "roc_auc": roc_auc_score(y_test, probabilities),
                    "pr_auc": average_precision_score(y_test, probabilities),
                    "balanced_accuracy": balanced_accuracy_score(
                        y_test, predictions
                    ),
                    "precision": precision_score(
                        y_test, predictions, zero_division=0
                    ),
                    "recall": recall_score(
                        y_test, predictions, zero_division=0
                    ),
                    "f1": f1_score(
                        y_test, predictions, zero_division=0
                    ),
                })

    if not rows:
        raise RuntimeError("No valid evaluation folds were produced.")

    results = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    print("\n=== CORRECTED SENSITIVITY EVALUATION ===")
    print("Dataset:", DATA)
    print("Rows:", len(df))
    print("Unique patients:", groups.nunique())
    print("Positive rate:", round(float(y.mean()), 4))
    print("Valid fold evaluations:", len(results))
    print("Results saved to:", OUTPUT)

    metrics = [
        "roc_auc", "pr_auc", "balanced_accuracy",
        "precision", "recall", "f1"
    ]
    print("\nMean and standard deviation across folds:")
    print(results.groupby("model")[metrics].agg(["mean", "std"]).round(4))

    print(
        "\nInterpretation: folds share patients across repeated repeats "
        "as a whole experiment, but each individual fold keeps its "
        "patients separated between training and test. Repeated folds "
        "are not independent replications. The target remains a "
        "dataset-defined symptom-worsening proxy, not a confirmed "
        "clinical RA flare."
    )

if __name__ == "__main__":
    main()
