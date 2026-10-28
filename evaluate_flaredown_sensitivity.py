
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

warnings.filterwarnings("ignore")

DATA = "data/ra_processed_v3_sensitivity.csv"
TARGETS = [
    "flare_next_3d_min1",
    "flare_next_3d_min2",
    "flare_next_3d_min3",
]

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

N_SPLITS = 5
N_REPEATS = 6
RANDOM_STATE = 42

df = pd.read_csv(DATA)

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
}

all_results = []

for target in TARGETS:
    data = df.dropna(subset=[target]).copy()
    data[target] = data[target].astype(int)

    X = data[FEATURES]
    y = data[target]
    groups = data["user_id"]

    print("\n" + "=" * 75)
    print(f"TARGET: {target}")
    print(f"Rows: {len(data)}")
    print(f"Patients: {groups.nunique()}")
    print(f"Positive rate: {y.mean():.4f}")
    print(f"Class counts: {y.value_counts().to_dict()}")

    if y.nunique() != 2:
        print("SKIPPED: target does not contain both classes.")
        continue

    for model_name, model in models.items():
        fold_results = []

        for repeat in range(N_REPEATS):
            splitter = StratifiedGroupKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=RANDOM_STATE + repeat,
            )

            for train_idx, test_idx in splitter.split(X, y, groups):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                # Grouped folds must contain both classes for ROC-AUC.
                if y_train.nunique() != 2 or y_test.nunique() != 2:
                    continue

                fitted = clone(model)
                fitted.fit(X_train, y_train)

                probabilities = fitted.predict_proba(X_test)[:, 1]
                predictions = (probabilities >= 0.5).astype(int)

                fold_results.append({
                    "roc_auc": roc_auc_score(y_test, probabilities),
                    "pr_auc": average_precision_score(
                        y_test, probabilities
                    ),
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

        if not fold_results:
            print(f"\n{model_name}: no valid evaluation folds.")
            continue

        scores = pd.DataFrame(fold_results)

        result = {
            "target": target,
            "model": model_name,
            "rows": len(data),
            "patients": groups.nunique(),
            "positive_rate": y.mean(),
            "valid_folds": len(scores),
        }

        for metric in [
            "roc_auc",
            "pr_auc",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
        ]:
            result[f"{metric}_mean"] = scores[metric].mean()
            result[f"{metric}_std"] = scores[metric].std()

        all_results.append(result)

        print(f"\n{model_name} ({len(scores)} valid folds)")
        for metric in [
            "roc_auc",
            "pr_auc",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
        ]:
            print(
                f"  {metric}: "
                f"{scores[metric].mean():.4f} "
                f"+/- {scores[metric].std():.4f}"
            )

if all_results:
    results = pd.DataFrame(all_results)
    output = "outputs/flaredown_sensitivity_results.csv"
    results.to_csv(output, index=False)

    print("\n" + "=" * 75)
    print("SENSITIVITY SUMMARY")
    print(results[
        [
            "target",
            "model",
            "rows",
            "patients",
            "positive_rate",
            "valid_folds",
            "roc_auc_mean",
            "pr_auc_mean",
            "f1_mean",
        ]
    ].to_string(index=False))

    print(f"\nResults saved to: {output}")
else:
    print("\nNo results produced. Check the target labels and class counts.")
