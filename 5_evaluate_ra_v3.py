"""
5_evaluate_ra_v3.py
===================
Honest evaluation of the RA 3-day flare model on data/ra_processed_v3.csv.

 * Patient-level splits only (StratifiedGroupKFold), repeated with 6 different shuffles
 * ROC-AUC and PR-AUC (threshold-free) + recall/precision at 0.5 with class_weight='balanced'
 * A NAIVE baseline that uses no machine learning: the ratio (past 7-day mean / own baseline).
   A model is only worth reporting if it beats this.
 * Results are mean +/- std over folds, not a single lucky split.
"""
import json, os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv("data/ra_processed_v3.csv")
FEATS = ["past_mean_severity", "past_std_severity", "past_trend", "past_last_severity",
         "past_engagement", "personal_baseline", "age", "sex"]
X, y, g = df[FEATS].values, df["flare_next_3d"].values, df["user_id"].values
print(f"{len(df)} rows | {df.user_id.nunique()} true-RA patients | flare rate {y.mean():.3f}")
print(f"(PR-AUC of a coin flip = flare rate = {y.mean():.3f})\n")


def make():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced", random_state=0),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 eval_metric="logloss", verbosity=0, random_state=0),
    }


res = {k: [] for k in make()}
for rep in range(6):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g):
        for name, m in make().items():
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            hard = (p >= 0.5).astype(int)
            res[name].append([roc_auc_score(y[te], p), average_precision_score(y[te], p),
                              balanced_accuracy_score(y[te], hard), recall_score(y[te], hard),
                              precision_score(y[te], hard, zero_division=0)])

names = ["ROC-AUC", "PR-AUC", "BalancedAcc", "Recall", "Precision"]
summary = {}
print(f"{'model':22s}" + "".join(f"{n:>16s}" for n in names))
for k, v in res.items():
    a = np.array(v)
    summary[k] = {n: {"mean": float(a[:, i].mean()), "std": float(a[:, i].std())} for i, n in enumerate(names)}
    print(f"{k:22s}" + "".join(f"{a[:, i].mean():>9.3f}±{a[:, i].std():.3f}" for i in range(len(names))))

score = df["past_mean_severity"] / df["personal_baseline"].replace(0, np.nan)
ok = score.notna()
naive = {"ROC-AUC": float(roc_auc_score(y[ok], score[ok])), "PR-AUC": float(average_precision_score(y[ok], score[ok]))}
print(f"\nNaive rule, no ML (past 7-day mean / own baseline): ROC-AUC {naive['ROC-AUC']:.3f} | PR-AUC {naive['PR-AUC']:.3f}")
print("-> report a model only as far as it beats this line.")

os.makedirs("outputs", exist_ok=True)
json.dump({"n_rows": int(len(df)), "n_patients": int(df.user_id.nunique()), "flare_rate": float(y.mean()),
           "models": summary, "naive_baseline": naive}, open("outputs/ra_v3_results.json", "w"), indent=2)
print("Saved outputs/ra_v3_results.json")
