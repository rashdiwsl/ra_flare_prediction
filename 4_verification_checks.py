"""
4_verification_checks.py
========================
Two sanity checks on the results that looked "too good" after the leakage fix.
Run from the project root (same folder as 1_train_models_v2.py):

    python 4_verification_checks.py

CHECK 1 - Sleep (97.3%): does it survive without 'Person ID' and with repeated CV?
CHECK 2 - HRV (100%):    does it survive when whole regions of feature space are
                         held out (a stand-in for "unseen subjects/sessions")?

Why HRV needs a stand-in: data/hrv.csv has no subject ID (datasetId is constant)
and the rows are SHUFFLED (condition changes every ~1.7 rows), so file-order
blocks and chronological splits do NOT separate recording sessions. Clustering
in feature space and holding out whole clusters is the best available proxy.
"""
import time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (GroupShuffleSplit, RepeatedStratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import LabelEncoder, StandardScaler

HRV_SAMPLE = 40000   # rows used for the HRV check (raise for a slower, tighter estimate)


# --------------------------------------------------------------------------
# CHECK 1 - SLEEP
# --------------------------------------------------------------------------
def check_sleep():
    print("\n" + "=" * 70 + "\n CHECK 1 - Sleep Health (374 rows)\n" + "=" * 70)
    # keep_default_na=False keeps the literal string "None" (= no disorder)
    # instead of letting pandas silently turn it into NaN.
    df = pd.read_csv("data/sleep.csv", keep_default_na=False)
    print("  Target classes:", df["Sleep Disorder"].value_counts().to_dict())
    df["t"] = (df["Sleep Disorder"] != "None").astype(int)
    df = df.drop(columns=["Sleep Disorder"])
    for c in df.select_dtypes(include="object").columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    all_cols = [c for c in df.columns if c != "t"]
    no_id = [c for c in all_cols if c != "Person ID"]

    def run(cols, tag, model=None):
        model = model or RandomForestClassifier(n_estimators=150, random_state=42)
        X, y = df[cols], df["t"]
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        s = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        hold = model.fit(Xtr, ytr).score(Xte, yte)
        print(f"  {tag:40s} holdout={hold:.3f}   5x5 CV = {s.mean():.3f} +/- {s.std():.3f}")

    run(all_cols, "1. as in v2 (WITH Person ID)")
    run(no_id, "2. WITHOUT Person ID")
    run(no_id, "3. WITHOUT Person ID (logistic reg)", LogisticRegression(max_iter=2000))
    dup = df[no_id + ["t"]].duplicated().sum()
    print(f"\n  Duplicate rows (ignoring Person ID): {dup} of {len(df)}")
    print("  -> report the 5x5 CV figure with +/- std, not the single 97.3% holdout.")


# --------------------------------------------------------------------------
# CHECK 2 - HRV
# --------------------------------------------------------------------------
def check_hrv():
    print("\n" + "=" * 70 + f"\n CHECK 2 - HRV Stress ({HRV_SAMPLE:,}-row sample)\n" + "=" * 70)
    h = pd.read_csv("data/hrv.csv")
    y = (h["condition"].str.strip().str.lower() != "no stress").astype(int)
    X = h.drop(columns=["condition", "datasetId"]).select_dtypes(include=[np.number])
    ok = ~X.isna().any(axis=1)
    X, y = X[ok].reset_index(drop=True), y[ok].reset_index(drop=True)

    keep = np.sort(np.random.default_rng(0).choice(len(X), HRV_SAMPLE, replace=False))
    X, y = X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)
    idx = np.arange(len(X))

    def score(tr, te, tag):
        t = time.time()
        m = RandomForestClassifier(n_estimators=40, n_jobs=-1, random_state=42).fit(X.iloc[tr], y.iloc[tr])
        acc = accuracy_score(y.iloc[te], m.predict(X.iloc[te]))
        auc = roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1])
        print(f"  {tag:46s} acc={acc:.4f}  auc={auc:.4f}  ({time.time()-t:.0f}s)")

    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    score(tr, te, "A. plain random split")
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, idx // 300))
    score(tr, te, "B. v2 300-row block groups")
    cut = int(0.8 * len(X))
    score(idx[:cut], idx[cut:], "C. first 80% / last 20% (file order)")

    Z = StandardScaler().fit_transform(X)
    for k in (50, 200):
        lab = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3).fit_predict(Z)
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, lab))
        score(tr, te, f"D. held-out feature clusters (k={k})")

    print("\n  Single-feature baselines (random split):")
    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    for f in ["HR", "MEAN_RR", "RMSSD", "SDRR"]:
        m = RandomForestClassifier(n_estimators=40, n_jobs=-1, random_state=42).fit(X.iloc[tr][[f]], y.iloc[tr])
        print(f"    {f:8s} acc={accuracy_score(y.iloc[te], m.predict(X.iloc[te][[f]])):.4f}")
    print("\n  -> A, B and C are ~100%: the file is shuffled, so none of them separates")
    print("     sessions. D (held-out clusters) is the more honest, lower estimate.")


if __name__ == "__main__":
    check_sleep()
    check_hrv()
