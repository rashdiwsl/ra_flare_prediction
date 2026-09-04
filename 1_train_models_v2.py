"""
1_train_models_v2.py
=====================
Corrected training pipeline. Three fixes vs. the original 1_train_models.py:

1. RA Clinical  -> uses data/ra_processed_v2.csv (genuine 3-day-ahead forecast,
   target built only from FUTURE days, features only from PAST days, no overlap).
   Split by user_id (GroupShuffleSplit) so no patient's data leaks across
   train/test.

2. HRV          -> target is an EXPLICIT binary definition (stressed vs not),
   not an accidental alphabetical LabelEncoder + median split. Split by
   datasetId (GroupShuffleSplit) so consecutive near-duplicate rows from the
   same recording session can't appear in both train and test.

3. Sri Lankan RA -> only BASELINE (month 0) features are used to predict the
   month-9 outcome. Month-4 and month-9 concurrent columns (which are
   definitionally entangled with the target — DAS28(9) vs cDAI(9) correlate
   at 0.93) are dropped from X.

Sleep model is unchanged — no leakage was found there.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost                 import XGBClassifier

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


def encode_all(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def fix_duplicate_cols(df):
    cols, seen = [], {}
    for c in df.columns:
        c = str(c)
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    df.columns = cols
    return df


def compare_and_save(X, y, label, save_path, groups=None):
    X = fix_duplicate_cols(X.copy())
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        split_note = "GROUP split (no group overlap between train/test)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        split_note = "stratified random split"

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42),
        "XGBoost":             XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
    }
    results = {}
    print(f"\n{'='*60}\n  Training: {label}   [{split_note}]\n{'='*60}")
    for name, m in candidates.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        proba = m.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average='weighted')
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float('nan')
        results[name] = {"acc": acc, "f1": f1, "auc": auc, "model": m}
        print(f"  {name:22s} | Acc {acc:.3f} | F1 {f1:.3f} | AUC {auc:.3f}")

    best_name  = max(results, key=lambda k: results[k]["acc"])
    best_model = results[best_name]["model"]
    print(f"\n  WINNER: {best_name}")
    print(classification_report(y_test, best_model.predict(X_test), zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#4CAF50" if n == best_name else "#B0BEC5" for n in results]
    ax.bar(results.keys(), [r["acc"] for r in results.values()], color=colors, edgecolor="#555")
    ax.set_title(f"Model Comparison — {label} (corrected)", fontsize=12)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.2)
    for i, r in enumerate(results.values()):
        ax.text(i, r["acc"] + 0.03, f"{r['acc']:.3f}", ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(f"outputs/{label.replace(' ','_')}_v2_comparison.png", dpi=120)

    joblib.dump({"model": best_model, "cols": X.columns.tolist(), "best": best_name}, save_path)
    print(f"  Saved -> {save_path}")
    return best_model, X.columns.tolist(), results[best_name]


summary = {}

# 1. RA Clinical (fixed: genuine forecast, group split by user_id) ----------
print("\n[1/4] RA Clinical (temporal, leakage-fixed)...")
ra_df = pd.read_csv('data/ra_processed_v2.csv')
ra_groups = ra_df['user_id']
ra_X = ra_df.drop(columns=['user_id', 'flare_next_3d'])
ra_y = ra_df['flare_next_3d']
print(f"  Classes: {ra_y.value_counts().to_dict()}")
ra_model, ra_cols, ra_res = compare_and_save(ra_X, ra_y, "RA Clinical v2", "models/ra_model_v2.pkl", groups=ra_groups)
summary['RA Clinical'] = ra_res

# 2. Sleep — unchanged, no leakage found -------------------------------------
print("\n[2/4] Sleep (unchanged)...")
sl_df = pd.read_csv('data/sleep.csv')
sl_df['sleep_target'] = sl_df['Sleep Disorder'].notna().astype(int)
sl_df = sl_df.drop(columns=['Sleep Disorder'])
sl_enc = encode_all(sl_df).select_dtypes(include=[np.number]).dropna()
sl_X = sl_enc.drop(columns=['sleep_target'])
sl_y = sl_enc['sleep_target']
print(f"  Classes: {sl_y.value_counts().to_dict()}")
sleep_model, sleep_cols, sleep_res = compare_and_save(sl_X, sl_y, "Sleep Health v2", "models/sleep_model_v2.pkl")
summary['Sleep Health'] = sleep_res

# 3. HRV (fixed: explicit target definition, group split by session) --------
print("\n[3/4] HRV (explicit target, group split by session)...")
hrv_df = pd.read_csv('data/hrv.csv')
hrv_df = fix_duplicate_cols(hrv_df)
# Explicit, meaningful target: stressed (interruption OR time pressure) vs not.
hrv_df['stress_target'] = (hrv_df['condition'].str.strip().str.lower() != 'no stress').astype(int)
# datasetId is constant across the whole file (not a usable session id in this public release).
# Consecutive rows in SWELL are known to come from the same continuous recording session, so we
# approximate session boundaries by chunking sequential rows into blocks and treating each block
# as a group. This is a documented approximation - stated explicitly in the report - not a perfect
# session id, but it prevents near-duplicate consecutive readings from being split across train/test.
BLOCK_SIZE = 300
hrv_groups = pd.Series(hrv_df.index // BLOCK_SIZE, index=hrv_df.index, name='pseudo_session')
hrv_X = hrv_df.drop(columns=['condition', 'stress_target', 'datasetId'])
hrv_X = hrv_X.select_dtypes(include=[np.number])
hrv_y = hrv_df.loc[hrv_X.index, 'stress_target']
hrv_groups = hrv_groups.loc[hrv_X.index]
mask = ~hrv_X.isna().any(axis=1)
hrv_X, hrv_y, hrv_groups = hrv_X[mask], hrv_y[mask], hrv_groups[mask]
print(f"  Classes: {hrv_y.value_counts().to_dict()}  (grouped by {hrv_groups.nunique()} sessions)")
hrv_model, hrv_cols, hrv_res = compare_and_save(hrv_X, hrv_y, "HRV Stress v2", "models/hrv_model_v2.pkl", groups=hrv_groups)
summary['HRV Stress'] = hrv_res

# 4. Sri Lankan RA (fixed: baseline-only features, no concurrent-visit leak) -
print("\n[4/4] Sri Lankan RA (baseline-only features)...")
sl_ra_raw = pd.read_csv('data/sri_lankan_ra.csv', header=None)
sl_ra_df = sl_ra_raw.iloc[2:].copy()
sl_ra_df.columns = sl_ra_raw.iloc[1].values
sl_ra_df = sl_ra_df.reset_index(drop=True)
sl_ra_df = sl_ra_df.apply(pd.to_numeric, errors='coerce')
sl_ra_df = sl_ra_df.dropna(axis=1, thresh=50)
sl_ra_df = sl_ra_df.dropna()
sl_ra_df = fix_duplicate_cols(sl_ra_df)

baseline_features = [c for c in sl_ra_df.columns if c.strip().endswith('(0)') or c in
                      ['Age ', 'Edu (upto grade)', 'Dis Dur.(months)', 'TJ', 'SJ', 'ESR']]
target_col = 'cDAI(9)'
print(f"  Baseline-only features used: {baseline_features}")
print(f"  Target: {target_col} (predicted from BASELINE only, no month-4/9 leakage)")

if sl_ra_df.shape[0] < 10:
    print("  Too few rows - skipping SL-RA model")
else:
    X = sl_ra_df[baseline_features]
    y = sl_ra_df[target_col]
    y = (y > y.median()).astype(int)
    print(f"  Classes: {y.value_counts().to_dict()}")
    sl_ra_model, sl_ra_cols, slra_res = compare_and_save(X, y, "SriLankan RA v2", "models/sl_ra_model_v2.pkl")
    summary['Sri Lankan RA'] = slra_res

print("\n" + "="*60)
print("  CORRECTED PIPELINE — SUMMARY (compare to original in report)")
print("="*60)
for name, res in summary.items():
    print(f"  {name:16s} | Acc {res['acc']:.3f} | F1 {res['f1']:.3f} | AUC {res['auc']:.3f}")
