import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import accuracy_score, f1_score, classification_report
from xgboost                 import XGBClassifier

os.makedirs('models',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ─── Utilities ────────────────────────────────────────────────────────────────
def encode_all(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def fix_duplicate_cols(df):
    cols = []
    seen = {}
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

def auto_prepare(df):
    df = fix_duplicate_cols(df)
    df = encode_all(df)
    df = df.select_dtypes(include=[np.number]).dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    if y.nunique() > 2:
        y = (y > y.median()).astype(int)
    print(f"  Classes: {y.value_counts().to_dict()}")
    return X, y

def compare_and_save(X, y, label, save_path):
    X = fix_duplicate_cols(X.copy())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42),
        "XGBoost":             XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
    }
    results = {}
    print(f"\n{'='*55}\n  Training: {label}\n{'='*55}")
    for name, m in candidates.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average='weighted')
        results[name] = {"acc": acc, "f1": f1, "model": m}
        print(f"  {name:22s} | Acc {acc:.3f} | F1 {f1:.3f}")

    best_name  = max(results, key=lambda k: results[k]["acc"])
    best_model = results[best_name]["model"]
    print(f"\n  WINNER: {best_name}")
    print(classification_report(y_test, best_model.predict(X_test), zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#4CAF50" if n == best_name else "#B0BEC5" for n in results]
    ax.bar(results.keys(), [r["acc"] for r in results.values()], color=colors, edgecolor="#555")
    ax.set_title(f"Model Comparison — {label}", fontsize=12)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.2)
    for i, r in enumerate(results.values()):
        ax.text(i, r["acc"] + 0.03, f"{r['acc']:.3f}", ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(f"outputs/{label.replace(' ','_')}_comparison.png", dpi=120)
    plt.show()

    joblib.dump({"model": best_model, "cols": X.columns.tolist(), "best": best_name}, save_path)
    print(f"  ✅ Saved → {save_path}")
    return best_model, X.columns.tolist()

# ─── 1. RA Clinical ──────────────────────────────────────────────────────────
print("\n[1/4] Loading RA Clinical...")
ra_df = pd.read_csv('data/ra_processed.csv')
ra_X, ra_y = auto_prepare(ra_df)
ra_model, ra_cols = compare_and_save(ra_X, ra_y, "RA Clinical", "models/ra_model.pkl")

# ─── 2. Sleep ────────────────────────────────────────────────────────────────
print("\n[2/4] Loading Sleep...")
sl_df = pd.read_csv('data/sleep.csv')

# NaN = no disorder = 0, Sleep Apnea/Insomnia = 1
sl_df['sleep_target'] = sl_df['Sleep Disorder'].notna().astype(int)
sl_df = sl_df.drop(columns=['Sleep Disorder'])

sl_enc = encode_all(sl_df).select_dtypes(include=[np.number]).dropna()
sl_X   = sl_enc.drop(columns=['sleep_target'])
sl_y   = sl_enc['sleep_target']
print(f"  Classes: {sl_y.value_counts().to_dict()}")
sleep_model, sleep_cols = compare_and_save(sl_X, sl_y, "Sleep Health", "models/sleep_model.pkl")

# ─── 3. HRV ──────────────────────────────────────────────────────────────────
print("\n[3/4] Loading HRV...")
hrv_df = pd.read_csv('data/hrv.csv')
print(f"  HRV shape: {hrv_df.shape}")
hrv_X, hrv_y = auto_prepare(hrv_df)
hrv_model, hrv_cols = compare_and_save(hrv_X, hrv_y, "HRV Stress", "models/hrv_model.pkl")

# ─── 4. Sri Lankan RA ────────────────────────────────────────────────────────
print("\n[4/4] Loading Sri Lankan RA...")

sl_ra_raw = pd.read_csv('data/sri_lankan_ra.csv', header=None)

# Row 0 = junk title, Row 1 = real headers, data from Row 2
sl_ra_df = sl_ra_raw.iloc[2:].copy()
sl_ra_df.columns = sl_ra_raw.iloc[1].values
sl_ra_df = sl_ra_df.reset_index(drop=True)

# Convert all to numeric, drop mostly-empty columns and NaN rows
sl_ra_df = sl_ra_df.apply(pd.to_numeric, errors='coerce')
sl_ra_df = sl_ra_df.dropna(axis=1, thresh=50)
sl_ra_df = sl_ra_df.dropna()

print(f"  After cleaning — shape: {sl_ra_df.shape}")
print(f"  Columns: {sl_ra_df.columns.tolist()}")

if sl_ra_df.shape[0] < 10:
    print("  ⚠ Too few rows — skipping SL-RA model")
else:
    # Fix duplicate column names (two ESR columns)
    cols = []
    seen = {}
    for c in sl_ra_df.columns:
        c = str(c)
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    sl_ra_df.columns = cols

    X = sl_ra_df.iloc[:, :-1]
    y = sl_ra_df.iloc[:, -1]
    if y.nunique() > 2:
        y = (y > y.median()).astype(int)
    print(f"  Classes: {y.value_counts().to_dict()}")
    sl_ra_model, sl_ra_cols = compare_and_save(X, y, "SriLankan RA", "models/sl_ra_model.pkl")

print("\n✅ ALL MODELS TRAINED AND SAVED")
print("  models/ra_model.pkl      ✅")
print("  models/sleep_model.pkl   ✅")
print("  models/hrv_model.pkl     ✅")
print("  models/sl_ra_model.pkl   ✅")