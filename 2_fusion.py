import pandas as pd
import numpy as np
import joblib, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model  import LogisticRegression
from sklearn.metrics       import accuracy_score
from sklearn.preprocessing import LabelEncoder

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

# ── Load each dataset the same way it was loaded during training ──────────────

# 1. RA
ra_df  = encode_all(pd.read_csv('data/ra_processed.csv')).select_dtypes(include=[np.number]).dropna()
ra_X   = ra_df.iloc[:, :-1]
ra_y   = ra_df.iloc[:, -1]

# 2. Sleep — must handle NaN target same way as training
sl_df  = pd.read_csv('data/sleep.csv')
sl_df['sleep_target'] = sl_df['Sleep Disorder'].notna().astype(int)
sl_df  = sl_df.drop(columns=['Sleep Disorder'])
sl_enc = encode_all(sl_df).select_dtypes(include=[np.number]).dropna()
sl_X   = sl_enc.drop(columns=['sleep_target'])
sl_y   = sl_enc['sleep_target']

# 3. HRV
hrv_df = fix_duplicate_cols(encode_all(pd.read_csv('data/hrv.csv'))).select_dtypes(include=[np.number]).dropna()
hrv_X  = hrv_df.iloc[:, :-1]
hrv_y  = hrv_df.iloc[:, -1]

# 4. Sri Lankan RA — same multi-header handling as training
sl_raw    = pd.read_csv('data/sri_lankan_ra.csv', header=None)
sll_df    = sl_raw.iloc[2:].copy()
sll_df.columns = sl_raw.iloc[1].values
sll_df    = sll_df.reset_index(drop=True)
sll_df    = sll_df.apply(pd.to_numeric, errors='coerce')
sll_df    = sll_df.dropna(axis=1, thresh=50).dropna()
sll_df    = fix_duplicate_cols(sll_df)
sll_X     = sll_df.iloc[:, :-1]
sll_y     = sll_df.iloc[:, -1]

print(f"Sizes — RA:{len(ra_X)}  Sleep:{len(sl_X)}  HRV:{len(hrv_X)}  SL-RA:{len(sll_X)}")

# ── Load models ───────────────────────────────────────────────────────────────
ra_model    = joblib.load("models/ra_model.pkl")["model"]
sleep_model = joblib.load("models/sleep_model.pkl")["model"]
hrv_model   = joblib.load("models/hrv_model.pkl")["model"]
sl_ra_model = joblib.load("models/sl_ra_model.pkl")["model"]

def safe_prob(model, X):
    n   = model.n_features_in_
    arr = np.array(X.values, dtype=float)
    if arr.shape[1] < n:
        arr = np.pad(arr, ((0,0),(0, n - arr.shape[1])))
    else:
        arr = arr[:, :n]
    return model.predict_proba(arr)[:, 1]

# ── Fuse ──────────────────────────────────────────────────────────────────────
n = min(len(ra_X), len(sl_X), len(hrv_X), len(sll_X))
print(f"Using {n} samples for fusion")

p1 = safe_prob(ra_model,    ra_X.iloc[:n])
p2 = safe_prob(sleep_model, sl_X.iloc[:n])
p3 = safe_prob(hrv_model,   hrv_X.iloc[:n])
p4 = safe_prob(sl_ra_model, sll_X.iloc[:n])

fusion_X = np.column_stack([p1, p2, p3, p4])
fusion_y = (ra_y.iloc[:n] > ra_y.iloc[:n].median()).astype(int).values

meta = LogisticRegression()
meta.fit(fusion_X, fusion_y)
acc = accuracy_score(fusion_y, meta.predict(fusion_X))
print(f"\n✅ Fusion Meta-Classifier Accuracy: {acc:.3f}")

joblib.dump(meta, "models/meta_model.pkl")
print("✅ Meta model saved → models/meta_model.pkl")
print("\nAll done! Now run: python 3_patient_console.py")