import pandas as pd
import numpy as np

print("Loading Flaredown dataset...")
df = pd.read_csv('data/ra.csv', low_memory=False)

print(f"Raw shape: {df.shape}")
print(df.columns.tolist())
print(df.head(3))

# ── Step 1: Keep only RA-related rows ────────────────────────────────────────
# Filter conditions that contain RA
ra_conditions = df[
    (df['trackable_type'] == 'Condition') &
    (df['trackable_name'].str.contains(
        'rheumatoid|arthritis|RA', case=False, na=False))
]
ra_user_ids = ra_conditions['user_id'].unique()
print(f"\nRA patients found: {len(ra_user_ids)}")

# Keep only those users
df_ra = df[df['user_id'].isin(ra_user_ids)].copy()

# ── Step 2: Pivot symptoms into columns ───────────────────────────────────────
symptoms = df_ra[df_ra['trackable_type'] == 'Symptom'].copy()
symptoms['trackable_value'] = pd.to_numeric(
    symptoms['trackable_value'], errors='coerce')

# Pivot: one row per user, columns = symptom names, values = mean severity
pivot = symptoms.pivot_table(
    index='user_id',
    columns='trackable_name',
    values='trackable_value',
    aggfunc='mean'
)

# ── Step 3: Keep only the most common symptoms (top 20) ──────────────────────
top_symptoms = pivot.isnull().sum().sort_values().head(20).index
pivot = pivot[top_symptoms].copy()

# ── Step 4: Add user profile info ─────────────────────────────────────────────
profile = df_ra[['user_id', 'age', 'sex']].drop_duplicates('user_id').set_index('user_id')
profile['sex'] = (profile['sex'] == 'female').astype(int)
profile['age'] = pd.to_numeric(profile['age'], errors='coerce')

final = pivot.join(profile, how='left')

# ── Step 5: Create target — high vs low flare ─────────────────────────────────
# Use overall mean symptom severity as target proxy
final['flare_risk'] = (final[top_symptoms].mean(axis=1) > 
                        final[top_symptoms].mean(axis=1).median()).astype(int)

final = final.dropna(subset=['flare_risk'])
final = final.fillna(final.median(numeric_only=True))

print(f"\n✅ Processed shape: {final.shape}")
print(f"   Flare=1: {final['flare_risk'].sum()}  |  Flare=0: {(final['flare_risk']==0).sum()}")
print(f"   Columns: {final.columns.tolist()}")

final.to_csv('data/ra_processed.csv', index=False)
print("\n✅ Saved → data/ra_processed.csv")
print("Now update your ra.csv reference in 1_train_models.py to ra_processed.csv")