"""
prepare_flaredown_v2.py
========================
Fixes the target-leakage problem in the original prepare_flaredown.py.

OLD APPROACH (leakage):
    flare_risk = (mean of symptom columns TODAY > population median of that same mean)
    -> target is computed directly from the same columns used as features.

NEW APPROACH (genuine forecast):
    For each RA patient with a long enough check-in history, build one row
    per "checkpoint day" using ONLY features from the past (days t-6 .. t),
    and label it using severity from the FUTURE (days t+1 .. t+3) — a real
    3-day-ahead flare forecast, matching what the README claims the system does.

    "Flare" is defined RELATIVE TO THE PATIENT'S OWN BASELINE (their personal
    median severity over their full history), not a population-wide threshold.
    This is closer to how flares are clinically understood — a deviation from
    an individual's normal state — and it means the label isn't just re-deriving
    population statistics from the same numbers used as features.

No feature computed from the future window is ever placed in X.
"""

import pandas as pd
import numpy as np

MIN_DAYS = 30          # minimum distinct check-in days to include a patient
PAST_WINDOW = 7         # days of history used to build features (t-6..t)
FUTURE_WINDOW = 3        # days ahead used to build the label (t+1..t+3)
MIN_STEP = 3             # spacing between sampled checkpoint days per user (reduces autocorrelation)

print("Loading Flaredown dataset...")
df = pd.read_csv('data/ra.csv', low_memory=False)

ra_conditions = df[
    (df['trackable_type'] == 'Condition') &
    (df['trackable_name'].str.contains('rheumatoid|arthritis|RA', case=False, na=False))
]
ra_user_ids = ra_conditions['user_id'].unique()
print(f"RA patients found: {len(ra_user_ids)}")

symptoms = df[(df['user_id'].isin(ra_user_ids)) & (df['trackable_type'] == 'Symptom')].copy()
symptoms['trackable_value'] = pd.to_numeric(symptoms['trackable_value'], errors='coerce')
symptoms['checkin_date'] = pd.to_datetime(symptoms['checkin_date'], errors='coerce')
symptoms = symptoms.dropna(subset=['checkin_date', 'trackable_value'])

# Daily aggregate severity per user (mean across all logged symptoms that day)
daily = (symptoms.groupby(['user_id', 'checkin_date'])['trackable_value']
         .mean()
         .reset_index()
         .rename(columns={'trackable_value': 'daily_severity'}))

# Also track how many symptoms were logged that day (engagement signal, legitimate feature)
daily_count = (symptoms.groupby(['user_id', 'checkin_date'])['trackable_value']
               .count()
               .reset_index()
               .rename(columns={'trackable_value': 'daily_symptom_count'}))
daily = daily.merge(daily_count, on=['user_id', 'checkin_date'])

eligible_users = daily.groupby('user_id')['checkin_date'].nunique()
eligible_users = eligible_users[eligible_users >= MIN_DAYS].index
daily = daily[daily['user_id'].isin(eligible_users)].sort_values(['user_id', 'checkin_date'])
print(f"Users with >= {MIN_DAYS} distinct check-in days: {len(eligible_users)}")

profile = df[df['user_id'].isin(eligible_users)][['user_id', 'age', 'sex']].drop_duplicates('user_id').set_index('user_id')
profile['sex'] = (profile['sex'] == 'female').astype(int)
profile['age'] = pd.to_numeric(profile['age'], errors='coerce')

rows = []
for uid, g in daily.groupby('user_id'):
    g = g.set_index('checkin_date').sort_index()
    # reindex to a continuous daily calendar so rolling windows are real elapsed time, not just "last N rows"
    full_range = pd.date_range(g.index.min(), g.index.max(), freq='D')
    g = g.reindex(full_range)
    g['daily_severity'] = g['daily_severity']
    g['daily_symptom_count'] = g['daily_symptom_count'].fillna(0)

    personal_baseline = g['daily_severity'].median(skipna=True)
    if pd.isna(personal_baseline):
        continue

    n = len(g)
    for t in range(PAST_WINDOW - 1, n - FUTURE_WINDOW, MIN_STEP):
        past = g['daily_severity'].iloc[t - PAST_WINDOW + 1: t + 1]
        future = g['daily_severity'].iloc[t + 1: t + 1 + FUTURE_WINDOW]

        if past.notna().sum() < 3 or future.notna().sum() < 1:
            continue  # not enough real data logged in this window, skip rather than fabricate

        past_mean = past.mean(skipna=True)
        past_std = past.std(skipna=True)
        past_trend = np.polyfit(range(len(past)), past.fillna(past_mean), 1)[0] if past.notna().sum() >= 2 else 0.0
        past_last = past.dropna().iloc[-1] if past.notna().any() else past_mean
        past_engagement = g['daily_symptom_count'].iloc[t - PAST_WINDOW + 1: t + 1].mean()

        future_mean = future.mean(skipna=True)
        flare = int(future_mean > personal_baseline * 1.15)  # 15% above own baseline = flare

        rows.append({
            'user_id': uid,
            'past_mean_severity': past_mean,
            'past_std_severity': past_std,
            'past_trend': past_trend,
            'past_last_severity': past_last,
            'past_engagement': past_engagement,
            'personal_baseline': personal_baseline,
            'flare_next_3d': flare,
        })

final = pd.DataFrame(rows)
final = final.join(profile, on='user_id')
final = final.dropna(subset=['flare_next_3d'])
final[['age']] = final[['age']].fillna(final['age'].median())
final['sex'] = final['sex'].fillna(0)

print(f"\nSamples built: {len(final)}  from {final['user_id'].nunique()} users")
print(f"Flare=1: {final['flare_next_3d'].sum()}  |  Flare=0: {(final['flare_next_3d']==0).sum()}")
print(f"Columns: {final.columns.tolist()}")

final.to_csv('data/ra_processed_v2.csv', index=False)
print("\nSaved -> data/ra_processed_v2.csv")
print("user_id retained so training can do a GROUP split (no user's days in both train and test).")
