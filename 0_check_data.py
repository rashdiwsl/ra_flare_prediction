import pandas as pd

files = {
    "RA":          "data/ra.csv",
    "Sleep":       "data/sleep.csv",
    "HRV":         "data/hrv.csv",
    "Sri Lankan":  "data/sri_lankan_ra.csv",
}

for name, path in files.items():
    try:
        df = pd.read_csv(path)
        print(f"\n{'='*50}")
        print(f"  {name} Dataset")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(df.head(2))
    except FileNotFoundError:
        print(f"\n[MISSING] {path} — download and place in data/ folder")