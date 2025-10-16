import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths (use your dataset filename)
DATA_PATH = os.getenv("DATA_PATH", "data/raw/clean_hotel_booking.csv")
OUT_DIR = os.getenv("OUT_DIR", "data/processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Allow overriding the target column via env var
TARGET_COL = os.getenv("TARGET_COL")  # if set, used directly

# Fallback list of likely target column names for hotel booking dataset
LIKELY_TARGETS = [
    "is_canceled", "is_cancelled", "cancelled", "cancellation",
    "booking_status", "reservation_status", "reserved", "target",
    "label", "y"
]

def detect_target_column(df: pd.DataFrame):
    # 1) If env var set and column exists -> use it
    if TARGET_COL:
        if TARGET_COL in df.columns:
            print(f"Using target column from env: '{TARGET_COL}'")
            return TARGET_COL
        else:
            print(f"Environment TARGET_COL='{TARGET_COL}' was set but not found in dataframe columns.")

    # 2) Search for an exact match in likely candidates
    for cand in LIKELY_TARGETS:
        if cand in df.columns:
            print(f"Auto-detected target column: '{cand}'")
            return cand

    # 3) Case-insensitive match
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in LIKELY_TARGETS:
        if cand.lower() in cols_lower:
            print(f"Auto-detected target column (case-insensitive): '{cols_lower[cand.lower()]}'")
            return cols_lower[cand.lower()]

    # 4) No target found — return None (caller will report)
    return None

def load_raw(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. Put your CSV at this path or set DATA_PATH.")
    df = pd.read_csv(path)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    # detect target
    target = detect_target_column(df)
    if target is None:
        # helpful error: list columns and instructions
        cols = df.columns.tolist()
        raise KeyError(
            "No target column found. Please set TARGET_COL env var to the correct column name, "
            "or rename your target column to one of the common names.\n\n"
            f"Available columns: {cols}\n\n"
            "Common names I checked for: " + ", ".join(LIKELY_TARGETS)
        )

    # drop rows missing target
    df = df.dropna(subset=[target])
    # Move target to a column named 'target' for downstream code expectations
    if target != "target":
        df = df.rename(columns={target: "target"})

    # example encoding for categorical columns (customize as needed)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for c in cat_cols:
        if c == "target":  # skip already renamed target
            continue
        df[c] = df[c].astype('category').cat.codes

    return df

def split_save(df):
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None
    )
    X_train.to_parquet(os.path.join(OUT_DIR, "X_train.parquet"))
    X_test.to_parquet(os.path.join(OUT_DIR, "X_test.parquet"))
    pd.DataFrame(y_train).to_parquet(os.path.join(OUT_DIR, "y_train.parquet"))
    pd.DataFrame(y_test).to_parquet(os.path.join(OUT_DIR, "y_test.parquet"))
    print("Saved processed data to", OUT_DIR)


if __name__ == "__main__":
    df = load_raw()
    df = clean(df)
    split_save(df)