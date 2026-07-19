"""
data_cleaning.py
-----------------
Load raw transactional sales data, clean it, and aggregate to a monthly time series.

Input:  data/superstore.csv
Output: outputs/monthly_sales.csv  (columns: date, sales, [optional: category, region])
"""

import pandas as pd
from pathlib import Path

# ---- Config -----------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "superstore.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "monthly_sales.csv"

# If your dataset uses different column names, update this map.
# Keys are the "standard" names this pipeline uses internally.
COLUMN_MAP = {
    "date": "Order Date",
    "sales": "Sales",
    "category": "Category",
    "region": "Region",
}

# Set to a category/region name to filter to a single segment, or None to use all data.
FILTER_CATEGORY = None
FILTER_REGION = None


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Couldn't find {path}. Download the Superstore dataset and save it there "
            f"(see README.md section 1)."
        )
    df = pd.read_csv(path, encoding="latin1")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names we care about
    df = df.rename(columns={
        COLUMN_MAP["date"]: "date",
        COLUMN_MAP["sales"]: "sales",
        COLUMN_MAP["category"]: "category",
        COLUMN_MAP["region"]: "region",
    })

    keep_cols = [c for c in ["date", "sales", "category", "region"] if c in df.columns]
    df = df[keep_cols]

    # Parse dates robustly (Superstore uses DD/MM/YYYY or MM/DD/YYYY depending on locale export)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    n_bad_dates = df["date"].isna().sum()
    if n_bad_dates:
        print(f"[clean] Dropping {n_bad_dates} rows with unparseable dates")
    df = df.dropna(subset=["date"])

    # Sales must be numeric and non-negative (negative sales here usually mean returns;
    # decide deliberately whether to keep them — we keep them by default since net demand matters)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    n_bad_sales = df["sales"].isna().sum()
    if n_bad_sales:
        print(f"[clean] Dropping {n_bad_sales} rows with non-numeric sales")
    df = df.dropna(subset=["sales"])

    # Optional filtering to a single segment (keeps the series simpler to model)
    if FILTER_CATEGORY and "category" in df.columns:
        df = df[df["category"] == FILTER_CATEGORY]
    if FILTER_REGION and "region" in df.columns:
        df = df[df["region"] == FILTER_REGION]

    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level rows into a continuous monthly sales series."""
    monthly = (
        df.set_index("date")["sales"]
        .resample("MS")  # month start
        .sum()
    )

    # Reindex to a continuous monthly range in case some months had zero recorded sales
    full_range = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_range, fill_value=0.0)
    monthly.index.name = "date"

    out = monthly.reset_index().rename(columns={"sales": "sales"})
    return out


def detect_outliers(monthly: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """Flag months whose sales are more than z_thresh standard deviations from the mean.
    Doesn't remove them automatically — outliers in sales are often real events
    (big promotions) that you want to keep and just annotate.
    """
    s = monthly["sales"]
    z = (s - s.mean()) / s.std()
    monthly = monthly.copy()
    monthly["is_outlier"] = z.abs() > z_thresh
    n_flagged = monthly["is_outlier"].sum()
    if n_flagged:
        print(f"[clean] Flagged {n_flagged} months as statistical outliers (kept in data)")
    return monthly


def run():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    raw = load_raw()
    print(f"[clean] Loaded {len(raw)} raw rows")

    cleaned = clean(raw)
    print(f"[clean] {len(cleaned)} rows after cleaning")

    monthly = aggregate_monthly(cleaned)
    monthly = detect_outliers(monthly)
    print(f"[clean] Aggregated to {len(monthly)} monthly data points "
          f"({monthly['date'].min().date()} to {monthly['date'].max().date()})")

    monthly.to_csv(OUTPUT_PATH, index=False)
    print(f"[clean] Saved -> {OUTPUT_PATH}")
    return monthly


if __name__ == "__main__":
    run()
