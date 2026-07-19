"""
feature_engineering.py
------------------------
Build time-based features for the monthly sales series:
calendar features, lag features, and rolling statistics.

Input:  outputs/monthly_sales.csv
Output: outputs/features.csv
"""

import pandas as pd
from pathlib import Path

INPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "monthly_sales.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "features.csv"

# Lag and rolling windows in MONTHS (since granularity is monthly)
LAG_MONTHS = [1, 2, 3, 12]        # last month, 2 months ago, 3 months ago, same month last year
ROLLING_WINDOWS = [3, 6, 12]      # 3/6/12-month rolling mean & std


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features plus cyclical month encoding, so the model understands
    Dec is 'close to' Jan rather than treating month as a plain linear number."""
    import numpy as np
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["is_year_end"] = (df["month"] == 12).astype(int)
    df["is_quarter_end"] = df["month"].isin([3, 6, 9, 12]).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    df = df.copy()
    for lag in LAG_MONTHS:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    df = df.copy()
    for w in ROLLING_WINDOWS:
        # shift(1) first so the rolling window only uses PAST data, never the current month
        df[f"rolling_mean_{w}"] = df[target_col].shift(1).rolling(window=w).mean()
        df[f"rolling_std_{w}"] = df[target_col].shift(1).rolling(window=w).std()
    return df


def add_trend_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_index"] = range(len(df))  # simple linear trend proxy
    return df


def run():
    monthly = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    df = add_calendar_features(monthly)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_feature(df)

    n_before = len(df)
    df_model_ready = df.dropna().reset_index(drop=True)
    print(f"[features] Built {df.shape[1]} columns; "
          f"{n_before - len(df_model_ready)} rows dropped due to lag/rolling warm-up "
          f"(expected — you need {max(LAG_MONTHS)} months of history before lags are valid)")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)  # save full version (with NaNs) for inspection
    print(f"[features] Saved -> {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    run()
