"""
modeling.py
-----------
Fit and generate forecasts from three models:
  1. Naive baseline   (same month last year)
  2. SARIMA           (classical time-series, captures seasonality)
  3. XGBoost/RandomForest regression (uses lag + calendar features)

Uses a time-ordered train/test split (last TEST_MONTHS held out) — never shuffle.

Input:  outputs/features.csv, outputs/monthly_sales.csv
Output: outputs/predictions.csv (actual + each model's prediction, test period only)
"""

import warnings
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")  # SARIMA convergence warnings are noisy on small data

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
FEATURES_PATH = OUTPUTS / "features.csv"
MONTHLY_PATH = OUTPUTS / "monthly_sales.csv"
PRED_PATH = OUTPUTS / "predictions.csv"

TEST_MONTHS = 6  # hold out the last 6 months as the test set

FEATURE_COLS = [
    "month", "quarter", "is_year_end", "is_quarter_end", "month_sin", "month_cos",
    "lag_1", "lag_2", "lag_3", "lag_12",
    "rolling_mean_3", "rolling_std_3", "rolling_mean_6", "rolling_mean_12",
    "time_index",
]
TARGET_COL = "sales"


def naive_baseline(monthly: pd.DataFrame, test_months: int = TEST_MONTHS) -> pd.Series:
    """Forecast = same month last year (12-month-ago lag)."""
    s = monthly.set_index("date")["sales"]
    pred = s.shift(12)
    return pred.tail(test_months)


def fit_sarima(monthly: pd.DataFrame, test_months: int = TEST_MONTHS):
    s = monthly.set_index("date")["sales"]
    train, test = s.iloc[:-test_months], s.iloc[-test_months:]

    # Seasonal order (1,1,1,12) is a reasonable default starting point for monthly
    # retail data with yearly seasonality — tune p,d,q via AIC search if you have time.
    model = SARIMAX(
        train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=test_months)
    forecast.index = test.index
    return forecast


def fit_regression(df: pd.DataFrame, test_months: int = TEST_MONTHS):
    """Fits XGBoost if available, else falls back to RandomForest."""
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

    train = df.iloc[:-test_months]
    test = df.iloc[-test_months:]

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test = test[FEATURE_COLS]

    if HAS_XGB:
        model = XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        model_name = "xgboost"
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
        model_name = "random_forest"

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    result = pd.Series(preds, index=test["date"].values)
    return result, model, model_name


def run():
    monthly = pd.read_csv(MONTHLY_PATH, parse_dates=["date"])
    features = pd.read_csv(FEATURES_PATH, parse_dates=["date"])

    if len(monthly) < TEST_MONTHS + 12:
        print(f"[model] WARNING: only {len(monthly)} months of data — "
              f"forecasts (especially SARIMA and lag_12) will be unreliable. "
              f"Ideally have 24+ months of history.")

    actual = monthly.set_index("date")["sales"].tail(TEST_MONTHS)

    baseline_pred = naive_baseline(monthly)
    sarima_pred = fit_sarima(monthly)
    reg_pred, reg_model, reg_name = fit_regression(features)

    results = pd.DataFrame({
        "date": actual.index,
        "actual": actual.values,
        "baseline_naive": baseline_pred.reindex(actual.index).values,
        "sarima": sarima_pred.reindex(actual.index).values,
        f"regression_{reg_name}": reg_pred.reindex(actual.index).values,
    })

    OUTPUTS.mkdir(exist_ok=True)
    results.to_csv(PRED_PATH, index=False)
    print(f"[model] Fit baseline, SARIMA, and {reg_name} regression")
    print(f"[model] Saved predictions -> {PRED_PATH}")
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    run()
