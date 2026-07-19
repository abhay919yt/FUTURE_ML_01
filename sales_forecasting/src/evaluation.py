"""
evaluation.py
-------------
Compute error metrics (MAE, RMSE, MAPE) for each model and do a light residual
analysis — useful both for picking the best model and for the write-up
("errors are largest in Nov/Dec, likely due to holiday demand spikes").

Input:  outputs/predictions.csv
Output: outputs/metrics.csv, outputs/residuals.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
PRED_PATH = OUTPUTS / "predictions.csv"
METRICS_PATH = OUTPUTS / "metrics.csv"
RESIDUALS_PATH = OUTPUTS / "residuals.csv"


def mae(actual, pred):
    return np.mean(np.abs(actual - pred))


def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))


def mape(actual, pred):
    # Guard against division by zero months
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100


def evaluate(results: pd.DataFrame) -> pd.DataFrame:
    model_cols = [c for c in results.columns if c not in ("date", "actual")]
    actual = results["actual"].values

    rows = []
    for col in model_cols:
        pred = results[col].values
        valid = ~np.isnan(pred)
        rows.append({
            "model": col,
            "MAE": round(mae(actual[valid], pred[valid]), 2),
            "RMSE": round(rmse(actual[valid], pred[valid]), 2),
            "MAPE_%": round(mape(actual[valid], pred[valid]), 2),
        })

    metrics = pd.DataFrame(rows).sort_values("MAPE_%").reset_index(drop=True)
    return metrics


def residual_analysis(results: pd.DataFrame) -> pd.DataFrame:
    """Per-month residuals for the best model (lowest overall MAPE), to spot
    whether errors cluster around specific months/seasons."""
    metrics = evaluate(results)
    best_model = metrics.iloc[0]["model"]

    res = results[["date", "actual", best_model]].copy()
    res["residual"] = res["actual"] - res[best_model]
    res["abs_pct_error"] = (res["residual"].abs() / res["actual"].replace(0, np.nan)) * 100
    res["best_model"] = best_model
    return res


def run():
    results = pd.read_csv(PRED_PATH, parse_dates=["date"])

    metrics = evaluate(results)
    residuals = residual_analysis(results)

    METRICS_PATH.parent.mkdir(exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    residuals.to_csv(RESIDUALS_PATH, index=False)

    print("[eval] Model comparison (sorted by MAPE, lower = better):")
    print(metrics.to_string(index=False))
    print(f"\n[eval] Best model: {metrics.iloc[0]['model']}")
    print(f"[eval] Saved -> {METRICS_PATH}, {RESIDUALS_PATH}")
    return metrics, residuals


if __name__ == "__main__":
    run()
