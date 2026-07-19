"""
visualization.py
-----------------
Business-friendly outputs:
  1. Actual vs forecast line chart (all models, test period)
  2. Full history + forecast chart (best model only, cleaner for a report)
  3. Plain-English summary table

Input:  outputs/predictions.csv, outputs/metrics.csv, outputs/monthly_sales.csv
Output: outputs/forecast_chart.png, outputs/summary_table.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
PRED_PATH = OUTPUTS / "predictions.csv"
METRICS_PATH = OUTPUTS / "metrics.csv"
MONTHLY_PATH = OUTPUTS / "monthly_sales.csv"
CHART_PATH = OUTPUTS / "forecast_chart.png"
SUMMARY_PATH = OUTPUTS / "summary_table.csv"

plt.style.use("seaborn-v0_8-whitegrid")


def plot_all_models(results: pd.DataFrame, save_path: Path = CHART_PATH):
    model_cols = [c for c in results.columns if c not in ("date", "actual")]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results["date"], results["actual"], label="Actual", color="black",
            marker="o", linewidth=2)

    for col in model_cols:
        ax.plot(results["date"], results[col], label=col.replace("_", " ").title(),
                 marker="o", linestyle="--", alpha=0.8)

    ax.set_title("Monthly Sales: Actual vs Forecast (Test Period)", fontsize=13, weight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Saved chart -> {save_path}")


def plot_full_history_with_forecast(monthly: pd.DataFrame, results: pd.DataFrame,
                                     best_model_col: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly["date"], monthly["sales"], label="Historical Actual", color="steelblue")
    ax.plot(results["date"], results[best_model_col], label=f"Forecast ({best_model_col})",
             color="darkorange", marker="o", linestyle="--")
    ax.axvspan(results["date"].min(), results["date"].max(), color="orange", alpha=0.08,
               label="Forecast period")

    ax.set_title("Sales History with Forecast", fontsize=13, weight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    full_path = save_path.parent / "forecast_full_history.png"
    fig.savefig(full_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Saved chart -> {full_path}")


def build_summary_table(results: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    best_model = metrics.iloc[0]["model"]
    best_mape = metrics.iloc[0]["MAPE_%"]

    summary = results[["date", "actual", best_model]].copy()
    summary = summary.rename(columns={best_model: "forecast"})
    summary["expected_range_low"] = summary["forecast"] * (1 - best_mape / 100)
    summary["expected_range_high"] = summary["forecast"] * (1 + best_mape / 100)
    summary["model_used"] = best_model

    summary[["actual", "forecast", "expected_range_low", "expected_range_high"]] = \
        summary[["actual", "forecast", "expected_range_low", "expected_range_high"]].round(0)

    return summary


def run():
    results = pd.read_csv(PRED_PATH, parse_dates=["date"])
    metrics = pd.read_csv(METRICS_PATH)
    monthly = pd.read_csv(MONTHLY_PATH, parse_dates=["date"])

    best_model_col = metrics.iloc[0]["model"]

    plot_all_models(results)
    plot_full_history_with_forecast(monthly, results, best_model_col, CHART_PATH)

    summary = build_summary_table(results, metrics)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"[viz] Saved summary table -> {SUMMARY_PATH}")
    print(summary.to_string(index=False))

    print(f"\n[viz] Headline: best model is '{best_model_col}' "
          f"(MAPE {metrics.iloc[0]['MAPE_%']}%) — "
          f"on average forecasts are within {metrics.iloc[0]['MAPE_%']}% of actual sales.")
    return summary


if __name__ == "__main__":
    run()
