"""Baseline forecasters for PJM energy demand.

Models:
- naive_last: repeats the last observed value from the training set.
- seasonal_naive[P]: repeats the last P observations (e.g., P=24 for daily, 168 for weekly) in order.
- moving_average[W]: recursive trailing mean with window W using only historical + previously
  predicted values (no peeking into test actuals).

Inputs (CSV schema from 01_make_dataset.py):
- train CSV with columns: ds (datetime), y (target)
- test CSV with columns: ds (datetime), y (target)

Outputs:
- reports/figs/baseline_<ZONE>.png  (overlay of test actual vs forecasts)
- reports/baseline_metrics.csv      (metrics table)

Example:
    python src/02_baselines.py \
        --train data/processed/AEP_train.csv \
        --test  data/processed/AEP_test.csv \
        --zone AEP \
        --periods 24,168 \
        --windows 24,168
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- Metrics --------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


# --------------------------- Baseline models --------------------------------

def forecast_naive_last(train: np.ndarray, horizon: int) -> np.ndarray:
    if train.size == 0:
        raise ValueError("Empty training series for naive forecast.")
    last_val = float(train[-1])
    return np.full(horizon, last_val, dtype=float)

def forecast_seasonal_naive(train: np.ndarray, horizon: int, period: int) -> np.ndarray:
    if train.size < period:
        # Fallback to naive_last if not enough seasonal history
        return forecast_naive_last(train, horizon)
    pattern = train[-period:]
    reps = int(np.ceil(horizon / period))
    fc = np.tile(pattern, reps)[:horizon]
    return fc.astype(float)

def forecast_moving_average(train: np.ndarray, horizon: int, window: int) -> np.ndarray:
    if train.size < window:
        window = max(1, int(train.size))
    hist = list(map(float, train[-window:]))  # trailing window seed
    out: List[float] = []
    for _ in range(horizon):
        pred = float(np.mean(hist[-window:]))
        out.append(pred)
        hist.append(pred)
    return np.asarray(out, dtype=float)


# ------------------------------- IO utils -----------------------------------

def load_series(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "ds" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Expected columns ['ds','y'] in {path}")
    s = (
        df.rename(columns={"ds": "ds", "y": "y"})
          .assign(ds=lambda d: pd.to_datetime(d["ds"], errors="coerce"))
          .dropna(subset=["ds", "y"])  # drop any failed parses
          .set_index("ds")["y"].astype(float).sort_index()
    )
    return s


def ensure_dirs(figs_dir: Path, reports_dir: Path) -> None:
    figs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------- Plotting -----------------------------------

def plot_overlay(test: pd.Series, preds: Dict[str, pd.Series], zone: str, out_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    test.plot(label="Actual")
    for name, s in preds.items():
        s.plot(label=name)
    plt.title(f"Baseline forecasts vs Actual — {zone}")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ------------------------------- Main ---------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run baseline forecasters on PJM demand",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--train", default="data/processed/AEP_train.csv", help="Path to train CSV")
    ap.add_argument("--test",  default="data/processed/AEP_test.csv", help="Path to test CSV")
    ap.add_argument("--zone",  default="AEP", help="Zone label for titles/filenames")
    ap.add_argument("--periods", default="24,168", help="Comma-separated seasonal periods (hours)")
    ap.add_argument("--windows", default="24,168", help="Comma-separated moving-average windows (hours)")
    ap.add_argument("--out-fig", default="reports/figs", help="Directory for figures")
    ap.add_argument("--out-reports", default="reports", help="Directory for metrics CSV")
    ap.add_argument("--metrics-file", default="baseline_metrics.csv", help="Filename for metrics CSV")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    figs_dir = Path(args.out_fig)
    reports_dir = Path(args.out_reports)
    ensure_dirs(figs_dir, reports_dir)

    train = load_series(Path(args.train))
    test = load_series(Path(args.test))

    # Align horizon and ensure no overlap issues
    horizon = len(test)

    # Compute forecasts
    forecasts: Dict[str, pd.Series] = {}
    y_tr = train.to_numpy()

    # Naive last
    yhat_naive = forecast_naive_last(y_tr, horizon)
    forecasts["naive_last"] = pd.Series(yhat_naive, index=test.index)

    # Seasonal naive(s)
    periods = [int(p.strip()) for p in str(args.periods).split(",") if p.strip()]
    for p in periods:
        yhat = forecast_seasonal_naive(y_tr, horizon, period=p)
        forecasts[f"seasonal_naive[{p}h]"] = pd.Series(yhat, index=test.index)

    # Moving averages
    windows = [int(w.strip()) for w in str(args.windows).split(",") if w.strip()]
    for w in windows:
        yhat = forecast_moving_average(y_tr, horizon, window=w)
        forecasts[f"moving_avg[{w}h]"] = pd.Series(yhat, index=test.index)

    # Metrics table
    rows: List[Dict[str, object]] = []
    y_true = test.to_numpy()
    for name, s in forecasts.items():
        y_pred = s.to_numpy()
        rows.append(
            {
                "zone": args.zone,
                "model": name,
                "MAE": mae(y_true, y_pred),
                "RMSE": rmse(y_true, y_pred),
                "MAPE_%": mape(y_true, y_pred),
                "n_test": horizon,
                "test_start": str(test.index.min()),
                "test_end": str(test.index.max()),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["RMSE", "MAE"])  # best at top
    metrics_path = reports_dir / args.metrics_file
    metrics.to_csv(metrics_path, index=False)

    # Plot overlay
    fig_path = figs_dir / f"baseline_{args.zone}.png"
    plot_overlay(test, forecasts, zone=args.zone, out_path=fig_path)

    # Console summary
    print("\n✅ Baselines completed")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {fig_path}")
    print("\nTop models by RMSE:\n", metrics[["model", "RMSE", "MAE", "MAPE_%"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
