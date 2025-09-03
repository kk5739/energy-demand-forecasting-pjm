"""Make dataset for PJM energy demand forecasting.

Usage (from repo root):
    # Real data
    python src/01_make_dataset.py \
        --csv data/raw/PJM_Load_hourly.csv \
        --zone AEP \
        --out data/processed \
        [--datetime-col Datetime] [--freq H] [--test-days 30]

    # Demo mode (no CSV needed)
    python src/01_make_dataset.py --demo --zone AEP

Outputs:
    data/processed/<ZONE>_train.csv
    data/processed/<ZONE>_test.csv

Notes:
    - If you run without --csv, the script falls back to a demo generator to avoid argparse errors.
    - Enforces regular time frequency and interpolates small gaps.
    - Splits the last N days (default 30) as the test set.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, List

import numpy as np
import pandas as pd


# --- CLI parsing ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prepare train/test CSVs from raw PJM CSV (or demo data)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv", default=None, help="Path to raw PJM CSV (omit to use --demo)")
    ap.add_argument("--zone", default="AEP", help="Zone column to use (e.g., AEP, PJME)")
    ap.add_argument("--out", default="data/processed", help="Output directory for train/test CSVs")
    ap.add_argument("--datetime-col", default=None, help="Name of datetime column if known")
    ap.add_argument("--freq", default="H", help="Target frequency for time index")
    ap.add_argument("--test-days", type=int, default=30, help="Number of days at the end to reserve for test")
    ap.add_argument("--demo", action="store_true", help="Generate a synthetic demo dataset if no CSV is provided")
    ap.add_argument("--demo-hours", type=int, default=24 * 120, help="Hours to generate for demo mode (~120 days)")
    ap.add_argument("--smoke-test", action="store_true", help="Run basic assertions after writing outputs")
    return ap.parse_args()


# --- Helpers ----------------------------------------------------------------

def infer_datetime_col(columns: List[str]) -> Optional[str]:
    """Return the first likely datetime column name or None if not found."""
    candidates = {c.lower(): c for c in columns}
    for key in ("datetime", "time", "timestamp", "date"):
        if key in candidates:
            return candidates[key]
    # last resort: look for any column containing 'time' or 'date'
    for c in columns:
        cl = c.lower()
        if "time" in cl or "date" in cl:
            return c
    return None


def safe_asfreq(s: pd.Series, freq: str) -> pd.Series:
    """Reindex to requested freq and interpolate.
    Why: downstream models expect an evenly spaced time grid.
    """
    if not freq:
        freq = "H"
    s = s.asfreq(freq)
    return s.interpolate(limit_direction="both")


def make_demo_csv(zone: str, hours: int, raw_dir: Path) -> Path:
    """Generate a synthetic hourly demand CSV with columns [Datetime, <zone>]."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.utcnow().floor("H")
    idx = pd.date_range(end - pd.Timedelta(hours=hours - 1), end, freq="H")

    # Daily + weekly seasonality with noise
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    daily = 200 * np.sin(2 * np.pi * hour / 24.0)
    weekly = 100 * np.cos(2 * np.pi * (dow + hour / 24.0) / 7.0)
    baseline = 1000
    noise = np.random.normal(0, 50, size=len(idx))
    y = baseline + daily + weekly + noise

    demo = pd.DataFrame({"Datetime": idx, zone: y})
    demo_path = raw_dir / f"demo_{zone}.csv"
    demo.to_csv(demo_path, index=False)
    print(f"[Info] Demo CSV created: {demo_path} (rows={len(demo)})")
    return demo_path


# --- Main -------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # If no CSV provided, auto-enable demo to prevent argparse SystemExit(2)
    if args.csv is None and not args.demo:
        print("[Info] No --csv provided; running in --demo mode with synthetic data.")
        args.demo = True

    # Resolve input path (real or demo)
    raw_path: Optional[Path] = None
    if args.demo:
        raw_path = make_demo_csv(args.zone, max(48, int(args.demo_hours)), Path("data/raw"))
        # Ensure datetime column name used by demo
        if args.datetime_col is None:
            args.datetime_col = "Datetime"
    else:
        raw_path = Path(args.csv)
        if not raw_path.exists():
            sys.exit(f"[Error] CSV not found: {raw_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    if df.empty:
        sys.exit("[Error] Input CSV is empty.")

    dt_col = args.datetime_col or infer_datetime_col(list(df.columns))
    if dt_col is None:
        sys.exit("[Error] Could not find a datetime column. Use --datetime-col to specify explicitly.")

    if args.zone not in df.columns:
        sys.exit(f"[Error] Zone '{args.zone}' not found. Available columns: {list(df.columns)}")

    # Parse time and coerce target to numeric
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df[args.zone] = pd.to_numeric(df[args.zone], errors="coerce")

    # Keep only needed columns and clean
    df = (
        df[[dt_col, args.zone]]
        .dropna()
        .rename(columns={dt_col: "ds", args.zone: "y"})
        .sort_values("ds")
        .drop_duplicates(subset=["ds"], keep="last")
        .set_index("ds")
    )

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        sys.exit("[Error] Failed to parse datetime index. Check --datetime-col.")

    # Enforce regular frequency and fill small gaps
    df["y"] = safe_asfreq(df["y"], args.freq)

    if df.index.size < 48:
        sys.exit("[Error] Not enough data after cleaning (need at least ~2 days).")

    # Train/test split
    test_days = max(1, int(args.test_days))  # avoid zero
    split_point = df.index.max() - pd.Timedelta(days=test_days)

    if split_point <= df.index.min():
        # Fallback if dataset is very short: 80/20 split
        cutoff = int(len(df) * 0.8)
        train, test = df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()
        print("[Warn] Dataset too short for requested test-days; used 80/20 split instead.")
    else:
        # Avoid overlapping index step
        train = df.loc[:split_point].copy()
        test = df.loc[split_point + pd.Timedelta(hours=1):].copy()

    # Save
    train_path = out_dir / f"{args.zone}_train.csv"
    test_path = out_dir / f"{args.zone}_test.csv"
    train.to_csv(train_path)
    test.to_csv(test_path)

    print(
        "\n✅ Dataset prepared",
        f"\n  Mode:        {'DEMO' if args.demo else 'REAL'}",
        f"\n  Zone:        {args.zone}",
        f"\n  Rows (all):  {len(df)}",
        f"\n  Train rows:  {len(train)}",
        f"\n  Test rows:   {len(test)}",
        f"\n  Saved:       {train_path}",
        f"\n              {test_path}\n",
    )

    # Optional smoke tests
    if args.smoke_test:
        assert len(train) > 0 and len(test) > 0, "Train/Test should be non-empty"
        assert train.index.max() < test.index.min(), "Train must end before test begins"
        assert pd.infer_freq(train.index) in {"H", "h"} or train.index.freq is not None, "Hourly frequency expected"
        print("[Tests] Smoke tests passed ✅")


if __name__ == "__main__":
    main()
