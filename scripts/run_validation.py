"""Run the full four-stage validation pipeline on real Alpaca data.

Usage (droplet, with keys exported):
    PYTHONPATH=. python3 scripts/run_validation.py SPY 2023-01-01 2026-07-01 150

Args: symbol, start date, end date, optuna trials (default 150).

Fetches multi-year 1-min SIP bars (cached to parquet after first download),
splits off the most recent 25% as untouched out-of-sample data, optimizes on
the rest, then runs SPP -> Monte Carlo -> out-of-sample. Prints the full
report and saves the winning parameters to validated_params_<SYMBOL>.json
ONLY if all four stages pass.
"""
import json
import sys
from datetime import datetime, timezone

from orb.backtest.pipeline import run_pipeline
from orb.config import BotConfig
from orb.data.alpaca_data import fetch_bars


def main():
    symbol = sys.argv[1]
    start = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)
    n_trials = int(sys.argv[4]) if len(sys.argv) > 4 else 150

    print(f"Fetching {symbol} 1-min bars {start.date()} -> {end.date()} (SIP)...")
    bars = fetch_bars(symbol, start, end)
    n_days = len(set(bars.index.date))
    print(f"{len(bars):,} bars across {n_days} sessions. Running pipeline "
          f"({n_trials} optimization trials — this can take a while)...\n")

    report = run_pipeline(bars, BotConfig(), n_trials=n_trials)
    print(report.summary())

    if report.passed:
        out = f"validated_params_{symbol}.json"
        with open(out, "w") as f:
            json.dump(report.best_params, f, indent=2)
        print(f"\nSaved winning parameters to {out}")
    else:
        print("\nNo parameters saved — configuration rejected by the pipeline.")


if __name__ == "__main__":
    main()
