"""Opening Range calculation.

This module does exactly one thing: given intraday OHLCV bars for one
symbol, compute the opening range (high/low) for each session.

Design rules:
- Market-agnostic. The session anchor (open time, timezone) comes from
  config.SESSIONS — nothing US-specific is hard-coded here.
- Input bars must be a DataFrame indexed by tz-aware UTC timestamps with
  columns: open, high, low, close, volume. (This is what the Alpaca data
  layer produces; the OANDA layer must produce the identical shape.)
- Strict about data quality: a session's range is only marked `valid` if
  bars actually cover the range window. A half-empty window (halted stock,
  missing data) produces valid=False and the strategy must skip that day.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pandas as pd

from orb.config import SESSIONS

REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class OpeningRange:
    session_date: date
    range_start: datetime      # tz-aware, session timezone
    range_end: datetime        # exclusive
    high: float
    low: float
    bar_count: int             # bars actually observed in the window
    expected_bars: int         # bars the window should contain
    valid: bool                # True only if coverage is complete

    @property
    def width(self) -> float:
        return self.high - self.low


def _validate_bars(bars: pd.DataFrame) -> None:
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must be indexed by a DatetimeIndex")
    if bars.index.tz is None:
        raise ValueError("bars index must be tz-aware (UTC)")
    missing = REQUIRED_COLS - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars index must be sorted ascending")


def compute_opening_ranges(
    bars: pd.DataFrame,
    session: str = "US_EQUITY",
    range_minutes: int = 15,
    bar_minutes: int = 1,
    min_coverage: float = 1.0,
) -> list[OpeningRange]:
    """Compute the opening range for every session date present in `bars`.

    min_coverage: fraction of expected bars that must be present for the
    range to be marked valid. Default 1.0 (all bars required) — the
    conservative choice; loosen only with a deliberate decision.
    """
    _validate_bars(bars)
    if range_minutes % bar_minutes != 0:
        raise ValueError("range_minutes must be a multiple of bar_minutes")

    cfg = SESSIONS[session]
    tz = cfg["tz"]
    open_h, open_m = cfg["open"]

    local = bars.tz_convert(tz)
    expected_bars = range_minutes // bar_minutes

    results: list[OpeningRange] = []
    for session_date, day_bars in local.groupby(local.index.date):
        start = datetime.combine(session_date, time(open_h, open_m), tzinfo=tz)
        end = start + timedelta(minutes=range_minutes)

        # Bar timestamps are bar-open times: include [start, end).
        window = day_bars[(day_bars.index >= start) & (day_bars.index < end)]
        n = len(window)

        if n == 0:
            # No bars in the window at all (holiday remnants, bad data) —
            # emit nothing rather than a garbage range.
            continue

        valid = n >= expected_bars * min_coverage
        results.append(
            OpeningRange(
                session_date=session_date,
                range_start=start,
                range_end=end,
                high=float(window["high"].max()),
                low=float(window["low"].min()),
                bar_count=n,
                expected_bars=expected_bars,
                valid=bool(valid),
            )
        )
    return results
