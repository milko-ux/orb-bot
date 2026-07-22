"""Tests for the placeholder signal engine.

The exact yellow-bar logic will be replaced, but two properties must hold
for ANY implementation and are locked in here:
  1. No signal from an invalid opening range.
  2. Volume baseline uses only PRIOR bars (no lookahead bias).
"""
from datetime import datetime, timezone

import pandas as pd

from orb.config import ORBParams
from orb.core.opening_range import compute_opening_ranges
from orb.core.signal_engine import Direction, evaluate

UTC = timezone.utc


def session_frame(closes, volumes, start=datetime(2026, 1, 15, 14, 30, tzinfo=UTC)):
    idx = pd.date_range(start, periods=len(closes), freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )
    return df.tz_convert("America/New_York")


def params():
    return ORBParams(range_minutes=15, rel_volume_mult=1.5, rel_volume_lookback=10)


def test_long_breakout_with_volume_confirms():
    # Range bars ~100, then a breakout close at 101.0 on 3x volume.
    closes = [100.0] * 15 + [100.2, 101.0, 101.1]
    volumes = [1000] * 15 + [1000, 3000, 1000]
    bars = session_frame(closes, volumes)
    orange = compute_opening_ranges(bars.tz_convert("UTC"), range_minutes=15)[0]

    sig = evaluate(bars, orange, params())
    assert sig is not None
    assert sig.direction == Direction.LONG
    assert sig.entry_price == 101.0


def test_breakout_without_volume_is_ignored():
    closes = [100.0] * 15 + [101.0, 101.2, 101.3]
    volumes = [1000] * 18  # never exceeds 1.5x average
    bars = session_frame(closes, volumes)
    orange = compute_opening_ranges(bars.tz_convert("UTC"), range_minutes=15)[0]
    assert evaluate(bars, orange, params()) is None


def test_short_breakout():
    closes = [100.0] * 15 + [99.5, 98.0]
    volumes = [1000] * 15 + [1000, 4000]
    bars = session_frame(closes, volumes)
    orange = compute_opening_ranges(bars.tz_convert("UTC"), range_minutes=15)[0]
    sig = evaluate(bars, orange, params())
    assert sig is not None and sig.direction == Direction.SHORT


def test_invalid_range_produces_no_signal():
    closes = [100.0] * 8 + [105.0]  # incomplete range window
    volumes = [1000] * 8 + [9000]
    bars = session_frame(closes, volumes)
    orange = compute_opening_ranges(bars.tz_convert("UTC"), range_minutes=15)[0]
    assert orange.valid is False
    assert evaluate(bars, orange, params()) is None


def test_no_lookahead_in_volume_baseline():
    """The confirming bar's own huge volume must not inflate the baseline
    it is compared against. Construct a case that only signals if the
    baseline excludes the current bar."""
    closes = [100.0] * 15 + [101.0]
    # avg of prior 10 bars = 1000; current bar volume 1600 => 1.6x, passes.
    # If the current bar leaked into its own baseline the average would rise
    # and 1.6x would fail at mult=1.5 (1600 vs 1.5*~1054=1581 still passes...)
    # so make it tight: mult such that inclusion flips the outcome.
    volumes = [1000] * 15 + [1600]
    bars = session_frame(closes, volumes)
    orange = compute_opening_ranges(bars.tz_convert("UTC"), range_minutes=15)[0]
    p = ORBParams(range_minutes=15, rel_volume_mult=1.55, rel_volume_lookback=10)
    # prior-only baseline: 1600 > 1.55*1000 = 1550 -> signal
    # leaky baseline (includes current bar): avg=1060, 1.55*1060=1643 -> no signal
    sig = evaluate(bars, orange, p)
    assert sig is not None and sig.direction == Direction.LONG
