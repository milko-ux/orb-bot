"""Prove the opening-range calculator is correct before any trading logic.

All tests use synthetic bars with hand-computed expected answers.
Critical traps covered:
  1. Basic high/low over the window, boundary exclusivity [start, end)
  2. UTC storage vs. ET session anchor across a DST transition
  3. Incomplete data -> valid=False (never a silent garbage range)
  4. Bars outside the window must not leak into the range
"""
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from orb.core.opening_range import compute_opening_ranges

UTC = timezone.utc


def make_bars(start_utc: datetime, n: int, highs=None, lows=None, volume=1000):
    """n consecutive 1-min bars starting at start_utc."""
    idx = pd.date_range(start_utc, periods=n, freq="1min", tz="UTC")
    highs = highs if highs is not None else [100.0] * n
    lows = lows if lows is not None else [99.0] * n
    return pd.DataFrame(
        {
            "open": [(h + l) / 2 for h, l in zip(highs, lows)],
            "high": highs,
            "low": lows,
            "close": [(h + l) / 2 for h, l in zip(highs, lows)],
            "volume": [volume] * n,
        },
        index=idx,
    )


class TestBasicCorrectness:
    def test_known_high_low(self):
        # Jan 15 2026: winter, 9:30 ET == 14:30 UTC. 15 bars = full window.
        start = datetime(2026, 1, 15, 14, 30, tzinfo=UTC)
        highs = [100 + i for i in range(15)]          # max = 114 at last bar
        lows = [99 - i * 0.5 for i in range(15)]      # min = 92.0 at last bar
        bars = make_bars(start, 15, highs, lows)

        ranges = compute_opening_ranges(bars, "US_EQUITY", range_minutes=15)
        assert len(ranges) == 1
        r = ranges[0]
        assert r.session_date == date(2026, 1, 15)
        assert r.high == 114.0
        assert r.low == 92.0
        assert r.valid is True
        assert r.bar_count == 15 and r.expected_bars == 15

    def test_window_end_is_exclusive(self):
        # 16 bars: the 16th (09:45 open) has an extreme high that must NOT count.
        start = datetime(2026, 1, 15, 14, 30, tzinfo=UTC)
        highs = [100.0] * 15 + [999.0]
        lows = [99.0] * 16
        bars = make_bars(start, 16, highs, lows)

        r = compute_opening_ranges(bars, "US_EQUITY", range_minutes=15)[0]
        assert r.high == 100.0

    def test_premarket_bars_excluded(self):
        # Bars from 09:00 ET onward; extreme low sits pre-open and must not leak.
        start = datetime(2026, 1, 15, 14, 0, tzinfo=UTC)  # 09:00 ET
        n = 60
        lows = [50.0] * 30 + [99.0] * 30   # garbage low only before 09:30
        highs = [100.0] * n
        bars = make_bars(start, n, highs, lows)

        r = compute_opening_ranges(bars, "US_EQUITY", range_minutes=15)[0]
        assert r.low == 99.0
        assert r.high == 100.0


class TestTimezones:
    def test_dst_transition(self):
        """9:30 ET is 14:30 UTC before spring-forward and 13:30 UTC after.
        If the calculator hard-coded a UTC offset, one of these days breaks.
        DST 2026: US springs forward Sunday March 8."""
        # Friday Mar 6 2026 (EST): 9:30 ET == 14:30 UTC
        winter = make_bars(datetime(2026, 3, 6, 14, 30, tzinfo=UTC), 15,
                           highs=[110.0] * 15, lows=[105.0] * 15)
        # Monday Mar 9 2026 (EDT): 9:30 ET == 13:30 UTC
        summer = make_bars(datetime(2026, 3, 9, 13, 30, tzinfo=UTC), 15,
                           highs=[120.0] * 15, lows=[115.0] * 15)
        bars = pd.concat([winter, summer])

        ranges = compute_opening_ranges(bars, "US_EQUITY", range_minutes=15)
        assert len(ranges) == 2
        assert ranges[0].session_date == date(2026, 3, 6)
        assert (ranges[0].high, ranges[0].low) == (110.0, 105.0)
        assert ranges[0].valid
        assert ranges[1].session_date == date(2026, 3, 9)
        assert (ranges[1].high, ranges[1].low) == (120.0, 115.0)
        assert ranges[1].valid

    def test_naive_index_rejected(self):
        idx = pd.date_range("2026-01-15 14:30", periods=15, freq="1min")  # no tz
        bars = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1},
            index=idx,
        )
        with pytest.raises(ValueError, match="tz-aware"):
            compute_opening_ranges(bars)


class TestDataQuality:
    def test_incomplete_window_marked_invalid(self):
        # Only 9 of 15 expected bars (e.g. halted / missing data).
        start = datetime(2026, 1, 15, 14, 30, tzinfo=UTC)
        bars = make_bars(start, 9)
        r = compute_opening_ranges(bars, "US_EQUITY", range_minutes=15)[0]
        assert r.valid is False
        assert r.bar_count == 9

    def test_day_with_no_window_bars_skipped(self):
        # Bars exist only in the afternoon — no opening range should be emitted.
        start = datetime(2026, 1, 15, 19, 0, tzinfo=UTC)  # 14:00 ET
        bars = make_bars(start, 30)
        assert compute_opening_ranges(bars, "US_EQUITY", range_minutes=15) == []

    def test_multi_day_grouping(self):
        d1 = make_bars(datetime(2026, 1, 15, 14, 30, tzinfo=UTC), 15,
                       highs=[101.0] * 15, lows=[100.0] * 15)
        d2 = make_bars(datetime(2026, 1, 16, 14, 30, tzinfo=UTC), 15,
                       highs=[201.0] * 15, lows=[200.0] * 15)
        ranges = compute_opening_ranges(pd.concat([d1, d2]), range_minutes=15)
        assert [r.session_date for r in ranges] == [date(2026, 1, 15), date(2026, 1, 16)]
        assert ranges[0].high == 101.0 and ranges[1].high == 201.0

    def test_range_minutes_must_divide_evenly(self):
        bars = make_bars(datetime(2026, 1, 15, 14, 30, tzinfo=UTC), 15)
        with pytest.raises(ValueError):
            compute_opening_ranges(bars, range_minutes=7, bar_minutes=5)
