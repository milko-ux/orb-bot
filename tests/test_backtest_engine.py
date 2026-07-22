"""Backtest engine correctness tests.

Each test constructs a synthetic session where the correct trade outcome is
known by hand, then verifies the engine produces exactly that outcome:
  - clean run to target -> ~ +take_profit_r R (minus costs)
  - reversal to stop    -> ~ -1 R (minus costs)
  - ambiguous bar (touches stop AND target) -> stop wins (pessimistic)
  - no signal -> no trade
  - costs are always > 0 and reduce pnl
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

from orb.backtest.engine import run_backtest
from orb.config import BotConfig, ORBParams, RiskParams

UTC = timezone.utc
START = datetime(2026, 1, 15, 14, 30, tzinfo=UTC)  # 09:30 ET


def build_day(rows):
    """rows: list of (open, high, low, close, volume) 1-min bars from 09:30."""
    idx = pd.date_range(START, periods=len(rows), freq="1min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def cfg(**orb_kwargs):
    orb = ORBParams(
        range_minutes=15,
        rel_volume_mult=1.5,
        rel_volume_lookback=10,
        stop_mode="range",
        take_profit_r=2.0,
        use_trailing=False,
        flatten_before_close_min=5,
        **orb_kwargs,
    )
    risk = RiskParams(risk_per_trade_pct=1.0, slippage_bps=0.0,
                      commission_per_share=0.0, regulatory_fee_per_share=0.0)
    return BotConfig(orb=orb, risk=risk)


def range_then(post_rows):
    """15 range bars (100.5-99.5, vol 1000) + custom post-range bars."""
    rows = [(100.0, 100.5, 99.5, 100.0, 1000)] * 15 + post_rows
    return build_day(rows)


class TestOutcomes:
    def test_clean_run_to_target(self):
        # Breakout close 101 on 3x volume. Range low 99.5 -> stop dist 1.5,
        # target = 101 + 2*1.5 = 104. Price then runs straight up through 104.
        day = range_then([
            (100.4, 101.1, 100.3, 101.0, 3000),   # signal bar
            (101.0, 102.5, 100.9, 102.4, 1200),
            (102.4, 104.5, 102.3, 104.2, 1500),   # target 104 hit here
        ])
        res = run_backtest(day, cfg())
        assert len(res.trades) == 1
        t = res.trades[0]
        assert t.exit_reason == "target"
        assert t.exit_price == pytest.approx(104.0)
        assert t.r_multiple == pytest.approx(2.0, abs=0.01)

    def test_reversal_to_stop(self):
        # Same entry; price reverses down through range low 99.5.
        day = range_then([
            (100.4, 101.1, 100.3, 101.0, 3000),   # signal bar
            (101.0, 101.2, 100.2, 100.3, 900),
            (100.3, 100.4, 99.2, 99.3, 1100),     # stop 99.5 hit here
        ])
        res = run_backtest(day, cfg())
        t = res.trades[0]
        assert t.exit_reason == "stop"
        assert t.exit_price == pytest.approx(99.5)
        assert t.r_multiple == pytest.approx(-1.0, abs=0.01)

    def test_ambiguous_bar_assumes_stop_first(self):
        # One giant bar touches BOTH the stop (99.5) and target (104):
        # pessimistic rule says the stop fires.
        day = range_then([
            (100.4, 101.1, 100.3, 101.0, 3000),   # signal bar
            (101.0, 105.0, 99.0, 104.0, 5000),    # touches both
        ])
        t = run_backtest(day, cfg()).trades[0]
        assert t.exit_reason == "stop"
        assert t.r_multiple < 0

    def test_time_exit_at_flatten(self):
        # Price breaks out then drifts sideways forever -> closed by time exit,
        # never left open overnight.
        drift = [(101.0, 101.2, 100.8, 101.0, 900)] * 370
        day = range_then([(100.4, 101.1, 100.3, 101.0, 3000)] + drift)
        t = run_backtest(day, cfg()).trades[0]
        assert t.exit_reason == "time"

    def test_no_volume_no_trade(self):
        day = range_then([
            (100.4, 101.1, 100.3, 101.0, 1000),   # breakout but flat volume
            (101.0, 101.5, 100.9, 101.4, 1000),
        ])
        assert run_backtest(day, cfg()).trades == []

    def test_short_side(self):
        # Breakdown below 99.5 on volume; runs down to target.
        day = range_then([
            (99.6, 99.7, 98.9, 99.0, 3000),       # short signal, stop 100.5
            (99.0, 99.1, 95.0, 95.2, 2000),       # target 99-2*1.5=96 hit
        ])
        t = run_backtest(day, cfg()).trades[0]
        assert t.direction.value == "short"
        assert t.exit_reason == "target"
        assert t.r_multiple == pytest.approx(2.0, abs=0.01)


class TestCosts:
    def test_costs_reduce_pnl(self):
        day = range_then([
            (100.4, 101.1, 100.3, 101.0, 3000),
            (101.0, 104.5, 100.9, 104.2, 1500),
        ])
        free = cfg()
        costly = cfg()
        costly.risk = RiskParams(risk_per_trade_pct=1.0, slippage_bps=5.0,
                                 commission_per_share=0.01,
                                 regulatory_fee_per_share=0.000166)
        pnl_free = run_backtest(day, free).trades[0].pnl
        t_costly = run_backtest(day, costly).trades[0]
        assert t_costly.pnl < pnl_free
        assert t_costly.costs > 0

    def test_equity_curve_tracks_pnl(self):
        day = range_then([
            (100.4, 101.1, 100.3, 101.0, 3000),
            (101.0, 104.5, 100.9, 104.2, 1500),
        ])
        res = run_backtest(day, cfg(), start_equity=50_000)
        assert res.end_equity == pytest.approx(50_000 + res.trades[0].pnl)
