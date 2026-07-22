"""Event-driven ORB backtest engine.

Design principles (all mandated by the build brief):
- Costs are NOT optional: every fill pays slippage, commissions, and
  regulatory fees. There is no code path that runs a cost-free backtest.
- Pessimistic ambiguity resolution: if a bar touches both the stop and the
  target, the stop is assumed to have been hit first.
- Gap handling: if price gaps through a stop/target, the fill is at the
  bar's open (where you'd realistically get filled), not the stop price.
- No lookahead: ATR is computed from prior bars only (shifted by one).
- One trade per session maximum (first confirmed signal), long or short.
- Every trade has a hard stop. No exceptions, by construction.

This is deliberately a purpose-built engine rather than vectorbt/backtrader:
~200 lines we fully understand and can test beats a framework whose corner
cases we'd be guessing about.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from orb.config import SESSIONS, BotConfig
from orb.core.opening_range import compute_opening_ranges
from orb.core.signal_engine import Direction, evaluate


@dataclass
class Trade:
    symbol: str
    session_date: date
    direction: Direction
    entry_time: datetime
    entry_price: float          # includes slippage
    exit_time: datetime
    exit_price: float           # includes slippage
    qty: int
    stop_price: float
    target_price: float
    exit_reason: str            # "stop" | "target" | "time" | "trail"
    costs: float                # commissions + regulatory fees (slippage is in the prices)
    pnl: float                  # net, after all costs
    r_multiple: float           # pnl / initial dollar risk


@dataclass
class BacktestResult:
    trades: list[Trade]
    daily_equity: pd.Series     # equity at end of each session date
    start_equity: float

    @property
    def end_equity(self) -> float:
        return float(self.daily_equity.iloc[-1]) if len(self.daily_equity) else self.start_equity


def compute_atr(bars: pd.DataFrame, period: int) -> pd.Series:
    """ATR from PRIOR bars only (shifted 1) — safe to read on the signal bar."""
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean().shift(1)


def _slip(price: float, bps: float, against_long: bool) -> float:
    """Apply slippage in bps, always against the trader."""
    factor = 1 + bps / 10_000 if against_long else 1 - bps / 10_000
    return price * factor


def run_backtest(
    bars: pd.DataFrame,
    cfg: BotConfig,
    symbol: str = "SYM",
    start_equity: float = 100_000.0,
) -> BacktestResult:
    """Backtest one symbol over the full bar history in `bars` (UTC-indexed)."""
    p, r = cfg.orb, cfg.risk
    sess = SESSIONS[p.session]
    tz = sess["tz"]
    close_h, close_m = sess["close"]

    local = bars.tz_convert(tz)
    atr_all = compute_atr(local, p.atr_period)

    ranges = {
        orng.session_date: orng
        for orng in compute_opening_ranges(bars, p.session, p.range_minutes)
    }

    equity = start_equity
    trades: list[Trade] = []
    eq_dates, eq_vals = [], []

    for session_date, day in local.groupby(local.index.date):
        orng = ranges.get(session_date)
        trade = None
        if orng is not None and orng.valid:
            flatten_at = datetime.combine(
                session_date, time(close_h, close_m), tzinfo=tz
            ) - timedelta(minutes=p.flatten_before_close_min)
            trade = _simulate_session(
                day, atr_all, orng, cfg, symbol, equity, flatten_at
            )
        if trade is not None:
            equity += trade.pnl
            trades.append(trade)
        eq_dates.append(session_date)
        eq_vals.append(equity)

    return BacktestResult(
        trades=trades,
        daily_equity=pd.Series(eq_vals, index=pd.to_datetime(eq_dates)),
        start_equity=start_equity,
    )


def _simulate_session(
    day: pd.DataFrame,
    atr_all: pd.Series,
    orng,
    cfg: BotConfig,
    symbol: str,
    equity: float,
    flatten_at: datetime,
) -> Trade | None:
    p, r = cfg.orb, cfg.risk

    sig = evaluate(day, orng, p)
    if sig is None or sig.timestamp >= flatten_at:
        return None

    is_long = sig.direction == Direction.LONG
    side = 1 if is_long else -1

    # --- stop placement ---
    atr = atr_all.loc[sig.timestamp]
    if p.stop_mode == "atr" and not np.isnan(atr) and atr > 0:
        stop = sig.entry_price - side * p.atr_stop_mult * float(atr)
    else:  # "range" mode, and the fallback if ATR isn't warmed up yet
        stop = orng.low if is_long else orng.high

    entry = _slip(sig.entry_price, r.slippage_bps, against_long=is_long)
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return None

    target = entry + side * p.take_profit_r * stop_dist

    # --- position sizing: fixed fractional risk, no leverage ---
    risk_dollars = equity * (r.risk_per_trade_pct / 100)
    qty = int(risk_dollars // stop_dist)
    qty = min(qty, int(equity // entry))  # cannot buy more than equity affords
    if qty <= 0:
        return None

    # --- walk forward bar by bar ---
    post = day[day.index > sig.timestamp]
    exit_price, exit_time, reason = None, None, None
    trail_stop = stop

    for ts, bar in post.iterrows():
        if ts >= flatten_at:
            exit_price, exit_time, reason = float(bar["close"]), ts, "time"
            break

        eff_stop = max(stop, trail_stop) if is_long else min(stop, trail_stop)

        if is_long:
            gap_out = bar["open"] <= eff_stop
            stop_hit = bar["low"] <= eff_stop
            tgt_hit = bar["high"] >= target
        else:
            gap_out = bar["open"] >= eff_stop
            stop_hit = bar["high"] >= eff_stop
            tgt_hit = bar["low"] <= target

        if gap_out:
            exit_price, exit_time, reason = float(bar["open"]), ts, "stop"
            break
        if stop_hit:  # pessimistic: stop before target on ambiguous bars
            exit_price, exit_time = eff_stop, ts
            reason = "trail" if eff_stop != stop else "stop"
            break
        if tgt_hit:
            exit_price, exit_time, reason = target, ts, "target"
            break

        if p.use_trailing:
            a = atr_all.loc[ts]
            if not np.isnan(a) and a > 0:
                candidate = float(bar["close"]) - side * p.trail_atr_mult * float(a)
                if is_long:
                    trail_stop = max(trail_stop, candidate)
                else:
                    trail_stop = min(trail_stop, candidate)

    if exit_price is None:  # data ended mid-trade: close at last bar
        last = post.iloc[-1] if len(post) else day.iloc[-1]
        exit_price, exit_time, reason = float(last["close"]), day.index[-1], "time"

    exit_fill = _slip(exit_price, r.slippage_bps, against_long=not is_long)

    # --- explicit costs ---
    commissions = r.commission_per_share * qty * 2
    reg_fees = r.regulatory_fee_per_share * qty  # sell side
    costs = commissions + reg_fees

    pnl = side * (exit_fill - entry) * qty - costs
    return Trade(
        symbol=symbol,
        session_date=orng.session_date,
        direction=sig.direction,
        entry_time=sig.timestamp,
        entry_price=entry,
        exit_time=exit_time,
        exit_price=exit_fill,
        qty=qty,
        stop_price=stop,
        target_price=target,
        exit_reason=reason,
        costs=costs,
        pnl=pnl,
        r_multiple=pnl / (stop_dist * qty),
    )
