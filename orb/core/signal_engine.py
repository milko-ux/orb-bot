"""Signal engine — breakout confirmation logic.

*** PLACEHOLDER ***
The "yellow bar" logic below is the stand-in described in the build brief:
    yellow bar = bar volume > rel_volume_mult × rolling average volume
                 AND bar closes beyond the range boundary in the breakout
                 direction.

The real indicator (Pine Script) will replace `is_yellow_bar` before
Phase 1 validation runs. Nothing downstream may depend on the internals
of this module — only on the Signal dataclass returned by evaluate().
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd

from orb.config import ORBParams
from orb.core.opening_range import OpeningRange


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class Signal:
    timestamp: datetime
    direction: Direction
    entry_price: float       # close of the confirming bar
    range_high: float
    range_low: float


def is_yellow_bar(
    bar: pd.Series,
    avg_volume: float,
    rel_volume_mult: float,
) -> bool:
    """PLACEHOLDER volume-confirmation check. Replace with real indicator."""
    if avg_volume <= 0:
        return False
    return bar["volume"] > rel_volume_mult * avg_volume


def evaluate(
    session_bars: pd.DataFrame,
    orange: OpeningRange,
    params: ORBParams,
) -> Signal | None:
    """Scan post-range bars of one session; return the first confirmed
    breakout signal, or None.

    session_bars: bars for ONE session date, in session-local tz,
    covering at minimum the range window plus the trading window.
    """
    if not orange.valid:
        return None

    post = session_bars[session_bars.index >= orange.range_end]
    if post.empty:
        return None

    rolling_avg = (
        session_bars["volume"]
        .rolling(params.rel_volume_lookback, min_periods=params.rel_volume_lookback)
        .mean()
        .shift(1)  # average of PRIOR bars only — no lookahead
    )

    for ts, bar in post.iterrows():
        avg = rolling_avg.loc[ts]
        if pd.isna(avg):
            continue
        if bar["close"] > orange.high and is_yellow_bar(bar, avg, params.rel_volume_mult):
            return Signal(ts, Direction.LONG, float(bar["close"]), orange.high, orange.low)
        if bar["close"] < orange.low and is_yellow_bar(bar, avg, params.rel_volume_mult):
            return Signal(ts, Direction.SHORT, float(bar["close"]), orange.high, orange.low)
    return None
