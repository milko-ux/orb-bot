"""Central configuration for the ORB bot.

Every parameter that the quant validation pipeline will optimize lives here,
in one dataclass, so the optimizer can sweep it and the live bot can load
the validated set from a single source of truth.

NOTHING in here is broker-specific. Broker credentials/settings live in
environment variables read by the broker adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

# Session anchors. Phase 1 uses only US_EQUITY. Forex anchors are defined
# now so the engine is market-agnostic, but they are NOT used until Phase 2.
SESSIONS = {
    "US_EQUITY": {"tz": ZoneInfo("America/New_York"), "open": (9, 30), "close": (16, 0)},
    # Phase 2 (do not enable until Phase 1 fully validated):
    "FX_LONDON": {"tz": ZoneInfo("UTC"), "open": (8, 0), "close": (16, 30)},
    "FX_NEWYORK": {"tz": ZoneInfo("UTC"), "open": (13, 30), "close": (22, 0)},
}


@dataclass
class ORBParams:
    """Strategy parameters — the search space for the optimizer."""

    session: str = "US_EQUITY"

    # Opening range window in minutes. Optimizer sweeps {5, 15, 30}.
    range_minutes: int = 15

    # --- Signal engine (PLACEHOLDER — see signal_engine.py) ---
    # Relative-volume multiple the breakout bar must exceed vs. its
    # rolling average ("yellow bar" placeholder logic).
    rel_volume_mult: float = 1.5
    rel_volume_lookback: int = 20  # bars used for the average-volume baseline

    # --- Exits ---
    # Stop mode: "atr" or "range" (opposite side of the opening range).
    # The optimizer decides which; both are implemented.
    stop_mode: str = "range"
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    take_profit_r: float = 1.5   # fixed R-multiple target
    use_trailing: bool = False   # optimizable
    trail_atr_mult: float = 1.0
    # Hard time exit: flatten this many minutes before session close.
    flatten_before_close_min: int = 5


@dataclass
class RiskParams:
    """Risk limits — NOT optimizable. These are hard constraints.

    Defaults are set to the conservative end per the brief (first-time
    trader). Confirm before live; never loosened by the optimizer.
    """

    risk_per_trade_pct: float = 0.5    # % of equity risked per trade
    max_daily_loss_pct: float = 1.5    # halt for the day if hit
    max_concurrent_positions: int = 1
    # Cost model for backtests — no backtest may run with zero costs.
    slippage_bps: float = 2.0          # per side, conservative for liquid US equities
    commission_per_share: float = 0.0  # Alpaca is commission-free; SEC/TAF fees below
    regulatory_fee_per_share: float = 0.000166  # TAF sell-side approx; refine later


@dataclass
class BotConfig:
    orb: ORBParams = field(default_factory=ORBParams)
    risk: RiskParams = field(default_factory=RiskParams)
    symbols: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"])
    bar_timeframe: str = "1Min"
