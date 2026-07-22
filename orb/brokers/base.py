"""Broker adapter interface.

The strategy engine talks ONLY to this interface. Alpaca (Phase 1) and
OANDA (Phase 2) each implement it. If any strategy-side code ever imports
alpaca-py or an OANDA client directly, that is a bug.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class Position:
    symbol: str
    qty: float          # positive = long, negative = short
    avg_entry: float


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    filled_qty: float
    filled_avg_price: float | None
    status: str


class BrokerAdapter(ABC):
    """Everything the ORB engine needs from a broker, and nothing more."""

    @abstractmethod
    def get_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Min"
    ) -> pd.DataFrame:
        """Return OHLCV bars indexed by tz-aware UTC timestamps with columns
        open/high/low/close/volume — the canonical shape used everywhere."""

    @abstractmethod
    def get_account_equity(self) -> float: ...

    @abstractmethod
    def get_open_positions(self) -> list[Position]: ...

    @abstractmethod
    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: str,               # "buy" | "sell"
        stop_price: float,
        take_profit_price: float,
    ) -> OrderResult:
        """Entry + attached stop-loss + take-profit as one atomic bracket.
        Every entry MUST go through this — no naked entries."""

    @abstractmethod
    def flatten_all(self) -> None:
        """Kill switch: cancel all open orders and close all positions."""

    @abstractmethod
    def is_market_open(self) -> bool: ...
