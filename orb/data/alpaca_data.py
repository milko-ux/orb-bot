"""Historical data pipeline (Phase 1: Alpaca US equities).

Fetches 1-minute OHLCV via alpaca-py, normalizes it to the canonical
shape (tz-aware UTC index; open/high/low/close/volume columns), and
caches to parquet so the backtest/validation pipeline never re-downloads.

Env vars required (same pattern as the alpaca-bot):
    ALPACA_API_KEY
    ALPACA_SECRET_KEY

Notes:
- Alpaca's free/Basic plan restricts LIVE real-time data to IEX only, but
  historical queries (end time 15+ minutes old) can request the full SIP
  (all-exchanges) feed for free. Since backtesting/validation only ever
  touches historical data, we default to feed='sip' here — this matters a
  lot for the yellow-bar logic, which depends on accurate volume. IEX
  alone is ~2-3% of consolidated volume and would badly distort it.
  Live trading later (Step 4+) still only sees IEX in real time unless
  the account is upgraded — that's a separate decision for that stage.
- We request raw 1Min bars and filter to regular session downstream; the
  opening-range calculator anchors to 09:30 America/New_York itself.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(os.environ.get("ORB_DATA_DIR", "data_cache"))

CANONICAL_COLS = ["open", "high", "low", "close", "volume"]


def _cache_path(symbol: str, start: datetime, end: datetime, timeframe: str, feed: str) -> Path:
    key = f"{symbol}_{timeframe}_{feed}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    return CACHE_DIR / key


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Force any bar frame into the canonical shape and validate it."""
    df = df.rename(columns=str.lower)
    df = df[CANONICAL_COLS].copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive prices in bar data — refusing to cache")
    bad = (df["high"] < df["low"]).sum()
    if bad:
        raise ValueError(f"{bad} bars have high < low — corrupt data")
    return df


def fetch_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1Min",
    use_cache: bool = True,
    feed: str = "sip",
) -> pd.DataFrame:
    """Fetch (or load cached) historical bars for one symbol.

    feed: 'sip' (default) for full consolidated volume — free for historical
    queries ending 15+ minutes ago, which every backtest query does. Pass
    'iex' explicitly only to deliberately reproduce what the live bot will
    see in real time before an account upgrade.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, start, end, timeframe, feed)
    if use_cache and path.exists():
        return pd.read_parquet(path)

    # Imported lazily so the rest of the package works without alpaca-py
    # installed (e.g. running unit tests on synthetic data).
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed

    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    )
    tf = TimeFrame(1, TimeFrameUnit.Minute) if timeframe == "1Min" else TimeFrame.Day
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end,
        feed=DataFeed(feed),
    )
    raw = client.get_stock_bars(req).df

    # alpaca-py returns a MultiIndex (symbol, timestamp) — drop the symbol level.
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.xs(symbol, level="symbol")

    df = normalize(raw)
    df.to_parquet(path)
    return df
