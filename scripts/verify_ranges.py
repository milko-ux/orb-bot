"""Step-1 acceptance check: run against REAL Alpaca data on the droplet.

Usage (with ALPACA_API_KEY / ALPACA_SECRET_KEY exported):
    python scripts/verify_ranges.py SPY 2026-06-01 2026-07-01          # SIP (default)
    python scripts/verify_ranges.py SPY 2026-06-01 2026-07-01 iex      # IEX, for comparison

Prints each session's opening range so you can spot-check a handful of
days against a TradingView chart (draw the 09:30–09:45 ET box manually
and compare high/low). Sign-off on ~5 random days = step 1 done.

Also prints total window volume so IEX vs SIP can be compared directly —
if SIP is really unlocked for this account, the SIP number should be
roughly 30-50x the IEX number (IEX is ~2-3% of consolidated volume).
"""
import sys
from datetime import datetime, timezone

from orb.core.opening_range import compute_opening_ranges
from orb.data.alpaca_data import fetch_bars


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    start = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)
    feed = sys.argv[4] if len(sys.argv) > 4 else "sip"

    bars = fetch_bars(symbol, start, end, feed=feed, use_cache=False)
    print(f"{len(bars)} bars fetched for {symbol} (feed={feed})")

    ranges = compute_opening_ranges(bars, "US_EQUITY", range_minutes=15)
    print(f"{len(ranges)} session ranges computed\n")
    print(f"{'date':<12}{'high':>10}{'low':>10}{'width':>8}{'bars':>6}  {'volume':>12}  valid")
    for r in ranges:
        window_vol = bars.loc[(bars.index >= r.range_start) & (bars.index < r.range_end), "volume"].sum()
        print(
            f"{r.session_date!s:<12}{r.high:>10.2f}{r.low:>10.2f}"
            f"{r.width:>8.2f}{r.bar_count:>4}/{r.expected_bars}  {window_vol:>12,.0f}  {r.valid}"
        )

    invalid = [r for r in ranges if not r.valid]
    if invalid:
        print(f"\nNOTE: {len(invalid)} sessions have incomplete range windows "
              "(will be skipped by the strategy). If this is more than a few, "
              "check the data feed (IEX vs SIP).")


if __name__ == "__main__":
    main()
