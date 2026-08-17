"""Download bars in chunks and assemble the cache file the pipeline expects.

Usage:
    PYTHONPATH=. python3 scripts/prefetch_chunked.py SPY 2022-01-01 2026-07-01
"""
import sys
from datetime import datetime, timezone
import pandas as pd
from orb.data.alpaca_data import fetch_bars, _cache_path, CACHE_DIR

def main():
    symbol = sys.argv[1]
    start = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    final = _cache_path(symbol, start, end, "1Min", "sip")
    if final.exists():
        print(f"Already cached: {final}")
        return

    edges = pd.date_range(start, end, freq="QS", tz="UTC").tolist()
    if not edges or edges[0] > start:
        edges.insert(0, start)
    if edges[-1] < end:
        edges.append(end)

    parts = []
    for i in range(len(edges) - 1):
        a, b = edges[i].to_pydatetime(), edges[i + 1].to_pydatetime()
        print(f"  [{i+1}/{len(edges)-1}] {a.date()} -> {b.date()} ...", flush=True)
        df = fetch_bars(symbol, a, b)
        print(f"      {len(df):,} bars", flush=True)
        parts.append(df)

    full = pd.concat(parts).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    full.to_parquet(final)
    print(f"\nAssembled {len(full):,} bars across "
          f"{len(set(full.index.date)):,} sessions -> {final}")

if __name__ == "__main__":
    main()
