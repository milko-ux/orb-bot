# ORB Scalper Bot

Autonomous Opening Range Breakout bot. Phase 1: US stocks via Alpaca. Phase 2 (only after Phase 1 is fully validated): forex via OANDA. One strategy, no scope creep.

## Architecture

The strategy engine never touches a broker SDK directly. All execution goes through `orb/brokers/base.py::BrokerAdapter` — Alpaca implements it now, OANDA implements the same interface later. All bar data, from any source, is normalized to one canonical shape: tz-aware UTC index, columns `open/high/low/close/volume`.

```
orb/
  config.py              all strategy params (optimizer search space) + hard risk limits
  core/
    opening_range.py     session-anchored range calculation  [DONE, tested]
    signal_engine.py     yellow-bar confirmation  [PLACEHOLDER — awaiting real indicator]
  brokers/
    base.py              abstract BrokerAdapter (bracket orders, kill switch)  [DONE]
    alpaca_adapter.py    [step 4]
  data/
    alpaca_data.py       historical 1-min bars, validated + parquet-cached  [DONE]
tests/                   14 correctness tests, all passing
scripts/
  verify_ranges.py       step-1 acceptance check against real Alpaca data
```

## Build order status

1. **Data pipeline + opening-range calculator — THIS DELIVERY.** Correctness proven on synthetic data (boundary exclusivity, DST transitions, incomplete-data rejection, no premarket leakage). Final sign-off: run `scripts/verify_ranges.py` on the droplet with real data and spot-check ~5 days against TradingView.
2. Signal engine (real yellow-bar logic) + basic backtester with cost model.
3. Validation pipeline: Optuna optimization → SPP perturbation → Monte Carlo → out-of-sample holdout.
4. Alpaca paper trading, minimum 8 weeks, varied conditions. Telegram status + kill switch.
5. Live, minimum size, risk caps active.
6. Phase 2: OANDA adapter, London/NY session anchors (already defined in config, disabled).

## Running step 1 on the droplet

```bash
git clone <repo> && cd orb-bot
pip install -r requirements.txt
python -m pytest tests/ -v                       # must be 14/14
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
python scripts/verify_ranges.py SPY 2026-06-01 2026-07-01
```

## Design decisions already locked in

- A session with incomplete range-window data is `valid=False` and is **skipped** — never traded on a garbage range.
- Volume baselines use prior bars only (lookahead bias is tested against, not just avoided).
- Risk limits (`RiskParams`) are hard constraints, never part of the optimizer's search space.
- Every backtest includes slippage + fees (`RiskParams` cost model) — zero-cost backtests are not allowed to exist.
- Every live entry is a bracket order (entry + stop + target atomically); no naked entries.

## Open items (from the brief)

- **Real yellow-bar indicator** — needed before step 2 finishes. Drop the Pine Script into the chat and it gets translated into `signal_engine.py` with tests.
- Data feed: free Alpaca = IEX only (partial volume). Volume-based signals will behave differently on IEX vs SIP — decide before validation runs.
- Per-trade risk % and starting capital — defaults set to 0.5% / max 1 position; confirm before live.
