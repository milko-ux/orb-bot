"""Performance metrics computed from a BacktestResult.

The optimizer targets `score` — a risk-adjusted number (MAR = CAR/MaxDD),
never raw return, per the brief. Too few trades = automatic disqualification,
because any metric on 8 trades is noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orb.backtest.engine import BacktestResult

TRADING_DAYS = 252
MIN_TRADES = 30  # below this, results are statistical noise -> score 0


@dataclass
class Metrics:
    n_trades: int
    win_rate: float
    avg_r: float
    profit_factor: float
    total_return_pct: float
    car_pct: float              # compound annual return
    max_drawdown_pct: float
    mar: float                  # CAR / MaxDD — the optimization target
    sharpe: float
    sortino: float
    score: float                # what the optimizer maximizes

    def summary(self) -> str:
        return (
            f"trades={self.n_trades}  win={self.win_rate:.1%}  avgR={self.avg_r:+.2f}  "
            f"PF={self.profit_factor:.2f}  CAR={self.car_pct:+.1f}%  "
            f"MaxDD={self.max_drawdown_pct:.1f}%  MAR={self.mar:.2f}  "
            f"Sharpe={self.sharpe:.2f}  Sortino={self.sortino:.2f}  score={self.score:.3f}"
        )


def compute_metrics(result: BacktestResult) -> Metrics:
    trades = result.trades
    n = len(trades)
    eq = result.daily_equity

    if n == 0 or len(eq) < 2:
        return Metrics(n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)

    rs = np.array([t.r_multiple for t in trades])
    pnls = np.array([t.pnl for t in trades])
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    win_rate = len(wins) / n
    profit_factor = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf")

    total_return = eq.iloc[-1] / result.start_equity - 1
    years = max(len(eq) / TRADING_DAYS, 1e-9)
    car = (eq.iloc[-1] / result.start_equity) ** (1 / years) - 1

    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    max_dd = float(-dd)

    daily_ret = eq.pct_change().dropna()
    if daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS))
    else:
        sharpe = 0.0
    downside = daily_ret[daily_ret < 0]
    if len(downside) and downside.std() > 0:
        sortino = float(daily_ret.mean() / downside.std() * np.sqrt(TRADING_DAYS))
    else:
        sortino = sharpe

    mar = float(car / max_dd) if max_dd > 0 else 0.0

    if n < MIN_TRADES or car <= 0:
        score = 0.0
    else:
        score = mar

    return Metrics(
        n_trades=n,
        win_rate=win_rate,
        avg_r=float(rs.mean()),
        profit_factor=profit_factor,
        total_return_pct=total_return * 100,
        car_pct=car * 100,
        max_drawdown_pct=max_dd * 100,
        mar=mar,
        sharpe=sharpe,
        sortino=sortino,
        score=score,
    )
