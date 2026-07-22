"""The four-stage quant validation pipeline.

    Stage 1  Optimization        — Optuna TPE search on IN-SAMPLE data only
    Stage 2  SPP                 — perturb each parameter ±5% / ±10%; reject
                                   parameter sets whose performance collapses
    Stage 3  Monte Carlo         — bootstrap trade sequence; 95th-percentile
                                   worst drawdown and risk-of-ruin estimate
    Stage 4  Out-of-sample       — final params on held-out recent data,
                                   untouched during optimization

A parameter set must pass ALL FOUR stages. Any failure = rejected, no matter
how good the in-sample backtest looked. The out-of-sample data is split off
BEFORE stage 1 and never seen by the optimizer.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from orb.backtest.engine import run_backtest
from orb.backtest.metrics import Metrics, compute_metrics
from orb.config import BotConfig

# --- pass/fail thresholds (conservative; tighten rather than loosen) ---
SPP_MEDIAN_RETENTION = 0.5     # median perturbed score must keep >=50% of original
SPP_POSITIVE_FRACTION = 0.8    # >=80% of perturbed runs must remain profitable
MC_N_PATHS = 2000
MC_MAX_P95_DRAWDOWN = 0.20     # reject if 95th-pct worst drawdown > 20%
OOS_RETENTION = 0.4            # OOS score must be >=40% of in-sample score
OOS_SPLIT = 0.75               # last 25% of sessions held out


@dataclass
class StageReport:
    name: str
    passed: bool
    details: str


@dataclass
class ValidationReport:
    best_params: dict
    in_sample: Metrics | None = None
    out_of_sample: Metrics | None = None
    stages: list[StageReport] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.stages)

    def summary(self) -> str:
        lines = ["=" * 70, "VALIDATION REPORT", "=" * 70]
        lines.append(f"Best params: {self.best_params}")
        if self.in_sample:
            lines.append(f"In-sample : {self.in_sample.summary()}")
        if self.out_of_sample:
            lines.append(f"Out-sample: {self.out_of_sample.summary()}")
        lines.append("-" * 70)
        for s in self.stages:
            lines.append(f"[{'PASS' if s.passed else 'FAIL'}] {s.name}: {s.details}")
        lines.append("-" * 70)
        lines.append(
            "VERDICT: ELIGIBLE FOR PAPER TRADING" if self.passed
            else "VERDICT: REJECTED — do not trade this configuration"
        )
        return "\n".join(lines)


def split_in_out_of_sample(bars: pd.DataFrame, frac: float = OOS_SPLIT):
    """Chronological split by session date — never random for time series."""
    dates = sorted(set(bars.index.date))
    cut = dates[int(len(dates) * frac)]
    is_mask = np.array([d < cut for d in bars.index.date])
    return bars[is_mask], bars[~is_mask]


def apply_params(cfg: BotConfig, params: dict) -> BotConfig:
    cfg = copy.deepcopy(cfg)
    for k, v in params.items():
        setattr(cfg.orb, k, v)
    return cfg


def _score(bars, cfg) -> tuple[float, Metrics]:
    m = compute_metrics(run_backtest(bars, cfg))
    return m.score, m


# ---------------------------------------------------------------- Stage 1
def optimize(bars_is: pd.DataFrame, base_cfg: BotConfig, n_trials: int = 150) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "range_minutes": trial.suggest_categorical("range_minutes", [5, 15, 30]),
            "rel_volume_mult": trial.suggest_float("rel_volume_mult", 1.2, 3.0, step=0.1),
            "stop_mode": trial.suggest_categorical("stop_mode", ["range", "atr"]),
            "atr_stop_mult": trial.suggest_float("atr_stop_mult", 1.0, 3.0, step=0.25),
            "take_profit_r": trial.suggest_float("take_profit_r", 1.0, 3.0, step=0.25),
            "use_trailing": trial.suggest_categorical("use_trailing", [False, True]),
        }
        score, _ = _score(bars_is, apply_params(base_cfg, params))
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------- Stage 2
PERTURBABLE = ["rel_volume_mult", "atr_stop_mult", "take_profit_r"]


def spp_check(bars_is: pd.DataFrame, base_cfg: BotConfig, params: dict) -> StageReport:
    base_score, _ = _score(bars_is, apply_params(base_cfg, params))
    if base_score <= 0:
        return StageReport("SPP", False, "base score is non-positive")

    scores = []
    for key in PERTURBABLE:
        for factor in (0.90, 0.95, 1.05, 1.10):
            perturbed = dict(params)
            perturbed[key] = round(params[key] * factor, 4)
            s, _ = _score(bars_is, apply_params(base_cfg, perturbed))
            scores.append(s)

    median_retention = float(np.median(scores)) / base_score
    positive_frac = float(np.mean([s > 0 for s in scores]))
    passed = median_retention >= SPP_MEDIAN_RETENTION and positive_frac >= SPP_POSITIVE_FRACTION
    return StageReport(
        "SPP",
        passed,
        f"median retention {median_retention:.0%} (need >={SPP_MEDIAN_RETENTION:.0%}), "
        f"profitable under perturbation {positive_frac:.0%} (need >={SPP_POSITIVE_FRACTION:.0%})",
    )


# ---------------------------------------------------------------- Stage 3
def monte_carlo_check(
    bars_is: pd.DataFrame, base_cfg: BotConfig, params: dict, seed: int = 7
) -> StageReport:
    cfg = apply_params(base_cfg, params)
    result = run_backtest(bars_is, cfg)
    rs = np.array([t.r_multiple for t in result.trades])
    if len(rs) < 30:
        return StageReport("Monte Carlo", False, f"only {len(rs)} trades — not enough to resample")

    rng = np.random.default_rng(seed)
    risk_frac = cfg.risk.risk_per_trade_pct / 100
    max_dds = np.empty(MC_N_PATHS)
    terminal = np.empty(MC_N_PATHS)
    for i in range(MC_N_PATHS):
        sample = rng.choice(rs, size=len(rs), replace=True)
        eq = np.cumprod(1 + risk_frac * sample)
        peak = np.maximum.accumulate(eq)
        max_dds[i] = float((1 - eq / peak).max())
        terminal[i] = eq[-1]

    p95_dd = float(np.percentile(max_dds, 95))
    p_loss = float(np.mean(terminal < 1.0))
    passed = p95_dd <= MC_MAX_P95_DRAWDOWN
    return StageReport(
        "Monte Carlo",
        passed,
        f"95th-pct worst drawdown {p95_dd:.1%} (limit {MC_MAX_P95_DRAWDOWN:.0%}), "
        f"P(net loss over period) {p_loss:.1%}, paths={MC_N_PATHS}",
    )


# ---------------------------------------------------------------- Stage 4
def out_of_sample_check(
    bars_oos: pd.DataFrame, base_cfg: BotConfig, params: dict, is_score: float
) -> tuple[StageReport, Metrics]:
    score, m = _score(bars_oos, apply_params(base_cfg, params))
    retention = score / is_score if is_score > 0 else 0.0
    passed = score > 0 and retention >= OOS_RETENTION
    return (
        StageReport(
            "Out-of-sample",
            passed,
            f"OOS score {score:.3f} vs IS {is_score:.3f} — retention {retention:.0%} "
            f"(need >={OOS_RETENTION:.0%} and positive)",
        ),
        m,
    )


# ---------------------------------------------------------------- Orchestrator
def run_pipeline(
    bars: pd.DataFrame, base_cfg: BotConfig, n_trials: int = 150
) -> ValidationReport:
    bars_is, bars_oos = split_in_out_of_sample(bars)

    params = optimize(bars_is, base_cfg, n_trials)
    is_score, is_metrics = _score(bars_is, apply_params(base_cfg, params))

    report = ValidationReport(best_params=params, in_sample=is_metrics)
    report.stages.append(
        StageReport(
            "Optimization",
            is_score > 0,
            f"best in-sample score {is_score:.3f} over {n_trials} trials "
            f"({is_metrics.n_trades} trades)",
        )
    )
    if is_score <= 0:
        report.stages.append(StageReport("SPP", False, "skipped — optimization failed"))
        report.stages.append(StageReport("Monte Carlo", False, "skipped"))
        report.stages.append(StageReport("Out-of-sample", False, "skipped"))
        return report

    report.stages.append(spp_check(bars_is, base_cfg, params))
    report.stages.append(monte_carlo_check(bars_is, base_cfg, params))
    oos_stage, oos_metrics = out_of_sample_check(bars_oos, base_cfg, params, is_score)
    report.stages.append(oos_stage)
    report.out_of_sample = oos_metrics
    return report
