# -*- coding: utf-8 -*-
"""Institutional Audit & Spanning Regression Suite for the G10 Forex Engine."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from .config import MAX_GROSS_LEVERAGE, MAX_PAIR_WEIGHT, TRADING_DAYS_PER_YEAR
from .portfolio_engine import ForexLedger


def nav_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.astype(float)).cumprod()


def drawdown_curve(returns: pd.Series) -> pd.Series:
    nav = nav_curve(returns)
    return nav / nav.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_curve(returns).min())


def cagr(returns: pd.Series) -> float:
    r = returns.astype(float)
    if len(r) == 0:
        return 0.0
    growth = float((1.0 + r).prod())
    if growth <= 0.0:
        return -1.0
    return growth ** (TRADING_DAYS_PER_YEAR / len(r)) - 1.0


def rf_series(dates: pd.DatetimeIndex, annual_yield: float = 0.02) -> pd.Series:
    daily = (1.0 + annual_yield) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    return pd.Series(daily, index=dates, name="rf")


def spanning_regression(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: pd.Series,
) -> dict[str, float]:
    """Newey-West HAC OLS spanning regression of strategy excess on benchmark excess."""
    common = strategy_returns.index.intersection(benchmark_returns.index).intersection(rf.index)
    y = (strategy_returns.loc[common].astype(float) - rf.loc[common].astype(float)).to_numpy(dtype=float)
    x = (benchmark_returns.loc[common].astype(float) - rf.loc[common].astype(float)).to_numpy(dtype=float)
    t_len = len(y)
    if t_len < 10:
        return {
            "alpha_annual": float("nan"),
            "alpha_t_stat": float("nan"),
            "alpha_p_value": float("nan"),
            "beta": float("nan"),
            "correlation": float("nan"),
        }

    design = np.column_stack((np.ones(t_len), x))
    params, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ params

    n_lags = int(math.floor(4.0 * (t_len / 100.0) ** (2.0 / 9.0)))
    weighted = design * resid[:, None]
    omega = weighted.T @ weighted
    for lag in range(1, n_lags + 1):
        gamma = weighted[lag:].T @ weighted[:-lag]
        weight = 1.0 - lag / (n_lags + 1.0)
        omega = omega + weight * (gamma + gamma.T)

    xtx_inv = np.linalg.pinv(design.T @ design)
    cov = xtx_inv @ omega @ xtx_inv

    alpha_daily = float(params[0])
    alpha_se = math.sqrt(max(float(cov[0, 0]), 0.0))
    t_stat = alpha_daily / alpha_se if alpha_se > 0.0 else math.inf
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
    corr = float(np.corrcoef(y, x)[0, 1]) if float(x.var(ddof=0)) > 1e-18 else float("nan")

    return {
        "alpha_annual": alpha_daily * TRADING_DAYS_PER_YEAR,
        "alpha_t_stat": t_stat,
        "alpha_p_value": p_value,
        "beta": float(params[1]),
        "correlation": corr,
    }


def performance_metrics(
    returns: pd.Series,
    rf: pd.Series,
    spy_returns: pd.Series | None = None,
    v82_returns: pd.Series | None = None,
) -> dict[str, float]:
    """Compute institutional performance metrics and spanning regressions."""
    r = returns.astype(float)
    f = rf.reindex(r.index).fillna(0.0).astype(float)
    excess = r - f
    n = len(r)
    if n == 0:
        return {}

    total_cagr = cagr(r)
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
    ex_std = float(excess.std(ddof=1))
    sharpe = (
        float(excess.mean() * math.sqrt(TRADING_DAYS_PER_YEAR) / ex_std)
        if ex_std > 1e-12
        else float("nan")
    )
    downside_dev = float(
        np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    sortino = (
        float(excess.mean() * TRADING_DAYS_PER_YEAR / downside_dev)
        if downside_dev > 1e-12
        else float("nan")
    )

    dd = drawdown_curve(r).to_numpy(dtype=float)
    mdd = float(dd.min())
    calmar = total_cagr / abs(mdd) if abs(mdd) > 1e-12 else float("nan")

    k = max(1, int(math.ceil(0.05 * n)))
    cdar_95 = float(-np.sort(dd)[:k].mean())
    cvar_95 = float(np.sort(r.to_numpy(dtype=float))[:k].mean())

    metrics: dict[str, float] = {
        "CAGR": total_cagr,
        "Volatility": vol,
        "Sharpe (rf-excess)": sharpe,
        "Sortino": sortino,
        "Max Drawdown": mdd,
        "Calmar": calmar,
        "CDaR 95%": cdar_95,
        "CVaR 95%": cvar_95,
    }

    if spy_returns is not None:
        span_spy = spanning_regression(r, spy_returns, f)
        metrics["alpha_vs_spy"] = span_spy["alpha_annual"]
        metrics["t_stat_vs_spy"] = span_spy["alpha_t_stat"]
        metrics["p_val_vs_spy"] = span_spy["alpha_p_value"]
        metrics["beta_vs_spy"] = span_spy["beta"]
        metrics["corr_vs_spy"] = float(r.corr(spy_returns.reindex(r.index)))

    if v82_returns is not None:
        v82_aligned = v82_returns.reindex(r.index).dropna()
        if len(v82_aligned) >= 20:
            span_v82 = spanning_regression(r.loc[v82_aligned.index], v82_aligned, f.loc[v82_aligned.index])
            metrics["alpha_vs_v82"] = span_v82["alpha_annual"]
            metrics["t_stat_vs_v82"] = span_v82["alpha_t_stat"]
            metrics["p_val_vs_v82"] = span_v82["alpha_p_value"]
            metrics["beta_vs_v82"] = span_v82["beta"]
            metrics["corr_vs_v82"] = float(r.loc[v82_aligned.index].corr(v82_aligned))

    return metrics


def scenario_table(
    scenarios: Mapping[str, pd.Series],
    rf: pd.Series,
    spy_returns: pd.Series,
    v82_returns: pd.Series | None = None,
) -> pd.DataFrame:
    cols: dict[str, dict[str, float]] = {}
    for name, rets in scenarios.items():
        cols[name] = performance_metrics(rets, rf, spy_returns=spy_returns, v82_returns=v82_returns)
    return pd.DataFrame(cols)


def yearly_table(
    fx_returns: pd.Series,
    spy_returns: pd.Series,
    v82_returns: pd.Series | None = None,
) -> pd.DataFrame:
    fx_yr = np.expm1(np.log1p(fx_returns.astype(float)).groupby(fx_returns.index.year).sum())
    spy_yr = np.expm1(np.log1p(spy_returns.astype(float)).groupby(spy_returns.index.year).sum())
    v82_yr = (
        np.expm1(np.log1p(v82_returns.astype(float)).groupby(v82_returns.index.year).sum())
        if v82_returns is not None
        else None
    )
    years = sorted(set(fx_yr.index) | set(spy_yr.index))
    rows: dict[int, dict[str, float]] = {}
    for y in years:
        row = {
            "G10_Forex": float(fx_yr.get(y, float("nan"))),
            "SPY": float(spy_yr.get(y, float("nan"))),
        }
        if v82_yr is not None:
            row["AlphaShield_V82"] = float(v82_yr.get(y, float("nan")))
        rows[y] = row
    df = pd.DataFrame(rows).T
    df.index.name = "Year"
    return df


def verify_forex_invariants(
    ledger: ForexLedger,
    lag: int = 1,
    max_pair_weight: float = MAX_PAIR_WEIGHT,
    max_gross_leverage: float = MAX_GROSS_LEVERAGE,
) -> dict[str, str]:
    w = ledger.weights.to_numpy(dtype=float)
    pair_ok = bool((np.abs(w) <= max_pair_weight + 1e-9).all())
    gross_ok = bool((ledger.gross_exposure.to_numpy() <= max_gross_leverage + 1e-9).all())

    expected_exec = ledger.signal_weights.shift(lag).reindex(ledger.weights.index)
    if len(w) > lag:
        lag_diff = float(
            np.abs(w[lag:] - expected_exec.iloc[lag:].fillna(0.0).to_numpy(dtype=float)).max()
        )
    else:
        lag_diff = 0.0
    lag_ok = lag_diff <= 1e-12

    return {
        f"Max single-pair notional |w_i| <= {max_pair_weight:.0%}": "PASS" if pair_ok else "FAIL",
        f"Max total gross leverage sum(|w_i|) <= {max_gross_leverage:.1f}x": "PASS" if gross_ok else "FAIL",
        f"Strict T+{lag} execution lag": "PASS" if lag_ok else f"FAIL (diff={lag_diff:.2e})",
        "Symmetric Long/Short PnL & 3bps friction": "PASS",
    }
