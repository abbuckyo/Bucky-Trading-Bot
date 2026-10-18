# -*- coding: utf-8 -*-
"""Module D — Institutional audit, stress testing & validation gates.

1. Standard institutional metrics: CAGR, annual volatility, Sharpe (excess
   of a risk-free proxy), Sortino, Max Drawdown, Calmar, CDaR (95%),
   CVaR (95%), upside/downside capture.
2. Stress testing: 2x transaction cost stress and +1 day execution lag (T+2).
3. Spanning regression: strategy excess returns on benchmark (SPY) excess
   returns with Newey-West HAC standard errors (alpha, alpha t-stat, beta,
   correlation).
4. Sub-period breakdown: yearly returns and historical regime windows.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR

BENCHMARK_LABEL: str = "SPY B&H"
SUBPERIODS: tuple[tuple[str, str, str], ...] = (
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2021 (Pandemic)", "2020-01-01", "2021-12-31"),
    ("2022 (Bear)", "2022-01-01", "2022-12-31"),
    ("2023-2026", "2023-01-01", "2099-12-31"),
)


# ---------------------------------------------------------------------------
# Basic series math
# ---------------------------------------------------------------------------

def nav_curve(returns: pd.Series) -> pd.Series:
    """Growth of 1 for a daily return series."""
    return (1.0 + returns.astype(float)).cumprod()


def drawdown_curve(returns: pd.Series) -> pd.Series:
    """Drawdown series (<= 0) of the growth-of-1 path."""
    nav = nav_curve(returns)
    return nav / nav.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_curve(returns)
    return float(dd.min())


def yearly_returns(returns: pd.Series) -> pd.Series:
    """Compounded return per calendar year."""
    r = returns.astype(float)
    grouped = np.log1p(r).groupby(r.index.year).sum()
    return np.expm1(grouped)


def cagr(returns: pd.Series) -> float:
    r = returns.astype(float)
    growth = float((1.0 + r).prod())
    if growth <= 0 or len(r) == 0:
        return -1.0
    return growth ** (TRADING_DAYS_PER_YEAR / len(r)) - 1.0


def rf_series(dates: pd.DatetimeIndex, annual_yield: float) -> pd.Series:
    """Constant daily risk-free proxy (money-market park)."""
    daily = (1.0 + annual_yield) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    return pd.Series(daily, index=dates, name="rf")


# ---------------------------------------------------------------------------
# Spanning regression with Newey-West HAC standard errors
# ---------------------------------------------------------------------------

def spanning_regression(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: pd.Series,
) -> dict[str, float | None]:
    """OLS of strategy excess on [1, benchmark excess] with Newey-West HAC.

    Returns annualized alpha, its HAC t-stat, beta, correlation, and the
    two-sided p-value of alpha (using erfc; no scipy dependency).
    """
    y = (strategy_returns.astype(float) - rf).to_numpy(dtype=float)
    x = (benchmark_returns.astype(float) - rf).to_numpy(dtype=float)
    t_len = len(y)
    design = np.column_stack((np.ones(t_len), x))
    params, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ params

    # Bartlett-kernel bandwidth (Ledoit's rule): L = floor(4 (T/100)^(2/9)).
    n_lags = int(math.floor(4.0 * (t_len / 100.0) ** (2.0 / 9.0)))
    weighted = design * resid[:, None]
    omega = weighted.T @ weighted  # heteroskedasticity meat: sum_t (x_t e_t)(x_t e_t)'
    for lag in range(1, n_lags + 1):
        gamma = weighted[lag:].T @ weighted[:-lag]
        weight = 1.0 - lag / (n_lags + 1.0)
        omega = omega + weight * (gamma + gamma.T)
    xtx_inv = np.linalg.inv(design.T @ design)
    cov = xtx_inv @ omega @ xtx_inv  # sandwich: Var(b) = (X'X)^-1 S (X'X)^-1

    alpha_daily = float(params[0])
    alpha_se = math.sqrt(max(float(cov[0, 0]), 0.0))
    t_stat = alpha_daily / alpha_se if alpha_se > 0 else math.inf
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
    var_x = float(x.var(ddof=0))
    corr = (
        float(np.corrcoef(y, x)[0, 1]) if var_x > 1e-18 else float("nan")
    )
    return {
        "alpha_annual": alpha_daily * TRADING_DAYS_PER_YEAR,
        "alpha_t_stat": t_stat,
        "alpha_p_value": p_value,
        "beta": float(params[1]),
        "correlation": corr,
        "hac_lags": n_lags,
    }


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def performance_metrics(
    returns: pd.Series,
    rf: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float]:
    """Standard institutional metrics for one daily return series."""
    r = returns.astype(float)
    f = rf.reindex(r.index).astype(float)
    excess = r - f
    n = len(r)

    growth = float(nav_curve(r).iloc[-1])
    total_cagr = growth ** (TRADING_DAYS_PER_YEAR / n) - 1.0
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

    # CDaR 95%: mean of the worst 5% of drawdown magnitudes.
    k = max(1, int(math.ceil(0.05 * n)))
    worst_dd = np.sort(dd)[:k]
    cdar_95 = float(-worst_dd.mean())
    # CVaR 95%: mean of the worst 5% of daily returns.
    worst_r = np.sort(r.to_numpy(dtype=float))[:k]
    cvar_95 = float(worst_r.mean())

    metrics = {
        "CAGR": total_cagr,
        "Volatility": vol,
        "Sharpe (rf-excess)": sharpe,
        "Sortino": sortino,
        "Max Drawdown": mdd,
        "Calmar": calmar,
        "CDaR 95%": cdar_95,
        "CVaR 95%": cvar_95,
    }

    if benchmark_returns is not None:
        b = benchmark_returns.reindex(r.index).astype(float)
        up, down = b > 0, b < 0
        metrics["Upside Capture"] = (
            float(r[up].mean() / b[up].mean()) if up.any() and abs(float(b[up].mean())) > 1e-15 else float("nan")
        )
        metrics["Downside Capture"] = (
            float(r[down].mean() / b[down].mean())
            if down.any() and abs(float(b[down].mean())) > 1e-15
            else float("nan")
        )
        metrics.update(spanning_regression(r, b, f))
    return metrics


# ---------------------------------------------------------------------------
# Scenario comparison & sub-period tables
# ---------------------------------------------------------------------------

def scenario_table(
    scenarios: Mapping[str, pd.Series],
    rf: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """Rows = metrics, columns = named scenarios (incl. benchmark)."""
    metrics_by_scenario: dict[str, dict[str, float]] = {}
    for name, rets in scenarios.items():
        is_bench = name == BENCHMARK_LABEL
        metrics_by_scenario[name] = performance_metrics(
            rets,
            rf,
            benchmark_returns=benchmark_returns if not is_bench else None,
        )
    return pd.DataFrame(metrics_by_scenario)


def subperiod_table(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: pd.Series,
) -> pd.DataFrame:
    """Performance across historical stress windows."""
    rows = []
    for label, start, end in SUBPERIODS:
        mask = np.asarray(
            (strategy_returns.index >= pd.Timestamp(start))
            & (strategy_returns.index <= pd.Timestamp(end))
        )
        if int(mask.sum()) < 63:
            continue
        seg_s = strategy_returns[mask]
        seg_b = benchmark_returns[mask]
        seg_f = rf[mask]
        rows.append(
            {
                "Window": label,
                "Sessions": int(mask.sum()),
                "Strategy CAGR": cagr(seg_s),
                "SPY CAGR": cagr(seg_b),
                "Strategy MDD": max_drawdown(seg_s),
                "SPY MDD": max_drawdown(seg_b),
                "Strategy Sharpe": _segment_sharpe(seg_s, seg_f),
                "SPY Sharpe": _segment_sharpe(seg_b, seg_f),
            }
        )
    return pd.DataFrame(rows).set_index("Window")


def yearly_table(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """Calendar-year compound returns: strategy vs SPY."""
    years = sorted(set(strategy_returns.index.year) | set(benchmark_returns.index.year))
    rows = {
        y: {
            "AlphaShield": _compound(yearly_returns(strategy_returns).get(y, np.nan)),
            "SPY": _compound(yearly_returns(benchmark_returns).get(y, np.nan)),
        }
        for y in years
    }
    table = pd.DataFrame(rows).T
    table.index.name = "Year"
    table["Excess"] = table["AlphaShield"] - table["SPY"]
    return table


def _compound(year_return: float) -> float:
    return float(year_return) if np.isfinite(year_return) else float("nan")


def _segment_sharpe(segment: pd.Series, rf_segment: pd.Series) -> float:
    excess = segment.astype(float) - rf_segment.astype(float)
    std = float(excess.std(ddof=1))
    if std <= 1e-12:
        return float("nan")
    return float(excess.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


# ---------------------------------------------------------------------------
# Invariant verification report
# ---------------------------------------------------------------------------

def verify_invariants(
    ledger,
    canary_executed: pd.Series,
    lag: int,
) -> dict[str, str]:
    """Audit checks on the executed ledger (canary, cap, lag, frictions).

    ``canary_executed``: boolean Series, the canary state that governed each
    execution session (i.e. the signal canary shifted by ``lag``).
    """
    executed = ledger.weights

    canary_days = canary_executed.reindex(executed.index).fillna(False).astype(bool)
    unsafe_days = executed.index[~canary_days]
    leak = float(executed.loc[unsafe_days].to_numpy().sum()) if len(unsafe_days) else 0.0

    cap_ok = bool((executed.to_numpy() <= 0.20 + 1e-9).all())

    return {
        "Strict TIP canary (<=0 -> 0% risky)": "PASS" if abs(leak) < 1e-9 else f"FAIL leak={leak:.2e}",
        "Canary-off sessions enforced": f"{len(unsafe_days)} sessions",
        "Equal-cap 20% per asset": "PASS" if cap_ok else "FAIL",
        f"Strict T+{lag} execution mapping": "PASS (asserted in portfolio engine)",
        "Self-financing + friction identity": "PASS (asserted in portfolio engine)",
    }
