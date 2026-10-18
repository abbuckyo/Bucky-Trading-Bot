# -*- coding: utf-8 -*-
"""Main entry point — end-to-end AlphaShield backtest & institutional audit.

Runs the configured window (default 2015 -> present):
  * loads or builds the cached market snapshot (5 core ETFs + TIP),
  * computes signals with the strict TIP canary + sequential tranche ladder,
  * simulates the strict T+1 self-financing ledger (5 bps, 2% cash park),
  * stress-tests 2x cost (10 bps) and +1 execution lag (T+2),
  * prints institutional audit tables and saves the performance charts.

Usage:
  python -m alphashield.main [--refresh] [--show] [--start YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .audit_suite import (  # noqa: E402
    BENCHMARK_LABEL,
    nav_curve,
    performance_metrics,
    scenario_table,
    subperiod_table,
    verify_invariants,
    yearly_table,
)
from .config import BENCHMARK_TICKER, DataConfig, PortfolioConfig, SignalConfig  # noqa: E402
from .data_engine import DataEngine  # noqa: E402
from .portfolio_engine import PortfolioEngine  # noqa: E402
from .signal_engine import SignalEngine  # noqa: E402

_LINE = "=" * 78


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaShield institutional backtest")
    parser.add_argument("--refresh", action="store_true", help="re-download market data")
    parser.add_argument("--show", action="store_true", help="open interactive chart windows")
    parser.add_argument("--start", default=None, help="backtest start (ISO date)")
    return parser.parse_args()


def _fmt_table(df: pd.DataFrame, percent: tuple[str, ...] = ()) -> str:
    out = df.copy()
    for row in percent:
        if row in out.index:
            out.loc[row] = out.loc[row].map(lambda v: f"{100 * v: .2f}")
    with pd.option_context("display.width", 200, "display.max_columns", 32):
        return out.to_string(float_format=lambda v: f"{float(v): .4f}")


def main() -> None:
    args = _parse_args()
    data_cfg = DataConfig(backtest_start=args.start or DataConfig.backtest_start, refresh=args.refresh)
    signal_cfg = SignalConfig()
    portfolio_cfg = PortfolioConfig()

    print(_LINE)
    print("ALPHASHIELD CORE — INSTITUTIONAL BACKTEST & AUDIT")
    print(_LINE)

    # [1] Data ----------------------------------------------------------------
    market = DataEngine(data_cfg).load()
    print(
        f"[1/4] data  : {len(market.calendar)} aligned sessions "
        f"{market.calendar[0].date()} -> {market.calendar[-1].date()}"
    )

    # [2] Signals ---------------------------------------------------------------
    signals = SignalEngine(market, signal_cfg).compute()
    print("[2/4] signal: strict TIP 13612 canary + sequential tranche ladder")

    engine = PortfolioEngine(
        market.opens[signal_cfg.universe],
        market.closes[list(signal_cfg.universe)],
        portfolio_cfg,
    )

    # [3] Ledger + stress scenarios ----------------------------------------------
    start = data_cfg.backtest_start
    base = engine.run(signals.weights, start=start)
    stressed_cost = engine.run(signals.weights, start=start, cost_bps=2 * portfolio_cfg.cost_bps)
    stressed_lag = engine.run(signals.weights, start=start, execution_lag=portfolio_cfg.execution_lag + 1)

    benchmark = (
        market.closes[BENCHMARK_TICKER]
        .pct_change(fill_method=None)
        .loc[base.returns.index]
        .fillna(0.0)
    )
    from .audit_suite import rf_series

    rf = rf_series(base.returns.index, portfolio_cfg.cash_yield_annual)

    print(
        f"[3/4] ledger: {len(base.returns)} sessions "
        f"{base.returns.index[0].date()} -> {base.returns.index[-1].date()}; "
        f"friction {portfolio_cfg.cost_bps:.0f} bps, cash park "
        f"{portfolio_cfg.cash_yield_annual:.1%} p.a."
    )

    # [4] Institutional audit --------------------------------------------------
    print("[4/4] audit : metrics, stress deltas, spanning regression")
    scenarios = {
        "AlphaShield (Base)": base.returns,
        "AlphaShield (10bps)": stressed_cost.returns,
        "AlphaShield (T+2)": stressed_lag.returns,
        BENCHMARK_LABEL: benchmark,
    }
    table = scenario_table(scenarios, rf, benchmark)
    pct_rows = ("CAGR", "Volatility", "Max Drawdown", "CDaR 95%", "CVaR 95%")
    print("\n--- Performance summary (5bps / T+1 vs stresses vs SPY) ---")
    print(_fmt_table(table, pct_rows))

    stress_deltas = pd.DataFrame(
        {
            "2x cost (10bps)": {
                "dCAGR": table["AlphaShield (10bps)"]["CAGR"] - table["AlphaShield (Base)"]["CAGR"],
                "dSharpe": table["AlphaShield (10bps)"]["Sharpe (rf-excess)"]
                - table["AlphaShield (Base)"]["Sharpe (rf-excess)"],
                "dMDD": table["AlphaShield (10bps)"]["Max Drawdown"]
                - table["AlphaShield (Base)"]["Max Drawdown"],
            },
            "+1 lag (T+2)": {
                "dCAGR": table["AlphaShield (T+2)"]["CAGR"] - table["AlphaShield (Base)"]["CAGR"],
                "dSharpe": table["AlphaShield (T+2)"]["Sharpe (rf-excess)"]
                - table["AlphaShield (Base)"]["Sharpe (rf-excess)"],
                "dMDD": table["AlphaShield (T+2)"]["Max Drawdown"]
                - table["AlphaShield (Base)"]["Max Drawdown"],
            },
        }
    ).T
    print("\n--- Stress deltas vs base ---")
    print(_fmt_table(stress_deltas, ("dCAGR", "dMDD")))

    canary_executed = signals.canary_safe.shift(portfolio_cfg.execution_lag)
    print("\n--- Invariant checks ---")
    for check, verdict in verify_invariants(base, canary_executed, portfolio_cfg.execution_lag).items():
        print(f"  {check:<52} {verdict}")

    metrics = performance_metrics(base.returns, rf, benchmark)
    print("\n--- Spanning regression vs SPY (Newey-West HAC) ---")
    print(
        f"  alpha (annual) = {metrics['alpha_annual']:.4f}   "
        f"HAC t = {metrics['alpha_t_stat']:.2f}   p = {metrics['alpha_p_value']:.4f}"
    )
    print(
        f"  beta = {metrics['beta']:.4f}   correlation = {metrics['correlation']:.4f}"
        f"   (Bartlett lags = {metrics['hac_lags']})"
    )

    leads = ("Base", "10bps", "T+2")
    flows = pd.DataFrame(
        {
            "Avg daily turnover": [base.turnover.mean(), stressed_cost.turnover.mean(), stressed_lag.turnover.mean()],
            "2-way turnover p.a.": [252.0 * base.turnover.mean(), 252.0 * stressed_cost.turnover.mean(), 252.0 * stressed_lag.turnover.mean()],
            "Avg risky exposure": [base.risky_exposure.mean(), stressed_cost.risky_exposure.mean(), stressed_lag.risky_exposure.mean()],
            "Avg cash parked": [base.cash_weight.mean(), stressed_cost.cash_weight.mean(), stressed_lag.cash_weight.mean()],
        },
        index=leads,
    )
    print("\n--- Turnover, frictions & cash park ---")
    print(_fmt_table(flows, ()))

    print("\n--- Yearly compound returns ---")
    print(_fmt_table(yearly_table(base.returns, benchmark), ()))

    print("\n--- Sub-period stress windows ---")
    print(_fmt_table(subperiod_table(base.returns, benchmark, rf), ()))

    chart_path = Path("output") / "alphashield_performance.png"
    chart_path.parent.mkdir(exist_ok=True)
    _plot(base, stressed_cost, stressed_lag, benchmark, chart_path, args.show)
    print(f"\nCharts saved: {chart_path}")
    print(_LINE)


def _plot(
    base,
    stressed_cost,
    stressed_lag,
    benchmark: pd.Series,
    path: Path,
    show: bool,
) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 11), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 1.0]}
    )

    axes[0].plot(nav_curve(base.returns), label="AlphaShield (5bps, T+1, park 2%)", lw=1.8)
    axes[0].plot(nav_curve(stressed_cost.returns), label="Stress: 10 bps (2x cost)", lw=1.0, alpha=0.85)
    axes[0].plot(nav_curve(stressed_lag.returns), label="Stress: T+2 lag", lw=1.0, alpha=0.85)
    axes[0].plot(nav_curve(benchmark), label="SPY buy & hold", lw=1.2, ls="--", color="grey")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Growth of $1 (log)")
    axes[0].set_title("AlphaShield cumulative performance & drawdown — strict T+1 audit")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    dd = lambda rets: nav_curve(rets) / nav_curve(rets).cummax() - 1.0  # noqa: E731
    axes[1].plot(dd(base.returns), label="AlphaShield drawdown", lw=1.2)
    axes[1].plot(dd(benchmark), label="SPY drawdown", lw=1.0, alpha=0.8, color="grey")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.25)

    axes[2].plot(base.risky_exposure, lw=0.9)
    axes[2].set_ylabel("Risky exposure")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
