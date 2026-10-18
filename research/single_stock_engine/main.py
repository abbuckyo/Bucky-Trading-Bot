# -*- coding: utf-8 -*-
"""CLI runner for the Cross-Sectional Single-Stock Momentum Engine.

Supports:
  --self-test   Run offline deterministic invariant unit tests
  --run         Execute end-to-end historical backtest & institutional audit
  --show        Display interactive performance chart window
  --refresh     Force sequential re-download of market data from Yahoo Finance
  --start       Override backtest start date (YYYY-MM-DD)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.single_stock_engine.audit_suite import (  # noqa: E402
    BENCHMARK_LABEL,
    drawdown_curve,
    nav_curve,
    performance_metrics,
    rf_series,
    scenario_table,
    spanning_regression,
    verify_invariants_report,
    yearly_table,
)
from research.single_stock_engine.config import EngineConfig  # noqa: E402
from research.single_stock_engine.data_loader import DataLoader  # noqa: E402
from research.single_stock_engine.portfolio_engine import (  # noqa: E402
    PortfolioEngine,
    build_signal_weights,
)
from research.single_stock_engine.ranking_engine import RankingEngine  # noqa: E402
from research.single_stock_engine.regime_guard import RegimeGuard  # noqa: E402
from research.single_stock_engine.risk_guards import RiskGuards  # noqa: E402
from research.single_stock_engine.test_invariants import run_all_tests  # noqa: E402

_DIVIDER = "=" * 82


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-Sectional Single-Stock Momentum Engine (Research Module)"
    )
    parser.add_argument("--self-test", action="store_true", help="Run offline invariant unit tests")
    parser.add_argument("--run", action="store_true", help="Run full backtest & institutional audit")
    parser.add_argument("--show", action="store_true", help="Show interactive performance plots")
    parser.add_argument("--refresh", action="store_true", help="Force sequential Yahoo Finance refresh")
    parser.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
    return parser.parse_args()


def _try_load_v81_returns(cfg: EngineConfig, target_index: pd.DatetimeIndex) -> pd.Series | None:
    """Compute AlphaShield V8 ETF sleeve returns cached under research/single_stock_engine/cache/."""
    try:
        import time
        import yfinance as yf
        from research.alphashield_v82.config import CANARY_TICKER, CORE_UNIVERSE, PortfolioConfig, SignalConfig
        from research.alphashield_v82.data_engine import MarketData
        from research.alphashield_v82.portfolio_engine import PortfolioEngine as V82PortfolioEngine
        from research.alphashield_v82.signal_engine import SignalEngine

        v8_cache_file = Path(cfg.cache_dir) / "v8_etf_ohlc.parquet"
        v8_tickers = list(CORE_UNIVERSE) + [CANARY_TICKER]
        snap: pd.DataFrame | None = None
        if v8_cache_file.exists() and not cfg.refresh:
            snap = pd.read_parquet(v8_cache_file)

        if snap is None:
            frames: dict[str, pd.DataFrame] = {}
            end_dt = str(pd.Timestamp.now(tz="America/New_York").date())
            for sym in v8_tickers:
                raw = yf.download(
                    sym,
                    start=cfg.data_start,
                    end=end_dt,
                    auto_adjust=True,
                    actions=False,
                    progress=False,
                    threads=False,
                    timeout=30,
                )
                frames[sym] = DataLoader.validate_raw_ohlc(sym, raw)
                time.sleep(0.10)
            snap = pd.concat(frames, names=("Ticker", "Field")).sort_index()
            v8_cache_file.parent.mkdir(parents=True, exist_ok=True)
            snap.to_parquet(v8_cache_file)

        closes_all = pd.DataFrame(
            {t: snap.xs(t, level="Ticker")["Close"] for t in v8_tickers}
        ).dropna(axis=0, how="any")
        opens_all = pd.DataFrame(
            {t: snap.xs(t, level="Ticker")["Open"] for t in CORE_UNIVERSE}
        ).reindex(closes_all.index).dropna(axis=0, how="any")
        closes_all = closes_all.reindex(opens_all.index)

        mkt = MarketData(closes=closes_all, opens=opens_all, risk_assets=CORE_UNIVERSE)
        sig = SignalEngine(mkt, SignalConfig()).compute()
        led = V82PortfolioEngine(
            mkt.opens[list(CORE_UNIVERSE)],
            mkt.closes[list(CORE_UNIVERSE)],
            PortfolioConfig(),
        ).run(sig.weights, start=target_index[0])
        return led.returns.reindex(target_index).dropna().rename("AlphaShield_V8")
    except Exception:  # noqa: BLE001
        return None


def _fmt_df(df: pd.DataFrame, pct_rows: tuple[str, ...] = ()) -> str:
    out = df.astype(object)
    for row in out.index:
        if row in pct_rows:
            out.loc[row] = [f"{100.0 * float(v): .2f}%" if pd.notna(v) else "   nan" for v in df.loc[row]]
        else:
            out.loc[row] = [f"{float(v): .4f}" if pd.notna(v) else "   nan" for v in df.loc[row]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        return out.to_string()


def run_backtest(cfg: EngineConfig, show: bool = False) -> None:
    print(_DIVIDER)
    print("SINGLE-STOCK CROSS-SECTIONAL MOMENTUM ENGINE — INSTITUTIONAL AUDIT")
    print(_DIVIDER)

    # 1. Data ingestion
    bundle = DataLoader(cfg).load(fetch_earnings=True)
    print(
        f"[1/5] Data      : {len(bundle.calendar)} aligned sessions "
        f"({bundle.calendar[0].date()} -> {bundle.calendar[-1].date()}), "
        f"{len(bundle.tickers)} Large-Cap equities + {cfg.benchmark}"
    )

    # 2. Macro & Earnings Regime Guard
    regime = RegimeGuard(
        spy_ema_window=cfg.spy_ema_window,
        earnings_blackout_days=cfg.earnings_blackout_days,
    ).evaluate(
        spy_close=bundle.spy_close,
        tickers=bundle.tickers,
        earnings_dates=bundle.earnings_dates,
    )
    print(
        f"[2/5] Regime    : SPY > EMA{cfg.spy_ema_window} gate + "
        f"{cfg.earnings_blackout_days}-day Earnings Blackout filter"
    )

    # 3. Cross-Sectional Volatility-Adjusted Momentum Ranking
    ranking = RankingEngine(
        lookback_long=cfg.mom_lookback_long,
        lookback_skip=cfg.mom_lookback_skip,
        vol_lookback=cfg.vol_lookback,
        vol_floor=cfg.vol_floor,
        top_quantile=cfg.top_quantile,
        max_active_holdings=cfg.max_active_holdings,
    ).compute(bundle.closes)
    print(
        f"[3/5] Ranking   : r_{{12M-1M}} / max(vol_60D, {cfg.vol_floor:.2f}), "
        f"weekly Monday rebalance, top {cfg.top_quantile:.0%} pool"
    )

    # 4. Individual Trend & Chandelier Trailing Stop Guards
    guards = RiskGuards(
        ema_window=cfg.stock_ema_window,
        atr_window=cfg.atr_window,
        chandelier_high_window=cfg.chandelier_high_window,
        chandelier_atr_mult=cfg.chandelier_atr_mult,
        earnings_tight_atr_mult=cfg.earnings_tight_atr_mult,
    ).evaluate(
        highs=bundle.highs,
        lows=bundle.lows,
        closes=bundle.closes,
        earnings_blackout=regime.earnings_blackout,
    )

    # 5. Fixed Fractional Risk Parity Sizing & T+1 Execution Ledger
    signal_weights = build_signal_weights(
        closes=bundle.closes,
        ranking=ranking,
        regime=regime,
        guards=guards,
        config=cfg,
    )
    engine = PortfolioEngine(bundle.opens, bundle.closes, cfg)
    bt_start = cfg.backtest_start
    base_ledger = engine.run(signal_weights, start=bt_start)
    stress_10bps = engine.run(signal_weights, start=bt_start, cost_bps=2.0 * cfg.cost_bps)
    stress_t2 = engine.run(signal_weights, start=bt_start, execution_lag=cfg.execution_lag + 1)

    spy_returns = (
        bundle.spy_close.pct_change(fill_method=None)
        .reindex(base_ledger.returns.index)
        .fillna(0.0)
    )
    rf = rf_series(base_ledger.returns.index, cfg.cash_yield_annual)
    v81_returns = _try_load_v81_returns(cfg, base_ledger.returns.index)

    print(
        f"[4/5] Ledger    : {len(base_ledger.returns)} sessions "
        f"({base_ledger.returns.index[0].date()} -> {base_ledger.returns.index[-1].date()}), "
        f"T+{cfg.execution_lag} lag, {cfg.cost_bps:.1f} bps cost, {cfg.cash_yield_annual:.1%} cash park"
    )

    scenarios: dict[str, pd.Series] = {
        "SingleStock (Base 5bps T+1)": base_ledger.returns,
        "Stress (10bps 2x Cost)": stress_10bps.returns,
        "Stress (T+2 Lag)": stress_t2.returns,
    }
    if v81_returns is not None and len(v81_returns) == len(base_ledger.returns):
        scenarios["AlphaShield V8 (ETF)"] = v81_returns
    scenarios[BENCHMARK_LABEL] = spy_returns

    table = scenario_table(scenarios, rf, spy_returns, v81_returns=v81_returns)
    pct_rows = ("CAGR", "Volatility", "Max Drawdown", "CDaR 95%", "CVaR 95%", "alpha_annual", "v81_alpha_annual")
    print("\n--- [5/5] Institutional Performance Summary ---")
    print(_fmt_df(table, pct_rows=pct_rows))

    print("\n--- Invariant Verification Report ---")
    for check_name, status in verify_invariants_report(
        base_ledger,
        lag=cfg.execution_lag,
        max_weight_cap=cfg.max_weight_per_stock,
        max_active_holdings=cfg.max_active_holdings,
    ).items():
        print(f"  {check_name:<50} {status}")

    base_metrics = performance_metrics(base_ledger.returns, rf, spy_returns, v81_returns=v81_returns)
    print("\n--- Spanning Regression vs SPY (Newey-West HAC) ---")
    print(
        f"  Annualized Alpha : {base_metrics['alpha_annual']:+.2%}   "
        f"(HAC t-stat = {base_metrics['alpha_t_stat']:.2f}, p = {base_metrics['alpha_p_value']:.4f})"
    )
    print(
        f"  Market Beta (SPY): {base_metrics['beta']:.3f}    "
        f"Correlation = {base_metrics['correlation']:.3f}"
    )

    if v81_returns is not None and "v81_alpha_annual" in base_metrics:
        print("\n--- Spanning Regression vs AlphaShield V8 (Newey-West HAC) ---")
        print(
            f"  Annualized Alpha : {base_metrics['v81_alpha_annual']:+.2%}   "
            f"(HAC t-stat = {base_metrics['v81_alpha_t_stat']:.2f}, p = {base_metrics['v81_alpha_p_value']:.4f})"
        )
        print(
            f"  Beta vs V8       : {base_metrics['v81_beta']:.3f}    "
            f"Correlation = {base_metrics['v81_correlation']:.3f}"
        )
        joint = spanning_regression(
            base_ledger.returns,
            pd.DataFrame({"SPY": spy_returns, "V8": v81_returns}).dropna(),
            rf,
        )
        print(
            f"  Joint (SPY + V8) : Alpha = {joint['alpha_annual']:+.2%} "
            f"(t = {joint['alpha_t_stat']:.2f}, beta_SPY = {joint['beta_SPY']:.3f}, beta_V8 = {joint['beta_V8']:.3f})"
        )

    print("\n--- Portfolio Exposure & Turnover Summary ---")
    print(f"  Avg Active Holdings   : {base_ledger.active_count.mean():.2f} stocks")
    print(f"  Max Single Weight     : {base_ledger.weights.to_numpy().max():.2%}")
    print(f"  Avg Risky Exposure    : {base_ledger.risky_exposure.mean():.2%}")
    print(f"  Avg Cash Parked       : {base_ledger.cash_weight.mean():.2%}")
    print(f"  Annualized 1-Way Turn : {252.0 * base_ledger.turnover.mean():.2f}x")

    print("\n--- Calendar-Year Compound Returns ---")
    yr_df = yearly_table(base_ledger.returns, spy_returns)
    with pd.option_context("display.width", 120):
        print(yr_df.map(lambda v: f"{100.0 * float(v): .2f}%" if pd.notna(v) else "   nan").to_string())

    chart_path = Path(cfg.output_dir) / "single_stock_momentum_audit.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    _save_chart(base_ledger, stress_10bps, stress_t2, spy_returns, chart_path, show=show)
    print(f"\nAudit chart saved -> {chart_path}")
    print(_DIVIDER)


def _save_chart(
    base_ledger,
    stress_10bps,
    stress_t2,
    spy_returns: pd.Series,
    chart_path: Path,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 1.0]}
    )

    axes[0].plot(nav_curve(base_ledger.returns), label="Single-Stock Momentum (5bps, T+1)", lw=1.8)
    axes[0].plot(nav_curve(stress_10bps.returns), label="Stress: 10 bps (2x cost)", lw=1.0, alpha=0.85)
    axes[0].plot(nav_curve(stress_t2.returns), label="Stress: T+2 lag", lw=1.0, alpha=0.85)
    axes[0].plot(nav_curve(spy_returns), label="SPY Buy & Hold", lw=1.2, ls="--", color="grey")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Growth of $1 (log)")
    axes[0].set_title("Cross-Sectional Single-Stock Momentum Engine — Strict T+1 Audit")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    axes[1].plot(drawdown_curve(base_ledger.returns), label="Strategy Drawdown", lw=1.2)
    axes[1].plot(drawdown_curve(spy_returns), label="SPY Drawdown", lw=1.0, alpha=0.8, color="grey")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.25)

    axes[2].plot(base_ledger.risky_exposure, label="Risky Exposure", lw=1.0)
    axes[2].set_ylabel("Allocated Weight")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    # Default to running self-tests if neither --self-test nor --run was specified
    if not args.self_test and not args.run:
        args.self_test = True

    if args.self_test:
        ok = run_all_tests()
        if not ok:
            sys.exit(1)

    if args.run:
        cfg = EngineConfig(
            backtest_start=args.start or EngineConfig.backtest_start,
            refresh=args.refresh,
        )
        run_backtest(cfg, show=args.show)


if __name__ == "__main__":
    main()
