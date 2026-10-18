# -*- coding: utf-8 -*-
"""CLI runner for the Regime-Switched G10 Forex Quantitative Engine (Engine 3).

Supports:
  --self-test   Run offline deterministic unit tests
  --run         Execute 2017–2026 historical backtest & institutional audit
  --show        Display interactive chart window
  --refresh     Force sequential re-download from Yahoo Finance
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

from research.forex_engine.audit_suite import (  # noqa: E402
    drawdown_curve,
    nav_curve,
    performance_metrics,
    rf_series,
    scenario_table,
    verify_forex_invariants,
    yearly_table,
)
from research.forex_engine.config import ForexConfig  # noqa: E402
from research.forex_engine.data_loader import ForexDataLoader  # noqa: E402
from research.forex_engine.portfolio_engine import (  # noqa: E402
    ForexPortfolioEngine,
    compute_target_weights,
)
from research.forex_engine.regime_filter import ForexRegimeFilter  # noqa: E402
from research.forex_engine.signal_engine import ForexSignalEngine  # noqa: E402
from research.forex_engine.test_invariants import run_all_tests  # noqa: E402

_DIVIDER = "=" * 86


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regime-Switched G10 Forex Quantitative Engine (Engine 3)"
    )
    parser.add_argument("--self-test", action="store_true", help="Run offline unit tests")
    parser.add_argument("--run", action="store_true", help="Run full 2017-2026 backtest & audit")
    parser.add_argument("--show", action="store_true", help="Show interactive performance plot")
    parser.add_argument("--refresh", action="store_true", help="Force sequential yfinance refresh")
    parser.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
    return parser.parse_args()


def _try_load_v82_returns(cfg: ForexConfig, target_index: pd.DatetimeIndex) -> pd.Series | None:
    """Compute AlphaShield V8.2 Core ETF returns cached under research/forex_engine/cache/."""
    try:
        import time
        import yfinance as yf
        from research.alphashield_v82.config import (
            CANARY_TICKER,
            CORE_UNIVERSE,
            PortfolioConfig as V82PortfolioConfig,
            SignalConfig as V82SignalConfig,
        )
        from research.alphashield_v82.data_engine import MarketData as V82MarketData
        from research.alphashield_v82.portfolio_engine import PortfolioEngine as V82PortfolioEngine
        from research.alphashield_v82.signal_engine import SignalEngine as V82SignalEngine

        v8_cache = Path(cfg.cache_dir) / "v82_etf_reference.parquet"
        v8_syms = list(CORE_UNIVERSE) + [CANARY_TICKER]
        snap: pd.DataFrame | None = None
        if v8_cache.exists() and not cfg.refresh:
            snap = pd.read_parquet(v8_cache)

        if snap is None:
            frames: dict[str, pd.DataFrame] = {}
            end_dt = str(pd.Timestamp.now(tz="America/New_York").date())
            for sym in v8_syms:
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
                frames[sym] = ForexDataLoader.validate_raw_ohlc(sym, raw)[["Open", "Close"]]
                time.sleep(0.10)
            snap = pd.concat(frames, names=("Ticker", "Field")).sort_index()
            v8_cache.parent.mkdir(parents=True, exist_ok=True)
            snap.to_parquet(v8_cache)

        closes_all = pd.DataFrame(
            {t: snap.xs(t, level="Ticker")["Close"] for t in v8_syms}
        ).dropna(axis=0, how="any")
        opens_all = pd.DataFrame(
            {t: snap.xs(t, level="Ticker")["Open"] for t in CORE_UNIVERSE}
        ).reindex(closes_all.index).dropna(axis=0, how="any")
        closes_all = closes_all.reindex(opens_all.index)

        mkt = V82MarketData(closes=closes_all, opens=opens_all, risk_assets=CORE_UNIVERSE)
        sig = V82SignalEngine(mkt, V82SignalConfig()).compute()
        led = V82PortfolioEngine(
            mkt.opens[list(CORE_UNIVERSE)],
            mkt.closes[list(CORE_UNIVERSE)],
            V82PortfolioConfig(),
        ).run(sig.weights, start=target_index[0])
        return led.returns.reindex(target_index).fillna(0.0).rename("AlphaShield V8.2")
    except Exception:  # noqa: BLE001
        return None


def _fmt_df(df: pd.DataFrame, pct_rows: tuple[str, ...] = ()) -> str:
    out = df.astype(object)
    for row in out.index:
        if row in pct_rows:
            out.loc[row] = [
                f"{100.0 * float(v): .2f}%" if pd.notna(v) else "   nan" for v in df.loc[row]
            ]
        else:
            out.loc[row] = [
                f"{float(v): .4f}" if pd.notna(v) else "   nan" for v in df.loc[row]
            ]
    with pd.option_context("display.width", 220, "display.max_columns", 12):
        return out.to_string()


def run_backtest(cfg: ForexConfig, show: bool = False) -> None:
    print(_DIVIDER)
    print("REGIME-SWITCHED G10 FOREX QUANTITATIVE ENGINE (ENGINE 3) — INSTITUTIONAL AUDIT")
    print(_DIVIDER)

    # 1. Data Ingestion
    bundle = ForexDataLoader(cfg).load()
    print(
        f"[1/5] Data      : {len(bundle.calendar)} aligned sessions "
        f"({bundle.calendar[0].date()} -> {bundle.calendar[-1].date()}), "
        f"7 G10 FX pairs + {cfg.dollar_index} + {cfg.equity_benchmark}"
    )

    # 2. Orthogonal Regime Detection (Wilder ADX_14 + Rolling Hurst H_100)
    regime = ForexRegimeFilter(
        adx_period=cfg.adx_period,
        hurst_window=cfg.hurst_window,
        adx_trend_threshold=cfg.adx_trend_threshold,
        hurst_trend_threshold=cfg.hurst_trend_threshold,
        adx_choppy_threshold=cfg.adx_choppy_threshold,
        hurst_choppy_threshold=cfg.hurst_choppy_threshold,
    ).evaluate(bundle.highs, bundle.lows, bundle.closes)
    print(
        f"[2/5] Regime    : Wilder ADX({cfg.adx_period}) + Rolling Hurst({cfg.hurst_window}) "
        f"[Trend: ADX>{cfg.adx_trend_threshold:.0f} or H>{cfg.hurst_trend_threshold:.2f} | "
        f"Choppy: ADX<{cfg.adx_choppy_threshold:.0f} & H<{cfg.hurst_choppy_threshold:.2f}]"
    )

    # 3. Signal & Carry Confirmation + ATR Trailing Stops
    signals = ForexSignalEngine(cfg).compute(
        bundle.highs, bundle.lows, bundle.closes, regime
    )
    target_weights = compute_target_weights(
        direction_signal=signals.direction_signal,
        vol20_annual=signals.vol20_annual,
        target_pair_vol=cfg.target_pair_vol,
        max_pair_weight=cfg.max_pair_weight,
        max_gross_leverage=cfg.max_gross_leverage,
    )
    print(
        f"[3/5] Signals   : Symmetric Long/Short EMA({cfg.ema_fast}/{cfg.ema_slow}) + DI(14) "
        f"+ 63d/126d Mom/Carry + {cfg.trailing_stop_atr_mult:.1f}x ATR({cfg.atr_window}) Stop"
    )

    # 4. Execution Ledger & Stress Scenarios
    engine = ForexPortfolioEngine(bundle.closes, cfg, opens=bundle.opens)
    bt_start = cfg.backtest_start
    base_ledger = engine.run(target_weights, start=bt_start)
    stress_6bps = engine.run(target_weights, start=bt_start, cost_bps=2.0 * cfg.cost_bps)
    stress_t2 = engine.run(target_weights, start=bt_start, execution_lag=cfg.execution_lag + 1)

    idx = base_ledger.returns.index
    spy_returns = bundle.spy_close.pct_change(fill_method=None).reindex(idx).fillna(0.0)
    dxy_returns = bundle.dxy_close.pct_change(fill_method=None).reindex(idx).fillna(0.0)
    rf = rf_series(idx, cfg.cash_yield_annual)
    v82_returns = _try_load_v82_returns(cfg, idx)

    print(
        f"[4/5] Ledger    : {len(idx)} sessions ({idx[0].date()} -> {idx[-1].date()}), "
        f"T+{cfg.execution_lag} lag, {cfg.cost_bps:.1f} bps cost, {cfg.cash_yield_annual:.1%} cash park"
    )

    scenarios: dict[str, pd.Series] = {
        "G10 Forex (Base 3bps T+1)": base_ledger.returns,
        "Stress (6bps 2x Cost)": stress_6bps.returns,
        "Stress (T+2 Lag)": stress_t2.returns,
    }
    if v82_returns is not None:
        scenarios["AlphaShield V8.2 (ETF)"] = v82_returns
    scenarios["DX-Y.NYB (USD Index)"] = dxy_returns
    scenarios["SPY B&H"] = spy_returns

    table = scenario_table(scenarios, rf, spy_returns=spy_returns, v82_returns=v82_returns)
    pct_rows = ("CAGR", "Volatility", "Max Drawdown", "CDaR 95%", "CVaR 95%", "alpha_vs_spy", "alpha_vs_v82")
    print("\n--- [5/5] Institutional Performance Summary (2017–2026) ---")
    print(_fmt_df(table, pct_rows=pct_rows))

    print("\n--- Invariant Verification Report ---")
    for check_name, verdict in verify_forex_invariants(
        base_ledger,
        lag=cfg.execution_lag,
        max_pair_weight=cfg.max_pair_weight,
        max_gross_leverage=cfg.max_gross_leverage,
    ).items():
        print(f"  {check_name:<52} {verdict}")

    m = performance_metrics(base_ledger.returns, rf, spy_returns=spy_returns, v82_returns=v82_returns)
    corr_dxy = float(base_ledger.returns.corr(dxy_returns))
    print("\n--- Correlation & Spanning Alpha Checks ---")
    print(
        f"  vs SPY              : Corr = {m['corr_vs_spy']:+.4f} | "
        f"Beta = {m['beta_vs_spy']:+.4f} | "
        f"Spanning Alpha = {m['alpha_vs_spy']:+.2%} (HAC t = {m['t_stat_vs_spy']:.2f}, p = {m['p_val_vs_spy']:.4f})"
    )
    if "corr_vs_v82" in m:
        print(
            f"  vs AlphaShield V8.2 : Corr = {m['corr_vs_v82']:+.4f} | "
            f"Beta = {m['beta_vs_v82']:+.4f} | "
            f"Spanning Alpha = {m['alpha_vs_v82']:+.2%} (HAC t = {m['t_stat_vs_v82']:.2f}, p = {m['p_val_vs_v82']:.4f})"
        )
    print(f"  vs DX-Y.NYB (DXY)   : Corr = {corr_dxy:+.4f}")

    print("\n--- Exposure & Regime Activity Summary ---")
    print(f"  Avg Gross Exposure  : {base_ledger.gross_exposure.mean():.2%}")
    print(f"  Max Gross Exposure  : {base_ledger.gross_exposure.max():.2%}")
    print(f"  Avg Net Exposure    : {base_ledger.net_exposure.mean():+.2%}")
    print(f"  Avg Cash Weight     : {base_ledger.cash_weight.mean():.2%}")
    print(f"  Trending Regime %   : {regime.is_trending.loc[idx].to_numpy().mean():.2%}")
    print(f"  Choppy Regime %     : {regime.is_choppy.loc[idx].to_numpy().mean():.2%}")
    print(f"  Annualized Turnover : {252.0 * base_ledger.turnover.mean():.2f}x")

    print("\n--- Calendar-Year Compound Returns ---")
    yr_df = yearly_table(base_ledger.returns, spy_returns, v82_returns=v82_returns)
    with pd.option_context("display.width", 140):
        print(yr_df.map(lambda v: f"{100.0 * float(v):+.2f}%" if pd.notna(v) else "   nan").to_string())

    chart_path = Path(cfg.output_dir) / "forex_engine_audit.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    _save_chart(base_ledger, stress_6bps, stress_t2, spy_returns, v82_returns, chart_path, show=show)
    print(f"\nAudit chart saved -> {chart_path}")
    print(_DIVIDER)


def _save_chart(
    base_ledger,
    stress_6bps,
    stress_t2,
    spy_returns: pd.Series,
    v82_returns: pd.Series | None,
    chart_path: Path,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 1.1]}
    )

    axes[0].plot(nav_curve(base_ledger.returns), label="G10 Forex Engine (3bps, T+1)", lw=2.0, color="#1f77b4")
    axes[0].plot(nav_curve(stress_6bps.returns), label="Stress: 6 bps (2x cost)", lw=1.0, alpha=0.85, color="#2ca02c")
    axes[0].plot(nav_curve(stress_t2.returns), label="Stress: T+2 lag", lw=1.0, alpha=0.85, color="#ff7f0e")
    if v82_returns is not None:
        axes[0].plot(nav_curve(v82_returns), label="AlphaShield V8.2 (ETF)", lw=1.3, ls="-.", color="#9467bd")
    axes[0].plot(nav_curve(spy_returns), label="SPY B&H", lw=1.1, ls="--", color="grey")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Growth of $1 (log)")
    axes[0].set_title("Regime-Switched G10 Forex Engine (Engine 3) — Strict T+1 Audit (2017–2026)")
    axes[0].legend(loc="upper left", ncol=2, fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].plot(100.0 * drawdown_curve(base_ledger.returns), label="G10 Forex Drawdown", lw=1.3, color="#1f77b4")
    if v82_returns is not None:
        axes[1].plot(100.0 * drawdown_curve(v82_returns), label="AlphaShield V8.2 Drawdown", lw=1.0, alpha=0.8, color="#9467bd")
    axes[1].plot(100.0 * drawdown_curve(spy_returns), label="SPY Drawdown", lw=0.9, alpha=0.6, color="grey")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend(loc="lower left", ncol=3, fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(base_ledger.gross_exposure, label="Gross Exposure (sum |w|)", lw=1.2, color="#d62728")
    axes[2].plot(base_ledger.net_exposure, label="Net Exposure (sum w)", lw=1.0, alpha=0.85, color="#1f77b4")
    axes[2].axhline(0.0, color="black", lw=0.7, ls=":")
    axes[2].set_ylabel("Exposure")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].legend(loc="upper left", ncol=2, fontsize=9)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if not args.self_test and not args.run:
        args.self_test = True

    if args.self_test:
        if not run_all_tests():
            sys.exit(1)

    if args.run:
        cfg = ForexConfig(
            backtest_start=args.start or ForexConfig.backtest_start,
            refresh=args.refresh,
        )
        run_backtest(cfg, show=args.show)


if __name__ == "__main__":
    main()
