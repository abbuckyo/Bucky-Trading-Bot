# -*- coding: utf-8 -*-
"""Portfolio Sleeve Blending Engine & Comparative Institutional Audit.

Evaluates and compares 5 allocation models across the common 2017-2026 window:
  1. 100% Core ETF (AlphaShield V8.2)
  2. 100% Single-Stock Engine
  3. Blend 70/30 (70% Core ETF + 30% Single-Stock, rebalanced monthly)
  4. Blend 50/50 (50% Core ETF + 50% Single-Stock, rebalanced monthly)
  5. Inverse-Volatility Risk Parity Blend (weights proportional to 1 / sigma_60D
     of each engine using strictly lagged t-1 returns, rebalanced monthly)

Saves equity & drawdown visualization to:
  research/output/sleeve_blending_audit.png

Usage:
  python -m research.blend_sleeve
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.alphashield_v82.config import (  # noqa: E402
    CANARY_TICKER,
    CORE_UNIVERSE,
    PortfolioConfig as V82PortfolioConfig,
    SignalConfig as V82SignalConfig,
)
from research.alphashield_v82.data_engine import MarketData as V82MarketData  # noqa: E402
from research.alphashield_v82.portfolio_engine import (  # noqa: E402
    PortfolioEngine as V82PortfolioEngine,
)
from research.alphashield_v82.signal_engine import (  # noqa: E402
    SignalEngine as V82SignalEngine,
)
from research.single_stock_engine.audit_suite import (  # noqa: E402
    cagr,
    drawdown_curve,
    max_drawdown,
    nav_curve,
    rf_series,
    spanning_regression,
)
from research.single_stock_engine.config import (  # noqa: E402
    EngineConfig as SingleStockConfig,
    TRADING_DAYS_PER_YEAR,
)
from research.single_stock_engine.data_loader import DataLoader  # noqa: E402
from research.single_stock_engine.portfolio_engine import (  # noqa: E402
    PortfolioEngine as SingleStockPortfolioEngine,
    build_signal_weights,
)
from research.single_stock_engine.ranking_engine import RankingEngine  # noqa: E402
from research.single_stock_engine.regime_guard import RegimeGuard  # noqa: E402
from research.single_stock_engine.risk_guards import RiskGuards  # noqa: E402

RISK_FREE_ANNUAL: float = 0.02
VOL_LOOKBACK_DAYS: int = 60
WARMUP_BACKTEST_START: str = "2016-09-01"  # 4-month pre-window warm-up for 60D sleeve vol
EVAL_START: str = "2017-01-01"
CHART_OUTPUT_PATH: Path = _REPO_ROOT / "research" / "output" / "sleeve_blending_audit.png"

BLEND_COST_BPS: float = 5.0            # per-side cost on monthly sleeve rebalance turnover
STRESS_COST_MULTIPLIER: float = 2.0    # 2x cost stress check

# Single-stock baseline from the pre-V2 audit run (before regime-adaptive budgeting).
BASELINE_SS_V1: dict[str, float] = {
    "CAGR (%)": 7.4889,
    "Sharpe (RF=2%)": 0.7288,
    "Max Drawdown (%)": -9.5256,
}


@dataclass(slots=True)
class EngineStreams:
    """Aligned daily net return streams for Engine 1, Engine 2, and SPY."""

    core_etf_full: pd.Series       # includes late-2016 warm-up for 60D vol
    single_stock_full: pd.Series   # includes late-2016 warm-up for 60D vol
    spy_full: pd.Series
    eval_index: pd.DatetimeIndex   # 2017-01-01 -> present evaluation calendar


def _load_v82_returns(ss_cfg: SingleStockConfig, start_date: str) -> pd.Series:
    """Run AlphaShield V8.2 Core ETF engine from cached Parquet snapshot."""
    v8_cache_file = Path(ss_cfg.cache_dir) / "v8_etf_ohlc.parquet"
    v8_tickers = list(CORE_UNIVERSE) + [CANARY_TICKER]
    snap: pd.DataFrame | None = None
    if v8_cache_file.exists():
        snap = pd.read_parquet(v8_cache_file)
    else:
        frames: dict[str, pd.DataFrame] = {}
        end_dt = str(pd.Timestamp.now(tz="America/New_York").date())
        for sym in v8_tickers:
            raw = yf.download(
                sym,
                start=ss_cfg.data_start,
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

    mkt = V82MarketData(closes=closes_all, opens=opens_all, risk_assets=CORE_UNIVERSE)
    sig = V82SignalEngine(mkt, V82SignalConfig()).compute()
    ledger = V82PortfolioEngine(
        mkt.opens[list(CORE_UNIVERSE)],
        mkt.closes[list(CORE_UNIVERSE)],
        V82PortfolioConfig(),
    ).run(sig.weights, start=start_date)
    return ledger.returns.rename("100% Core ETF (V8.2)")


def _load_single_stock_and_spy_returns(
    ss_cfg: SingleStockConfig,
    start_date: str,
) -> tuple[pd.Series, pd.Series]:
    """Run Single-Stock Momentum Engine from cached Parquet snapshot."""
    bundle = DataLoader(ss_cfg).load(fetch_earnings=True)
    regime = RegimeGuard(
        spy_ema_window=ss_cfg.spy_ema_window,
        earnings_blackout_days=ss_cfg.earnings_blackout_days,
    ).evaluate(
        spy_close=bundle.spy_close,
        tickers=bundle.tickers,
        earnings_dates=bundle.earnings_dates,
    )
    ranking = RankingEngine(
        lookback_long=ss_cfg.mom_lookback_long,
        lookback_skip=ss_cfg.mom_lookback_skip,
        vol_lookback=ss_cfg.vol_lookback,
        vol_floor=ss_cfg.vol_floor,
        top_quantile=ss_cfg.top_quantile,
        max_active_holdings=ss_cfg.max_active_holdings,
    ).compute(bundle.closes)
    guards = RiskGuards(
        ema_window=ss_cfg.stock_ema_window,
        atr_window=ss_cfg.atr_window,
        chandelier_high_window=ss_cfg.chandelier_high_window,
        chandelier_atr_mult=ss_cfg.chandelier_atr_mult,
        earnings_tight_atr_mult=ss_cfg.earnings_tight_atr_mult,
    ).evaluate(
        highs=bundle.highs,
        lows=bundle.lows,
        closes=bundle.closes,
        earnings_blackout=regime.earnings_blackout,
    )
    sig_weights = build_signal_weights(
        closes=bundle.closes,
        ranking=ranking,
        regime=regime,
        guards=guards,
        config=ss_cfg,
    )
    ledger = SingleStockPortfolioEngine(bundle.opens, bundle.closes, ss_cfg).run(
        sig_weights, start=start_date
    )
    spy_ret = (
        bundle.spy_close.pct_change(fill_method=None)
        .reindex(ledger.returns.index)
        .fillna(0.0)
        .rename("SPY B&H")
    )
    return ledger.returns.rename("100% Single-Stock"), spy_ret


def load_aligned_streams() -> EngineStreams:
    """Extract and align daily net return series from both engines (zero look-ahead)."""
    ss_cfg = SingleStockConfig()
    v82_ret = _load_v82_returns(ss_cfg, start_date=WARMUP_BACKTEST_START)
    ss_ret, spy_ret = _load_single_stock_and_spy_returns(ss_cfg, start_date=WARMUP_BACKTEST_START)

    common_idx = v82_ret.index.intersection(ss_ret.index).intersection(spy_ret.index).sort_values()
    v82_aligned = v82_ret.loc[common_idx].astype(float)
    ss_aligned = ss_ret.loc[common_idx].astype(float)
    spy_aligned = spy_ret.loc[common_idx].astype(float)

    eval_idx = common_idx[common_idx >= pd.Timestamp(EVAL_START)]
    if len(eval_idx) == 0:
        raise RuntimeError("No overlapping evaluation dates found >= 2017-01-01.")

    return EngineStreams(
        core_etf_full=v82_aligned,
        single_stock_full=ss_aligned,
        spy_full=spy_aligned,
        eval_index=eval_idx,
    )


def _is_month_start(index: pd.DatetimeIndex) -> np.ndarray:
    """Return boolean mask flagging the first trading day of each calendar month."""
    ym = index.year * 100 + index.month
    first = np.empty(len(index), dtype=bool)
    first[0] = True
    first[1:] = ym[1:] != ym[:-1]
    return first


def simulate_fixed_monthly_blend(
    r1: pd.Series,
    r2: pd.Series,
    w1_target: float,
    w2_target: float,
    name: str,
    cost_bps: float = BLEND_COST_BPS,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Simulate a monthly-rebalanced fixed-weight two-sleeve portfolio.

    Rebalances to ``(w1_target, w2_target)`` at the start of the first trading
    session of each calendar month (using prior month-end closing capital) and
    drifts intra-month with each sleeve's net daily return. Every monthly
    rebalance pays ``cost_bps`` on the traded sleeve notional (both legs).
    """
    if not r1.index.equals(r2.index):
        raise ValueError("Sleeve return series must share an identical DatetimeIndex.")
    if abs(w1_target + w2_target - 1.0) > 1e-9:
        raise ValueError("Sleeve target weights must sum to 1.0.")

    idx = r1.index
    r1_arr = r1.to_numpy(dtype=float)
    r2_arr = r2.to_numpy(dtype=float)
    rebal_flags = _is_month_start(idx)
    cost_rate = cost_bps * 1e-4

    n = len(idx)
    port_rets = np.zeros(n, dtype=float)
    weights_hist = np.zeros((n, 2), dtype=float)
    cost_hist = np.zeros(n, dtype=float)

    nav = 1.0
    cap1 = w1_target * nav
    cap2 = w2_target * nav

    for t in range(n):
        if rebal_flags[t]:
            target1 = w1_target * nav
            target2 = w2_target * nav
            cost = cost_rate * (abs(target1 - cap1) + abs(target2 - cap2))
            cap1, cap2 = target1, target2
            cost_hist[t] = cost
        else:
            cost = 0.0

        weights_hist[t, 0] = cap1 / nav
        weights_hist[t, 1] = cap2 / nav

        prev_nav = nav
        new_cap1 = cap1 * (1.0 + r1_arr[t])
        new_cap2 = cap2 * (1.0 + r2_arr[t])
        gross = new_cap1 + new_cap2
        nav = gross - cost
        port_rets[t] = nav / prev_nav - 1.0
        scale = nav / gross if gross > 0.0 else 0.0
        cap1, cap2 = new_cap1 * scale, new_cap2 * scale

    ret_series = pd.Series(port_rets, index=idx, name=name)
    w_df = pd.DataFrame(weights_hist, index=idx, columns=["Core_ETF_Weight", "Single_Stock_Weight"])
    cost_series = pd.Series(cost_hist, index=idx, name=f"{name} friction")
    return ret_series, w_df, cost_series


def simulate_inverse_vol_blend(
    r1_full: pd.Series,
    r2_full: pd.Series,
    eval_index: pd.DatetimeIndex,
    vol_window: int = VOL_LOOKBACK_DAYS,
    name: str = "Inv-Vol Risk Parity",
    cost_bps: float = BLEND_COST_BPS,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Simulate a monthly-rebalanced Inverse-Volatility Risk Parity blend.

    Strict Zero Look-Ahead:
      * 60-day realized annualized volatility is computed through session ``t-1``
        (``rolling(vol_window).std(ddof=1).shift(1) * sqrt(252)``).
      * On the first trading session of each month ``t``, target sleeve weights
        are set proportional to ``1 / sigma_{i, 60D}(t-1)``.
      * Intra-month, capital drifts naturally with each engine's daily return.
      * Every monthly rebalance pays ``cost_bps`` on the traded sleeve notional.
    """
    if not r1_full.index.equals(r2_full.index):
        raise ValueError("Full sleeve return series must share an identical DatetimeIndex.")

    # Strictly lagged 60D realized volatility (uses returns through t-1 only)
    vol1_lag = (
        r1_full.rolling(window=vol_window, min_periods=20).std(ddof=1).shift(1)
        * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    vol2_lag = (
        r2_full.rolling(window=vol_window, min_periods=20).std(ddof=1).shift(1)
        * math.sqrt(TRADING_DAYS_PER_YEAR)
    )

    r1 = r1_full.loc[eval_index].to_numpy(dtype=float)
    r2 = r2_full.loc[eval_index].to_numpy(dtype=float)
    v1 = vol1_lag.loc[eval_index].to_numpy(dtype=float)
    v2 = vol2_lag.loc[eval_index].to_numpy(dtype=float)
    rebal_flags = _is_month_start(eval_index)
    cost_rate = cost_bps * 1e-4

    n = len(eval_index)
    port_rets = np.zeros(n, dtype=float)
    weights_hist = np.zeros((n, 2), dtype=float)
    cost_hist = np.zeros(n, dtype=float)

    nav = 1.0
    w1_cur, w2_cur = 0.5, 0.5
    cap1, cap2 = 0.5 * nav, 0.5 * nav

    for t in range(n):
        cost = 0.0
        if rebal_flags[t]:
            s1 = max(float(v1[t]), 0.02) if np.isfinite(v1[t]) and v1[t] > 0.0 else 0.10
            s2 = max(float(v2[t]), 0.02) if np.isfinite(v2[t]) and v2[t] > 0.0 else 0.10
            inv1 = 1.0 / s1
            inv2 = 1.0 / s2
            w1_cur = inv1 / (inv1 + inv2)
            w2_cur = inv2 / (inv1 + inv2)
            target1 = w1_cur * nav
            target2 = w2_cur * nav
            cost = cost_rate * (abs(target1 - cap1) + abs(target2 - cap2))
            cap1, cap2 = target1, target2
            cost_hist[t] = cost

        weights_hist[t, 0] = cap1 / nav
        weights_hist[t, 1] = cap2 / nav

        prev_nav = nav
        new_cap1 = cap1 * (1.0 + r1[t])
        new_cap2 = cap2 * (1.0 + r2[t])
        gross = new_cap1 + new_cap2
        nav = gross - cost
        port_rets[t] = nav / prev_nav - 1.0
        scale = nav / gross if gross > 0.0 else 0.0
        cap1, cap2 = new_cap1 * scale, new_cap2 * scale

    ret_series = pd.Series(port_rets, index=eval_index, name=name)
    w_df = pd.DataFrame(weights_hist, index=eval_index, columns=["Core_ETF_Weight", "Single_Stock_Weight"])
    cost_series = pd.Series(cost_hist, index=eval_index, name=f"{name} friction")
    return ret_series, w_df, cost_series


def compute_audit_row(
    returns: pd.Series,
    spy_returns: pd.Series,
    rf: pd.Series,
) -> dict[str, float]:
    """Compute institutional metrics and Newey-West HAC spanning regression vs SPY."""
    r = returns.astype(float)
    f = rf.reindex(r.index).fillna(0.0).astype(float)
    excess = r - f

    ann_cagr = cagr(r)
    ann_vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
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
    mdd = max_drawdown(r)
    calmar = ann_cagr / abs(mdd) if abs(mdd) > 1e-12 else float("nan")

    span = spanning_regression(r, spy_returns.rename("SPY"), f)
    return {
        "CAGR (%)": 100.0 * ann_cagr,
        "Ann. Volatility (%)": 100.0 * ann_vol,
        "Sharpe (RF=2%)": sharpe,
        "Sortino": sortino,
        "Max Drawdown (%)": 100.0 * mdd,
        "Calmar": calmar,
        "Spanning Alpha vs SPY (%)": 100.0 * float(span["alpha_annual"]),
        "Alpha HAC t-stat": float(span["alpha_t_stat"]),
        "Alpha p-value": float(span["alpha_p_value"]),
        "Beta vs SPY": float(span["beta"]),
        "Corr vs SPY": float(span["correlation"]),
    }


def compute_yearly_breakdown(models: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute calendar-year compound returns (%) across all models and SPY."""
    years = sorted(next(iter(models.values())).index.year.unique())
    table: dict[int, dict[str, float]] = {y: {} for y in years}
    for name, s in models.items():
        yr_series = np.expm1(np.log1p(s.astype(float)).groupby(s.index.year).sum()) * 100.0
        for y in years:
            table[y][name] = float(yr_series.get(y, float("nan")))
    df = pd.DataFrame(table).T
    df.index.name = "Year"
    return df


def save_blending_chart(
    models: dict[str, pd.Series],
    spy_returns: pd.Series,
    inv_vol_weights: pd.DataFrame,
    out_path: Path,
) -> None:
    """Render and save 3-panel Equity, Drawdown, and Dynamic Risk Parity Weight chart."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1.0]}
    )

    palette = {
        "100% Core ETF (V8.2)": ("#1f77b4", 1.5, "-"),
        "100% Single-Stock": ("#ff7f0e", 1.5, "-"),
        "Blend 70/30": ("#2ca02c", 2.0, "-"),
        "Blend 50/50": ("#9467bd", 1.8, "-"),
        "Inv-Vol Risk Parity": ("#d62728", 2.1, "-"),
    }

    for label, s in models.items():
        color, lw, ls = palette.get(label, ("#333333", 1.4, "-"))
        axes[0].plot(nav_curve(s), label=label, color=color, lw=lw, ls=ls)
    axes[0].plot(nav_curve(spy_returns), label="SPY B&H", color="grey", lw=1.2, ls="--", alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Growth of $1 (log)")
    axes[0].set_title(
        "Portfolio Sleeve Blending Audit (2017–2026) — AlphaShield V8.2 ETF + Single-Stock Momentum"
    )
    axes[0].legend(loc="upper left", ncol=2, fontsize=9)
    axes[0].grid(alpha=0.25)

    for label, s in models.items():
        color, lw, ls = palette.get(label, ("#333333", 1.2, "-"))
        axes[1].plot(100.0 * drawdown_curve(s), label=label, color=color, lw=lw * 0.85, ls=ls)
    axes[1].plot(
        100.0 * drawdown_curve(spy_returns),
        label="SPY Drawdown",
        color="grey",
        lw=1.0,
        ls="--",
        alpha=0.7,
    )
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend(loc="lower left", ncol=3, fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].stackplot(
        inv_vol_weights.index,
        100.0 * inv_vol_weights["Core_ETF_Weight"],
        100.0 * inv_vol_weights["Single_Stock_Weight"],
        labels=["Core ETF Sleeve (V8.2)", "Single-Stock Sleeve"],
        colors=["#1f77b4", "#ff7f0e"],
        alpha=0.75,
    )
    axes[2].set_ylabel("Inv-Vol Weight (%)")
    axes[2].set_ylim(0.0, 100.0)
    axes[2].legend(loc="upper left", fontsize=9)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    divider = "=" * 96
    print(divider)
    print("PORTFOLIO SLEEVE BLENDING ENGINE & COMPARATIVE INSTITUTIONAL AUDIT (2017–2026)")
    print(divider)

    streams = load_aligned_streams()
    idx = streams.eval_index
    r_etf = streams.core_etf_full.loc[idx]
    r_ss = streams.single_stock_full.loc[idx]
    r_spy = streams.spy_full.loc[idx]
    rf = rf_series(idx, annual_yield=RISK_FREE_ANNUAL)

    blend_70_30, _, cost_70_30 = simulate_fixed_monthly_blend(
        r_etf, r_ss, w1_target=0.70, w2_target=0.30, name="Blend 70/30"
    )
    blend_50_50, _, cost_50_50 = simulate_fixed_monthly_blend(
        r_etf, r_ss, w1_target=0.50, w2_target=0.50, name="Blend 50/50"
    )
    blend_inv_vol, inv_vol_weights, cost_inv_vol = simulate_inverse_vol_blend(
        streams.core_etf_full,
        streams.single_stock_full,
        eval_index=idx,
        vol_window=VOL_LOOKBACK_DAYS,
        name="Inv-Vol Risk Parity",
    )

    models: dict[str, pd.Series] = {
        "100% Core ETF (V8.2)": r_etf,
        "100% Single-Stock": r_ss,
        "Blend 70/30": blend_70_30,
        "Blend 50/50": blend_50_50,
        "Inv-Vol Risk Parity": blend_inv_vol,
    }

    # 2x cost stress: identical blends with the sleeve rebalance friction doubled.
    stress_bps = STRESS_COST_MULTIPLIER * BLEND_COST_BPS
    stress_models: dict[str, pd.Series] = {
        "Blend 70/30": simulate_fixed_monthly_blend(
            r_etf, r_ss, w1_target=0.70, w2_target=0.30,
            name="Blend 70/30", cost_bps=stress_bps,
        )[0],
        "Blend 50/50": simulate_fixed_monthly_blend(
            r_etf, r_ss, w1_target=0.50, w2_target=0.50,
            name="Blend 50/50", cost_bps=stress_bps,
        )[0],
        "Inv-Vol Risk Parity": simulate_inverse_vol_blend(
            streams.core_etf_full,
            streams.single_stock_full,
            eval_index=idx,
            vol_window=VOL_LOOKBACK_DAYS,
            name="Inv-Vol Risk Parity",
            cost_bps=stress_bps,
        )[0],
    }

    print(
        f"Aligned Window : {idx[0].date()} -> {idx[-1].date()} "
        f"({len(idx)} trading sessions, Risk-Free = {RISK_FREE_ANNUAL:.1%})"
    )
    print(
        f"Inv-Vol Weights: Avg Core ETF = {inv_vol_weights['Core_ETF_Weight'].mean():.2%}, "
        f"Avg Single-Stock = {inv_vol_weights['Single_Stock_Weight'].mean():.2%}"
    )
    print(
        f"Costs          : {BLEND_COST_BPS:.1f} bps per side on monthly sleeve rebalance "
        f"(sleeve-level costs already embedded); 2x stress = {stress_bps:.1f} bps"
    )

    # 1. Summary Table
    summary_rows: dict[str, dict[str, float]] = {}
    for name, s in models.items():
        summary_rows[name] = compute_audit_row(s, r_spy, rf)
    summary_rows["SPY B&H"] = compute_audit_row(r_spy, r_spy, rf)
    summary_df = pd.DataFrame(summary_rows)

    print("\n--- 1. Comparative Performance & Spanning Alpha, net of costs (2017-2026) ---")
    with pd.option_context("display.width", 220, "display.max_columns", 10):
        print(summary_df.to_string(float_format=lambda v: f"{float(v): .4f}"))

    # 2. Correlation Matrix
    corr_df = pd.DataFrame(
        {
            "100% Core ETF (V8.2)": r_etf,
            "100% Single-Stock": r_ss,
            "Blend 70/30": blend_70_30,
            "Blend 50/50": blend_50_50,
            "Inv-Vol Risk Parity": blend_inv_vol,
            "SPY B&H": r_spy,
        }
    ).corr()
    print("\n--- 2. Daily Return Correlation Matrix ---")
    with pd.option_context("display.width", 200, "display.max_columns", 10):
        print(corr_df.to_string(float_format=lambda v: f"{float(v): .4f}"))

    # 3. Yearly Performance Breakdown
    models_with_spy = dict(models)
    models_with_spy["SPY B&H"] = r_spy
    yearly_df = compute_yearly_breakdown(models_with_spy)
    print("\n--- 3. Calendar-Year Compound Returns (%) [2018 & 2022 Bear Markets Highlighted] ---")
    with pd.option_context("display.width", 200, "display.max_columns", 10):
        print(yearly_df.to_string(float_format=lambda v: f"{float(v):+.2f}%"))

    # 4. Monthly rebalance cost drag
    years = len(idx) / 252.0
    cost_df = pd.DataFrame(
        {
            "Total friction (% of initial)": [
                100.0 * float(cost_70_30.sum()),
                100.0 * float(cost_50_50.sum()),
                100.0 * float(cost_inv_vol.sum()),
            ],
            "Annualised drag (bps)": [
                1e4 * float(cost_70_30.sum()) / years,
                1e4 * float(cost_50_50.sum()) / years,
                1e4 * float(cost_inv_vol.sum()) / years,
            ],
        },
        index=["Blend 70/30", "Blend 50/50", "Inv-Vol Risk Parity"],
    )
    print(f"\n--- 4. Monthly Sleeve Rebalance Cost Drag ({BLEND_COST_BPS:.1f} bps/side) ---")
    with pd.option_context("display.width", 160):
        print(cost_df.to_string(float_format=lambda v: f"{float(v): .2f}"))

    # 5. 2x cost stress check
    stress_rows: dict[str, dict[str, float]] = {}
    for name in stress_models:
        base_m = compute_audit_row(models[name], r_spy, rf)
        stress_m = compute_audit_row(stress_models[name], r_spy, rf)
        stress_rows[name] = {
            "CAGR base (%)": base_m["CAGR (%)"],
            "CAGR 2x (%)": stress_m["CAGR (%)"],
            "dCAGR (pp)": stress_m["CAGR (%)"] - base_m["CAGR (%)"],
            "Sharpe base": base_m["Sharpe (RF=2%)"],
            "Sharpe 2x": stress_m["Sharpe (RF=2%)"],
            "dSharpe": stress_m["Sharpe (RF=2%)"] - base_m["Sharpe (RF=2%)"],
            "MDD base (%)": base_m["Max Drawdown (%)"],
            "MDD 2x (%)": stress_m["Max Drawdown (%)"],
        }
    print(f"\n--- 5. 2x Cost Stress ({stress_bps:.1f} bps/side) ---")
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(pd.DataFrame(stress_rows).T.to_string(float_format=lambda v: f"{float(v): .4f}"))

    # 6. Single-stock V1 -> V2 attribution
    ss_m = compute_audit_row(r_ss, r_spy, rf)
    compare = pd.DataFrame(
        {
            "V1 (pre-V2)": [
                BASELINE_SS_V1["CAGR (%)"],
                BASELINE_SS_V1["Sharpe (RF=2%)"],
                BASELINE_SS_V1["Max Drawdown (%)"],
            ],
            "V2 (regime-adaptive)": [
                ss_m["CAGR (%)"],
                ss_m["Sharpe (RF=2%)"],
                ss_m["Max Drawdown (%)"],
            ],
        },
        index=["CAGR (%)", "Sharpe (RF=2%)", "Max Drawdown (%)"],
    )
    compare["Delta"] = compare["V2 (regime-adaptive)"] - compare["V1 (pre-V2)"]
    print("\n--- 6. Single-Stock Engine V1 -> V2 (cash-drag unlock) ---")
    with pd.option_context("display.width", 160):
        print(compare.to_string(float_format=lambda v: f"{float(v):+.4f}"))

    save_blending_chart(models, r_spy, inv_vol_weights, CHART_OUTPUT_PATH)
    print(f"\nVisualization saved -> {CHART_OUTPUT_PATH}")
    print(divider)


if __name__ == "__main__":
    main()
