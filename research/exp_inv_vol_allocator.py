# -*- coding: utf-8 -*-
"""Option 2 Inverse-Volatility & RAMom Core ETF Sleeve Allocator Backtest.

Compares 4 Core ETF allocation variants (zero look-ahead bias, T+1 execution):
  1. Baseline (Equal 20%): states / 3.0 * 0.20
  2. A: Inv-Vol (Monthly): monthly 1/vol_60 caps clipped to [0.08, 0.35], normalized to 1.0
  3. B: Inv-Vol + 2% Deadband: Variant A + 2% stateful deadband (full exits to 0% immediate)
  4. C: Top-3 RAMom + Inv-Vol: monthly Top-3 by 126D risk-adj momentum, 1/vol_60 (max 0.40), + 2% deadband

Evaluates all 4 across:
  - Table 1: Standalone 100% Core ETF (5 bps & 10 bps 2x stress)
  - Table 2: Flagship Blend 70/30 (70% Variant Core ETF + 30% Single-Stock, 5 bps & 10 bps)

Usage:
  uv run --with-requirements requirements.txt python research/exp_inv_vol_allocator.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.alphashield_v82.config import (
    CANARY_TICKER,
    CORE_UNIVERSE,
    EQUAL_CAP,
    PortfolioConfig as V82PortfolioConfig,
    SignalConfig as V82SignalConfig,
)
from research.alphashield_v82.data_engine import MarketData as V82MarketData
from research.alphashield_v82.portfolio_engine import (
    Ledger,
    PortfolioEngine as V82PortfolioEngine,
)
from research.alphashield_v82.signal_engine import SignalEngine as V82SignalEngine
from research.blend_sleeve import (
    BLEND_COST_BPS,
    EVAL_START,
    RISK_FREE_ANNUAL,
    STRESS_COST_MULTIPLIER,
    WARMUP_BACKTEST_START,
    _load_single_stock_and_spy_returns,
    compute_audit_row,
    simulate_fixed_monthly_blend,
)
from research.single_stock_engine.audit_suite import rf_series
from research.single_stock_engine.config import (
    EngineConfig as SingleStockConfig,
    TRADING_DAYS_PER_YEAR,
)


class FlexibleCapPortfolioEngine(V82PortfolioEngine):
    """Exact V8.2 PortfolioEngine allowing dynamic per-asset caps (total <= 100%)."""

    def _verify_invariants(
        self,
        ledger: Ledger,
        cost: float,
        lag: int,
        target_weights: pd.DataFrame,
    ) -> None:
        w = ledger.weights.to_numpy(dtype=float)
        if (w < -1e-12).any() or (w.sum(axis=1) > 1.0 + 1e-9).any():
            raise AssertionError("executed weights violate [0, 100%] portfolio bound")
        if not np.isfinite(ledger.returns).all():
            raise AssertionError("non-finite ledger returns")
        if (ledger.returns <= -1.0 + 1e-12).any():
            raise AssertionError("ledger return below -100%")


def _is_month_start(index: pd.DatetimeIndex) -> np.ndarray:
    periods = index.to_period("M")
    out = np.zeros(len(index), dtype=bool)
    out[0] = True
    out[1:] = periods[1:] != periods[:-1]
    return out


def _normalize_clipped_inv_vol(inv_vol: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip weights to [low, high] and iteratively normalize to sum to 1.0."""
    w = inv_vol / inv_vol.sum()
    for _ in range(10):
        w = np.clip(w, low, high)
        s = w.sum()
        if abs(s - 1.0) < 1e-12:
            break
        w = w / s
    return w


def _apply_deadband(target_weights: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    """Stateful deadband: keep prev_w if |target - prev| < threshold and both > 0."""
    arr = target_weights.to_numpy(dtype=float, copy=True)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for t in range(1, len(arr)):
        for j in range(arr.shape[1]):
            tw = arr[t, j]
            pw = out[t - 1, j]
            if tw > 0.0 and pw > 0.0 and abs(tw - pw) < threshold:
                out[t, j] = pw
            else:
                out[t, j] = tw
    return pd.DataFrame(out, index=target_weights.index, columns=target_weights.columns)


def build_variant_weights(
    closes: pd.DataFrame,
    tranche_states: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build 4 strictly backward-looking Core ETF weight DataFrames."""
    assets = list(tranche_states.columns)
    c = closes[assets].astype(float)
    rets = c.pct_change()

    # Strictly backward-looking 60D annualized vol (floored at 0.05) and 126D RAMom
    vol_60 = (rets.rolling(60).std() * math.sqrt(TRADING_DAYS_PER_YEAR)).clip(lower=0.05)
    ret_126 = c / c.shift(126) - 1.0
    ramom_126 = ret_126 / vol_60

    states_arr = tranche_states.to_numpy(dtype=float)
    month_starts = _is_month_start(closes.index)

    # 1. Baseline (Equal 20%)
    w_baseline = pd.DataFrame(
        states_arr / 3.0 * EQUAL_CAP,
        index=closes.index,
        columns=assets,
    )

    # 2. A: Inv-Vol (Monthly), clipped [0.08, 0.35], normalized to 1.0
    caps_a = np.full((len(closes), len(assets)), np.nan, dtype=float)
    vol_arr = vol_60.to_numpy(dtype=float)
    for t in range(len(closes)):
        if month_starts[t]:
            v = vol_arr[t]
            if np.all(np.isfinite(v)):
                caps_a[t] = _normalize_clipped_inv_vol(1.0 / v, 0.08, 0.35)
            else:
                caps_a[t] = EQUAL_CAP
    caps_a_df = pd.DataFrame(caps_a, index=closes.index, columns=assets).ffill().fillna(EQUAL_CAP)
    w_var_a = pd.DataFrame(
        (states_arr / 3.0) * caps_a_df.to_numpy(dtype=float),
        index=closes.index,
        columns=assets,
    )

    # 3. B: Inv-Vol + 2% Deadband
    w_var_b = _apply_deadband(w_var_a, threshold=0.02)

    # 4. C: Top-3 RAMom + Inv-Vol (max 0.40 each, bottom 2 = 0.0) + 2% Deadband
    caps_c = np.full((len(closes), len(assets)), np.nan, dtype=float)
    ramom_arr = ramom_126.to_numpy(dtype=float)
    for t in range(len(closes)):
        if month_starts[t]:
            rm = ramom_arr[t]
            v = vol_arr[t]
            if np.all(np.isfinite(rm)) and np.all(np.isfinite(v)):
                top3_idx = np.argsort(rm)[-3:]
                inv_v_top3 = 1.0 / v[top3_idx]
                w_top3 = _normalize_clipped_inv_vol(inv_v_top3, 0.0, 0.40)
                row_cap = np.zeros(len(assets), dtype=float)
                row_cap[top3_idx] = w_top3
                caps_c[t] = row_cap
            else:
                caps_c[t] = EQUAL_CAP
    caps_c_df = pd.DataFrame(caps_c, index=closes.index, columns=assets).ffill().fillna(EQUAL_CAP)
    w_var_c_raw = pd.DataFrame(
        (states_arr / 3.0) * caps_c_df.to_numpy(dtype=float),
        index=closes.index,
        columns=assets,
    )
    w_var_c = _apply_deadband(w_var_c_raw, threshold=0.02)

    return {
        "Baseline (Equal 20%)": w_baseline,
        "A: Inv-Vol (Monthly)": w_var_a,
        "B: Inv-Vol + 2% Deadband": w_var_b,
        "C: Top-3 RAMom + Inv-Vol": w_var_c,
    }


def _format_row(name: str, ledger: Ledger, spy_r: pd.Series, rf: pd.Series) -> dict[str, float | str]:
    audit = compute_audit_row(ledger.returns, spy_r, rf)
    ann_turnover = float(ledger.turnover.mean() * TRADING_DAYS_PER_YEAR * 100.0)
    return {
        "Variant": name,
        "CAGR (%)": round(audit["CAGR (%)"], 2),
        "Vol (%)": round(audit["Ann. Volatility (%)"], 2),
        "Sharpe (RF=2%)": round(audit["Sharpe (RF=2%)"], 3),
        "Max DD (%)": round(audit["Max Drawdown (%)"], 2),
        "Calmar": round(audit["Calmar"], 3),
        "Ann. Turnover (%)": round(ann_turnover, 1),
        "Spanning Alpha (%)": round(audit["Spanning Alpha vs SPY (%)"], 2),
        "p-value": round(audit["Alpha p-value"], 4),
    }


def _format_blend_row(
    name: str,
    blend_r: pd.Series,
    etf_ledger: Ledger,
    spy_r: pd.Series,
    rf: pd.Series,
) -> dict[str, float | str]:
    audit = compute_audit_row(blend_r, spy_r, rf)
    ann_turnover = float(etf_ledger.turnover.mean() * TRADING_DAYS_PER_YEAR * 100.0)
    return {
        "Variant": name,
        "CAGR (%)": round(audit["CAGR (%)"], 2),
        "Vol (%)": round(audit["Ann. Volatility (%)"], 2),
        "Sharpe (RF=2%)": round(audit["Sharpe (RF=2%)"], 3),
        "Max DD (%)": round(audit["Max Drawdown (%)"], 2),
        "Calmar": round(audit["Calmar"], 3),
        "Core Turnover (%)": round(ann_turnover, 1),
        "Spanning Alpha (%)": round(audit["Spanning Alpha vs SPY (%)"], 2),
        "p-value": round(audit["Alpha p-value"], 4),
    }


def main() -> None:
    import time
    import yfinance as yf
    from research.single_stock_engine.data_loader import DataLoader

    ss_cfg = SingleStockConfig()
    sig_cfg = V82SignalConfig()
    port_cfg = V82PortfolioConfig()

    v8_cache_file = Path(ss_cfg.cache_dir) / "v8_etf_ohlc.parquet"
    v8_tickers = list(CORE_UNIVERSE) + [CANARY_TICKER]
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

    data = V82MarketData(closes=closes_all, opens=opens_all, risk_assets=CORE_UNIVERSE)
    sig_frame = V82SignalEngine(data, sig_cfg).compute()
    engine = FlexibleCapPortfolioEngine(data, port_cfg)

    variants = build_variant_weights(data.closes, sig_frame.tranche_states)

    r_ss_full, r_spy_full = _load_single_stock_and_spy_returns(ss_cfg, WARMUP_BACKTEST_START)

    # Align evaluation calendar
    base_ledger_5 = engine.run(variants["Baseline (Equal 20%)"], start=EVAL_START, cost_bps=5.0)
    eval_idx = base_ledger_5.returns.index.intersection(r_ss_full.index).intersection(r_spy_full.index)
    r_ss = r_ss_full.loc[eval_idx]
    r_spy = r_spy_full.loc[eval_idx]
    rf = rf_series(eval_idx, annual_yield=RISK_FREE_ANNUAL)

    t1_rows = []
    t2_rows = []

    for cost_bps in (5.0, 10.0):
        blend_cost = BLEND_COST_BPS if cost_bps == 5.0 else BLEND_COST_BPS * STRESS_COST_MULTIPLIER
        tag = f"{int(cost_bps)} bps"
        for name, w_df in variants.items():
            ledger = engine.run(w_df, start=EVAL_START, cost_bps=cost_bps)
            t1_rows.append(_format_row(f"{name} [{tag}]", ledger, r_spy, rf))

            r_etf = ledger.returns.loc[eval_idx]
            blend_r, _, _ = simulate_fixed_monthly_blend(
                r_etf,
                r_ss,
                w1_target=0.70,
                w2_target=0.30,
                name=f"{name} [{tag}]",
                cost_bps=blend_cost,
            )
            t2_rows.append(_format_blend_row(f"{name} [{tag}]", blend_r, ledger, r_spy, rf))

    print("\n### Table 1: Standalone 100% Core ETF (5 bps & 10 bps 2x Stress)\n")
    print(pd.DataFrame(t1_rows).to_markdown(index=False))

    print("\n### Table 2: Flagship Blend 70/30 (70% Core ETF + 30% Single-Stock, 5 bps & 10 bps)\n")
    print(pd.DataFrame(t2_rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
