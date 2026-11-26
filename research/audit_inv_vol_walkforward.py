# -*- coding: utf-8 -*-
"""Walk-Forward, Crisis Stress & Parameter Sensitivity Audit for Formula A (Pure Inv-Vol).

Prints 3 Markdown tables:
  1. Table 1: Walk-Forward Split (2017-2021 vs 2022-2026) for 100% Core ETF & Blend 70/30
  2. Table 2: Crisis & Year-by-Year Stress (2018, 2020, 2022, 2025, 2026) for Blend 70/30
  3. Table 3: Parameter Sensitivity (40d/60d/90d lookback & [0.10, 0.30] bounds) for Blend 70/30

Usage:
  python research/audit_inv_vol_walkforward.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

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
from research.alphashield_v82.signal_engine import SignalEngine as V82SignalEngine
from research.blend_sleeve import (
    BLEND_COST_BPS,
    EVAL_START,
    RISK_FREE_ANNUAL,
    WARMUP_BACKTEST_START,
    _load_single_stock_and_spy_returns,
    compute_audit_row,
    simulate_fixed_monthly_blend,
)
from research.exp_inv_vol_allocator import (
    FlexibleCapPortfolioEngine,
    _is_month_start,
    _normalize_clipped_inv_vol,
)
from research.single_stock_engine.audit_suite import cagr, max_drawdown, rf_series
from research.single_stock_engine.config import (
    EngineConfig as SingleStockConfig,
    TRADING_DAYS_PER_YEAR,
)
from research.single_stock_engine.data_loader import DataLoader


def _download_ohlc_with_fallback(sym: str, start: str, end: str) -> pd.DataFrame:
    for attempt in range(2):
        try:
            raw = yf.download(
                sym,
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,
                timeout=30,
            )
            if raw is not None and not raw.empty:
                return raw
        except Exception:
            pass
        time.sleep(0.8 * (attempt + 1))

    from curl_cffi import requests as cffi_requests

    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(pd.Timestamp(end, tz="UTC").timestamp()) + 86400
    session = cffi_requests.Session(impersonate="chrome")
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = (
            f"https://{host}/v8/finance/chart/{sym}"
            f"?period1={p1}&period2={p2}&interval=1d&includeAdjustedClose=true"
        )
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code != 200:
                continue
            res = resp.json().get("chart", {}).get("result", [])
            if not res:
                continue
            ts = res[0].get("timestamp", [])
            quote = res[0].get("indicators", {}).get("quote", [{}])[0]
            adj = res[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
            if not ts or not quote:
                continue
            idx = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).normalize()
            df = pd.DataFrame(
                {
                    "Open": quote.get("open"),
                    "High": quote.get("high"),
                    "Low": quote.get("low"),
                    "Close": quote.get("close"),
                    "Adj Close": adj if adj is not None else quote.get("close"),
                    "Volume": quote.get("volume"),
                },
                index=idx,
            ).dropna(subset=["Close"])
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def _build_inv_vol_weights(
    closes: pd.DataFrame,
    tranche_states: pd.DataFrame,
    lookback: int = 60,
    low: float = 0.08,
    high: float = 0.35,
) -> pd.DataFrame:
    assets = list(tranche_states.columns)
    c = closes[assets].astype(float)
    rets = c.pct_change()
    vol = (rets.rolling(lookback).std() * math.sqrt(TRADING_DAYS_PER_YEAR)).clip(lower=0.05)

    states_arr = tranche_states.to_numpy(dtype=float)
    month_starts = _is_month_start(closes.index)
    caps = np.full((len(closes), len(assets)), np.nan, dtype=float)
    vol_arr = vol.to_numpy(dtype=float)

    for t in range(len(closes)):
        if month_starts[t]:
            v = vol_arr[t]
            if np.all(np.isfinite(v)):
                caps[t] = _normalize_clipped_inv_vol(1.0 / v, low, high)
            else:
                caps[t] = EQUAL_CAP

    caps_df = pd.DataFrame(caps, index=closes.index, columns=assets).ffill().fillna(EQUAL_CAP)
    return pd.DataFrame(
        (states_arr / 3.0) * caps_df.to_numpy(dtype=float),
        index=closes.index,
        columns=assets,
    )


def _subperiod_metrics(r: pd.Series, rf: pd.Series) -> dict[str, float]:
    f = rf.reindex(r.index).fillna(0.0).astype(float)
    excess = r.astype(float) - f
    ann_cagr = cagr(r)
    ann_vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
    ex_std = float(excess.std(ddof=1))
    sharpe = (
        float(excess.mean() * math.sqrt(TRADING_DAYS_PER_YEAR) / ex_std)
        if ex_std > 1e-12
        else float("nan")
    )
    mdd = max_drawdown(r)
    calmar = ann_cagr / abs(mdd) if abs(mdd) > 1e-12 else float("nan")
    return {
        "CAGR (%)": round(100.0 * ann_cagr, 2),
        "Vol (%)": round(100.0 * ann_vol, 2),
        "Sharpe (RF=2%)": round(sharpe, 3),
        "Max DD (%)": round(100.0 * mdd, 2),
        "Calmar": round(calmar, 3),
    }


def main() -> None:
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
            raw = _download_ohlc_with_fallback(sym, ss_cfg.data_start, end_dt)
            frames[sym] = DataLoader.validate_raw_ohlc(sym, raw)
            time.sleep(0.8)
        snap = pd.concat(frames, names=("Ticker", "Field")).sort_index()
        v8_cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            snap.to_parquet(v8_cache_file)
        except ImportError:
            pass

    closes_all = pd.DataFrame(
        {t: snap.xs(t, level="Ticker")["Close"] for t in v8_tickers}
    ).dropna(axis=0, how="any")
    opens_all = pd.DataFrame(
        {t: snap.xs(t, level="Ticker")["Open"] for t in CORE_UNIVERSE}
    ).reindex(closes_all.index).dropna(axis=0, how="any")
    closes_all = closes_all.reindex(opens_all.index)

    data = V82MarketData(closes=closes_all, opens=opens_all, risk_assets=CORE_UNIVERSE)
    sig_frame = V82SignalEngine(data, sig_cfg).compute()
    engine = FlexibleCapPortfolioEngine(opens_all, closes_all[list(CORE_UNIVERSE)], port_cfg)

    w_baseline = pd.DataFrame(
        sig_frame.tranche_states.to_numpy(dtype=float) / 3.0 * EQUAL_CAP,
        index=data.closes.index,
        columns=list(CORE_UNIVERSE),
    )

    r_ss_full, r_spy_full = _load_single_stock_and_spy_returns(ss_cfg, WARMUP_BACKTEST_START)

    base_etf = engine.run(w_baseline, start=EVAL_START, cost_bps=5.0).returns
    eval_idx = base_etf.index.intersection(r_ss_full.index).intersection(r_spy_full.index)
    r_ss = r_ss_full.loc[eval_idx]
    r_spy = r_spy_full.loc[eval_idx]
    rf = rf_series(eval_idx, annual_yield=RISK_FREE_ANNUAL)

    base_etf = base_etf.loc[eval_idx]
    base_blend, _, _ = simulate_fixed_monthly_blend(
        base_etf, r_ss, w1_target=0.70, w2_target=0.30, name="Baseline", cost_bps=BLEND_COST_BPS
    )

    w_a60 = _build_inv_vol_weights(data.closes, sig_frame.tranche_states, lookback=60, low=0.08, high=0.35)
    a60_etf = engine.run(w_a60, start=EVAL_START, cost_bps=5.0).returns.loc[eval_idx]
    a60_blend, _, _ = simulate_fixed_monthly_blend(
        a60_etf, r_ss, w1_target=0.70, w2_target=0.30, name="Formula A", cost_bps=BLEND_COST_BPS
    )

    # ── Table 1: Walk-Forward Split (2017-2021 vs 2022-2026) ─────────────────
    t1_rows = []
    splits = [
        ("2017-2021", eval_idx[(eval_idx >= "2017-01-01") & (eval_idx <= "2021-12-31")]),
        ("2022-2026", eval_idx[eval_idx >= "2022-01-01"]),
    ]
    for period_name, sub_idx in splits:
        for sleeve_name, s_base, s_a in [
            ("100% Core ETF", base_etf, a60_etf),
            ("Blend 70/30", base_blend, a60_blend),
        ]:
            m_b = _subperiod_metrics(s_base.loc[sub_idx], rf)
            m_a = _subperiod_metrics(s_a.loc[sub_idx], rf)
            t1_rows.append({"Period": period_name, "Sleeve": sleeve_name, "Strategy": "Baseline (Equal 20%)", **m_b})
            t1_rows.append({"Period": period_name, "Sleeve": sleeve_name, "Strategy": "Formula A (Inv-Vol 60d)", **m_a})

    print("\n### Table 1: Walk-Forward Split (2017-2021 vs 2022-2026, 5 bps)\n")
    print(pd.DataFrame(t1_rows).to_markdown(index=False))

    # ── Table 2: Crisis & Year-by-Year Stress (Blend 70/30) ──────────────────
    t2_rows = []
    for yr in (2018, 2020, 2022, 2025, 2026):
        yr_idx = eval_idx[eval_idx.year == yr]
        if len(yr_idx) == 0:
            continue
        rb = base_blend.loc[yr_idx]
        ra = a60_blend.loc[yr_idx]
        ret_b = ((1.0 + rb).prod() - 1.0) * 100.0
        ret_a = ((1.0 + ra).prod() - 1.0) * 100.0
        mdd_b = max_drawdown(rb) * 100.0
        mdd_a = max_drawdown(ra) * 100.0
        vol_b = float(rb.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) * 100.0
        vol_a = float(ra.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) * 100.0
        t2_rows.append({
            "Year": yr,
            "Baseline Return (%)": round(ret_b, 2),
            "Formula A Return (%)": round(ret_a, 2),
            "Baseline Max DD (%)": round(mdd_b, 2),
            "Formula A Max DD (%)": round(mdd_a, 2),
            "Vol Diff (%)": round(vol_a - vol_b, 2),
        })

    print("\n### Table 2: Crisis & Year-by-Year Stress (Blend 70/30, 5 bps)\n")
    print(pd.DataFrame(t2_rows).to_markdown(index=False))

    # ── Table 3: Parameter Sensitivity (Full 2017-2026 Blend 70/30) ──────────
    t3_rows = []
    configs = [
        ("Baseline (Equal 20%)", None),
        ("Inv-Vol 40d [0.08, 0.35]", (40, 0.08, 0.35)),
        ("Inv-Vol 60d [0.08, 0.35]", (60, 0.08, 0.35)),
        ("Inv-Vol 90d [0.08, 0.35]", (90, 0.08, 0.35)),
        ("Inv-Vol 60d [0.10, 0.30]", (60, 0.10, 0.30)),
    ]
    for label, params in configs:
        if params is None:
            blend_r = base_blend
        else:
            lb, lo, hi = params
            w_df = _build_inv_vol_weights(data.closes, sig_frame.tranche_states, lookback=lb, low=lo, high=hi)
            etf_r = engine.run(w_df, start=EVAL_START, cost_bps=5.0).returns.loc[eval_idx]
            blend_r, _, _ = simulate_fixed_monthly_blend(
                etf_r, r_ss, w1_target=0.70, w2_target=0.30, name=label, cost_bps=BLEND_COST_BPS
            )
        audit = compute_audit_row(blend_r, r_spy, rf)
        t3_rows.append({
            "Configuration": label,
            "CAGR (%)": round(audit["CAGR (%)"], 2),
            "Vol (%)": round(audit["Ann. Volatility (%)"], 2),
            "Sharpe (RF=2%)": round(audit["Sharpe (RF=2%)"], 3),
            "Max DD (%)": round(audit["Max Drawdown (%)"], 2),
            "Calmar": round(audit["Calmar"], 3),
            "p-value": round(audit["Alpha p-value"], 4),
        })

    print("\n### Table 3: Parameter Sensitivity (Full 2017-2026 Blend 70/30, 5 bps)\n")
    print(pd.DataFrame(t3_rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
