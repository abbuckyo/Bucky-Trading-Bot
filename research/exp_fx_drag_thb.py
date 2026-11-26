# -*- coding: utf-8 -*-
"""10-Year USD/THB FX Drag & Hedging Cost Audit for Formula A (Pure Inv-Vol 60d).

Compares 4 THB-denominated scenarios (2017-2026, zero look-ahead bias, T+1 lag):
  1. USD Native (Reference): raw Formula A USD return
  2. THB Fully Hedged (SCB -E style): USD return minus daily prorated hedge carry + 0.25% bank spread
  3. THB Unhedged: (1 + r_usd) * (1 + r_usdthb) - 1
  4. THB Dynamic FX Hedge (USD/THB < EMA100 -> Hedge, else Unhedged): monthly check with 10 bps switch cost

Usage:
  python research/exp_fx_drag_thb.py
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
    PortfolioConfig as V82PortfolioConfig,
    SignalConfig as V82SignalConfig,
)
from research.alphashield_v82.data_engine import MarketData as V82MarketData
from research.alphashield_v82.signal_engine import SignalEngine as V82SignalEngine
from research.blend_sleeve import EVAL_START, RISK_FREE_ANNUAL
from research.exp_inv_vol_allocator import (
    FlexibleCapPortfolioEngine,
    _is_month_start,
    build_variant_weights,
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


def _extract_close(raw: pd.DataFrame) -> pd.Series:
    if isinstance(raw.columns, pd.MultiIndex):
        s = raw["Close"].iloc[:, 0]
    else:
        s = raw["Close"]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _bot_policy_rate(index: pd.DatetimeIndex) -> pd.Series:
    """Piecewise historical Bank of Thailand policy rate proxy."""
    rates = np.where(
        index.year <= 2019,
        0.0150,
        np.where(
            index.year <= 2022,
            0.0050,
            np.where(index.year <= 2024, 0.0250, 0.0200),
        ),
    )
    return pd.Series(rates, index=index, dtype=float)


def _compute_metrics(r: pd.Series, rf: pd.Series, avg_hedge_cost_pct: float) -> dict[str, float]:
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
        "Avg Hedge Cost (%/yr)": round(avg_hedge_cost_pct, 2),
    }


def main() -> None:
    ss_cfg = SingleStockConfig()
    sig_cfg = V82SignalConfig()
    port_cfg = V82PortfolioConfig()

    v8_tickers = list(CORE_UNIVERSE) + [CANARY_TICKER]
    frames: dict[str, pd.DataFrame] = {}
    end_dt = str(pd.Timestamp.now(tz="America/New_York").date())
    for sym in v8_tickers:
        raw = _download_ohlc_with_fallback(sym, ss_cfg.data_start, end_dt)
        frames[sym] = DataLoader.validate_raw_ohlc(sym, raw)
        time.sleep(0.8)
    snap = pd.concat(frames, names=("Ticker", "Field")).sort_index()

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

    variants = build_variant_weights(data.closes, sig_frame.tranche_states)
    r_usd_full = engine.run(variants["A: Inv-Vol (Monthly)"], start=EVAL_START, cost_bps=5.0).returns

    # Fetch THB=X and ^IRX
    thb_raw = _download_ohlc_with_fallback("THB=X", ss_cfg.data_start, end_dt)
    irx_raw = _download_ohlc_with_fallback("^IRX", ss_cfg.data_start, end_dt)
    thb_close = _extract_close(thb_raw)
    irx_close = _extract_close(irx_raw) / 100.0  # ^IRX is in percentage points

    # Align to evaluation calendar with forward-fill for minor holiday gaps
    eval_idx = r_usd_full.index
    thb_aligned = thb_close.reindex(eval_idx, method="ffill").bfill()
    irx_aligned = irx_close.reindex(eval_idx, method="ffill").bfill().clip(lower=0.0)
    th_rate = _bot_policy_rate(eval_idx)

    # FX return & daily prorated hedge carry cost
    r_fx = thb_aligned.pct_change().fillna(0.0)
    annual_hedge_cost = (irx_aligned - th_rate).clip(lower=0.0) + 0.0025  # +25 bps bank spread
    daily_hedge_cost = annual_hedge_cost / TRADING_DAYS_PER_YEAR

    # Dynamic FX Hedge signal: USD/THB < EMA100 -> Hedge (1.0), else Unhedged (0.0)
    # Monthly check + T+1 lag (zero look-ahead bias)
    thb_full_aligned = thb_close.reindex(data.closes.index, method="ffill").bfill()
    ema100 = thb_full_aligned.ewm(span=100, adjust=False).mean()
    raw_hedge_signal = (thb_full_aligned < ema100).astype(float)

    month_starts = _is_month_start(data.closes.index)
    monthly_sig = np.full(len(data.closes.index), np.nan, dtype=float)
    raw_arr = raw_hedge_signal.to_numpy(dtype=float)
    for t in range(len(month_starts)):
        if month_starts[t]:
            monthly_sig[t] = raw_arr[t]
    hedge_ratio_full = pd.Series(monthly_sig, index=data.closes.index).ffill().fillna(1.0)
    hedge_ratio_exec = hedge_ratio_full.shift(1).reindex(eval_idx).fillna(1.0)

    # 4 Scenarios
    r_usd_native = r_usd_full.copy()
    r_thb_hedged = r_usd_full - daily_hedge_cost
    r_thb_unhedged = (1.0 + r_usd_full) * (1.0 + r_fx) - 1.0

    # Dynamic blend with 10 bps switch cost when hedge_ratio changes
    switch_turnover = hedge_ratio_exec.diff().abs().fillna(0.0)
    switch_cost = switch_turnover * 0.0010  # 10 bps
    r_thb_dynamic = (
        hedge_ratio_exec * r_thb_hedged
        + (1.0 - hedge_ratio_exec) * r_thb_unhedged
        - switch_cost
    )

    rf = rf_series(eval_idx, annual_yield=RISK_FREE_ANNUAL)

    scenarios = [
        ("1. USD Native (Reference)", r_usd_native, pd.Series(0.0, index=eval_idx)),
        ("2. THB Fully Hedged (SCB -E)", r_thb_hedged, annual_hedge_cost * 100.0),
        ("3. THB Unhedged", r_thb_unhedged, pd.Series(0.0, index=eval_idx)),
        ("4. THB Dynamic FX Hedge", r_thb_dynamic, (hedge_ratio_exec * annual_hedge_cost + switch_turnover * 0.0010 * TRADING_DAYS_PER_YEAR) * 100.0),
    ]

    # Table 1: Full 2017-2026
    t1_rows = []
    for label, s_ret, s_cost in scenarios:
        m = _compute_metrics(s_ret, rf, float(s_cost.mean()))
        t1_rows.append({"Scenario": label, **m})

    print("\n### Table 1: Full 2017-2026 THB Metrics (Formula A Inv-Vol 60d, 5 bps)\n")
    print(pd.DataFrame(t1_rows).to_markdown(index=False))

    # Table 2: Period Breakdown (2017-2021 vs 2022-2026)
    t2_rows = []
    splits = [
        ("2017-2021", eval_idx[(eval_idx >= "2017-01-01") & (eval_idx <= "2021-12-31")]),
        ("2022-2026", eval_idx[eval_idx >= "2022-01-01"]),
    ]
    for period_name, sub_idx in splits:
        for label, s_ret, s_cost in scenarios:
            m = _compute_metrics(s_ret.loc[sub_idx], rf, float(s_cost.loc[sub_idx].mean()))
            t2_rows.append({"Period": period_name, "Scenario": label, **m})

    print("\n### Table 2: Period Breakdown (2017-2021 vs 2022-2026)\n")
    print(pd.DataFrame(t2_rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
