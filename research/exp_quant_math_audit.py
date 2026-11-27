# -*- coding: utf-8 -*-
"""PSR/DSR Overfit Audit + EWMA Inv-Vol & Portfolio Vol-Target Challenge.

Audits Deployed Formula A (Rolling 60d Inv-Vol [0.08, 0.35]) with:
  1. Probabilistic Sharpe Ratio (PSR) & Deflated Sharpe Ratio (DSR, Bailey & Lopez de Prado 2014)
  2. Challenge 1: EWMA Inv-Vol (lambda = 0.94, monthly cap refresh, [0.08, 0.35])
  3. Challenge 2: Formula A + Portfolio Vol Target 8.0% (No Leverage, T+1 lagged)
  4. Challenge 3: Formula A + Portfolio Vol Target 9.0% (No Leverage, T+1 lagged)

Evaluated on Flagship Blend 70/30 (70% Core ETF + 30% Single-Stock, 5 bps),
with zero look-ahead bias and T+1 execution.

Usage:
  python research/exp_quant_math_audit.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

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
from research.alphashield_v82.portfolio_engine import Ledger
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


def _daily_excess_returns(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> np.ndarray:
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    return (returns.dropna().to_numpy(dtype=float)) - rf_daily


def daily_sharpe(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    ex = _daily_excess_returns(returns, rf_annual=rf_annual)
    if len(ex) < 2:
        return 0.0
    sd = float(np.std(ex, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(ex) / sd)


def psr(returns: pd.Series, sr_star: float = 0.0, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012/2014).

    Parameters
    ----------
    returns : pd.Series
        Daily net portfolio returns.
    sr_star : float
        Benchmark daily Sharpe ratio threshold (in same daily units as excess-return Sharpe).
    """
    ex = _daily_excess_returns(returns, rf_annual=rf_annual)
    n = len(ex)
    if n < 3:
        return 0.0
    sd = float(np.std(ex, ddof=1))
    if sd <= 1e-12:
        return 0.0
    sr_hat = float(np.mean(ex) / sd)
    g3 = float(skew(ex, bias=False))
    g4 = float(kurtosis(ex, fisher=False, bias=False))
    denom_sq = max(1e-12, 1.0 - g3 * sr_hat + ((g4 - 1.0) / 4.0) * (sr_hat ** 2))
    z = (sr_hat - sr_star) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def dsr(
    returns: pd.Series,
    trial_sharpes: list[float],
    n_trials: int = 12,
    rf_annual: float = RISK_FREE_ANNUAL,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Computes the expected maximum daily Sharpe under the null across `n_trials`
    independent strategies using the cross-sectional variance of `trial_sharpes`
    (in daily excess-return Sharpe units), then evaluates `psr(returns, sr_star=sr_0)`.
    """
    if n_trials <= 1 or len(trial_sharpes) < 2:
        return psr(returns, sr_star=0.0, rf_annual=rf_annual)
    v_sr = float(np.var(trial_sharpes, ddof=1))
    if v_sr <= 1e-16:
        return psr(returns, sr_star=0.0, rf_annual=rf_annual)
    emc = 0.5772156649015329
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    sr_0 = math.sqrt(v_sr) * ((1.0 - emc) * z1 + emc * z2)
    return psr(returns, sr_star=sr_0, rf_annual=rf_annual)


def _download_ohlc_with_fallback(sym: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    from curl_cffi import requests as cffi_requests

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


def _build_monthly_caps_from_vol(
    vol_df: pd.DataFrame,
    month_starts: np.ndarray,
    low: float = 0.08,
    high: float = 0.35,
) -> pd.DataFrame:
    assets = list(vol_df.columns)
    caps = np.full((len(vol_df), len(assets)), np.nan, dtype=float)
    vol_arr = vol_df.to_numpy(dtype=float)
    for t in range(len(vol_df)):
        if month_starts[t]:
            v = vol_arr[t]
            if np.all(np.isfinite(v)):
                caps[t] = _normalize_clipped_inv_vol(1.0 / v, low, high)
            else:
                caps[t] = EQUAL_CAP
    return pd.DataFrame(caps, index=vol_df.index, columns=assets).ffill().fillna(EQUAL_CAP)


def build_challenge_weights(
    closes: pd.DataFrame,
    tranche_states: pd.DataFrame,
    engine: FlexibleCapPortfolioEngine,
) -> dict[str, pd.DataFrame]:
    """Build all 5 strictly causal Core ETF target weight DataFrames."""
    assets = list(tranche_states.columns)
    c = closes[assets].astype(float)
    rets = c.pct_change()
    states_arr = tranche_states.to_numpy(dtype=float)
    month_starts = _is_month_start(closes.index)

    # 0. Baseline (Equal 20%)
    w_baseline = pd.DataFrame(
        states_arr / 3.0 * EQUAL_CAP,
        index=closes.index,
        columns=assets,
    )

    # 1. Deployed Formula A: Rolling 60d Inv-Vol [0.08, 0.35]
    vol_60 = (rets.rolling(60).std() * math.sqrt(TRADING_DAYS_PER_YEAR)).clip(lower=0.05)
    caps_a = _build_monthly_caps_from_vol(vol_60, month_starts, 0.08, 0.35)
    w_formula_a = pd.DataFrame(
        (states_arr / 3.0) * caps_a.to_numpy(dtype=float),
        index=closes.index,
        columns=assets,
    )

    # 2. Challenge 1: RiskMetrics EWMA Inv-Vol (lambda=0.94, Monthly) [0.08, 0.35]
    # alpha = 1 - lambda = 0.06; strictly causal up to close t, executed at open t+1
    vol_ewma = (
        rets.ewm(alpha=0.06, min_periods=20, adjust=False).std()
        * math.sqrt(TRADING_DAYS_PER_YEAR)
    ).clip(lower=0.05)
    caps_ewma = _build_monthly_caps_from_vol(vol_ewma, month_starts, 0.08, 0.35)
    w_ewma = pd.DataFrame(
        (states_arr / 3.0) * caps_ewma.to_numpy(dtype=float),
        index=closes.index,
        columns=assets,
    )

    # 3 & 4. Challenges 2 & 3: Formula A + Portfolio Vol Target (8.0% & 9.0%, No Leverage)
    # Compute Formula A realized 20d portfolio volatility strictly backward-looking,
    # then scale total exposure by min(1.0, target_vol / realized_port_vol_20d) for T+1 execution.
    ledger_a_warmup = engine.run(w_formula_a, start=WARMUP_BACKTEST_START, cost_bps=5.0)
    realized_port_vol_20d = (
        ledger_a_warmup.returns.rolling(20).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    ).reindex(closes.index)

    def _apply_vol_target(target_vol: float) -> pd.DataFrame:
        scale = (target_vol / realized_port_vol_20d.clip(lower=1e-6)).clip(upper=1.0).fillna(1.0)
        return w_formula_a.mul(scale, axis=0)

    w_vt8 = _apply_vol_target(0.08)
    w_vt9 = _apply_vol_target(0.09)

    return {
        "0. Baseline (Equal 20%)": w_baseline,
        "1. Deployed Formula A (Rolling 60d Inv-Vol)": w_formula_a,
        "2. Challenge 1: EWMA Inv-Vol (lam=0.94, Monthly)": w_ewma,
        "3. Challenge 2: Formula A + Vol Target 8.0% (No Lev)": w_vt8,
        "4. Challenge 3: Formula A + Vol Target 9.0% (No Lev)": w_vt9,
    }


def _period_metrics(ret: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> dict[str, float]:
    if len(ret) < 10:
        return {"CAGR (%)": 0.0, "Vol (%)": 0.0, "Sharpe": 0.0, "Max DD (%)": 0.0, "Calmar": 0.0}
    cagr_val = float(cagr(ret) * 100.0)
    vol_val = float(ret.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
    mdd_val = float(max_drawdown(ret) * 100.0)
    rf_d = rf_series(ret.index, annual_yield=rf_annual)
    ex = ret - rf_d
    sharpe_val = float(ex.mean() / ex.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if ex.std(ddof=1) > 1e-12 else 0.0
    calmar_val = cagr_val / abs(mdd_val) if abs(mdd_val) > 1e-9 else 0.0
    return {
        "CAGR (%)": round(cagr_val, 2),
        "Vol (%)": round(vol_val, 2),
        "Sharpe": round(sharpe_val, 3),
        "Max DD (%)": round(mdd_val, 2),
        "Calmar": round(calmar_val, 3),
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
            time.sleep(0.5)
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

    strategies = build_challenge_weights(data.closes, sig_frame.tranche_states, engine)
    r_ss_full, r_spy_full = _load_single_stock_and_spy_returns(ss_cfg, WARMUP_BACKTEST_START)

    base_ledger = engine.run(strategies["0. Baseline (Equal 20%)"], start=EVAL_START, cost_bps=5.0)
    eval_idx = base_ledger.returns.index.intersection(r_ss_full.index).intersection(r_spy_full.index)
    r_ss = r_ss_full.loc[eval_idx]
    r_spy = r_spy_full.loc[eval_idx]
    rf = rf_series(eval_idx, annual_yield=RISK_FREE_ANNUAL)

    blend_returns: dict[str, pd.Series] = {}
    etf_ledgers: dict[str, Ledger] = {}
    for name, w_df in strategies.items():
        ledger = engine.run(w_df, start=EVAL_START, cost_bps=5.0)
        etf_ledgers[name] = ledger
        r_etf = ledger.returns.loc[eval_idx]
        blend_r, _, _ = simulate_fixed_monthly_blend(
            r_etf,
            r_ss,
            w1_target=0.70,
            w2_target=0.30,
            name=name,
            cost_bps=BLEND_COST_BPS,
        )
        blend_returns[name] = blend_r

    # Trial daily Sharpes across all evaluated strategies for DSR (N=12 trials)
    trial_daily_sharpes = [daily_sharpe(r) for r in blend_returns.values()]

    # Table 1 — Full 2017–2026 Performance & PSR/DSR Audit (Blend 70/30, 5 bps)
    t1_rows = []
    for name, blend_r in blend_returns.items():
        audit = compute_audit_row(blend_r, r_spy, rf)
        ann_turn = float(etf_ledgers[name].turnover.mean() * TRADING_DAYS_PER_YEAR * 100.0)
        psr_pct = psr(blend_r, sr_star=0.0) * 100.0
        dsr_pct = dsr(blend_r, trial_sharpes=trial_daily_sharpes, n_trials=12) * 100.0
        t1_rows.append({
            "Strategy": name,
            "CAGR (%)": round(audit["CAGR (%)"], 2),
            "Vol (%)": round(audit["Ann. Volatility (%)"], 2),
            "Sharpe (RF=2%)": round(audit["Sharpe (RF=2%)"], 3),
            "Max DD (%)": round(audit["Max Drawdown (%)"], 2),
            "Calmar": round(audit["Calmar"], 3),
            "PSR(SR>0) (%)": round(psr_pct, 2),
            "DSR (N=12 trials) (%)": round(dsr_pct, 2),
            "Ann. Turnover (%)": round(ann_turn, 1),
        })

    # Table 2 — Walk-Forward Split (2017–2021 vs 2022–2026, Blend 70/30, 5 bps)
    challenge_names = [
        "1. Deployed Formula A (Rolling 60d Inv-Vol)",
        "2. Challenge 1: EWMA Inv-Vol (lam=0.94, Monthly)",
        "3. Challenge 2: Formula A + Vol Target 8.0% (No Lev)",
        "4. Challenge 3: Formula A + Vol Target 9.0% (No Lev)",
    ]
    t2_rows = []
    for name in challenge_names:
        r = blend_returns[name]
        r_p1 = r.loc[(r.index >= "2017-01-01") & (r.index <= "2021-12-31")]
        r_p2 = r.loc[r.index >= "2022-01-01"]
        m1 = _period_metrics(r_p1)
        m2 = _period_metrics(r_p2)
        t2_rows.append({
            "Strategy": name,
            "2017-21 CAGR (%)": m1["CAGR (%)"],
            "2017-21 Vol (%)": m1["Vol (%)"],
            "2017-21 Sharpe": m1["Sharpe"],
            "2017-21 Max DD (%)": m1["Max DD (%)"],
            "2017-21 Calmar": m1["Calmar"],
            "2022-26 CAGR (%)": m2["CAGR (%)"],
            "2022-26 Vol (%)": m2["Vol (%)"],
            "2022-26 Sharpe": m2["Sharpe"],
            "2022-26 Max DD (%)": m2["Max DD (%)"],
            "2022-26 Calmar": m2["Calmar"],
        })

    # Table 3 — Crisis Stress Test (2018, 2020, 2022, 2026 Max DD % & Return %, Blend 70/30)
    crisis_years = [2018, 2020, 2022, 2026]
    t3_rows = []
    for name in challenge_names:
        r = blend_returns[name]
        row: dict[str, float | str] = {"Strategy": name}
        for yr in crisis_years:
            r_yr = r.loc[r.index.year == yr]
            if len(r_yr) > 0:
                tot_ret = float(((1.0 + r_yr).prod() - 1.0) * 100.0)
                mdd = float(max_drawdown(r_yr) * 100.0)
            else:
                tot_ret = 0.0
                mdd = 0.0
            row[f"{yr} Ret (%)"] = round(tot_ret, 2)
            row[f"{yr} Max DD (%)"] = round(mdd, 2)
        t3_rows.append(row)

    print("\n### Table 1 — Full 2017–2026 Performance & PSR/DSR Audit (Blend 70/30, 5 bps)\n")
    print(pd.DataFrame(t1_rows).to_markdown(index=False))

    print("\n### Table 2 — Walk-Forward Split (2017–2021 vs 2022–2026, Blend 70/30, 5 bps)\n")
    print(pd.DataFrame(t2_rows).to_markdown(index=False))

    print("\n### Table 3 — Crisis Stress Test (2018, 2020, 2022, 2026 Max DD % & Return %, Blend 70/30)\n")
    print(pd.DataFrame(t3_rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
