#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 QUANT ASSET ALLOCATION BOT V8.0 (Beta) -- AlphaShield Tactical Switcher
================================================================================
 Strategy : Long-only mutual fund switcher (SCB Easy App).
            Safe-haven cash park = SCBTMFPLUS-E (Money Market Fund).
 Engine   : HAA TIP Regime Canary + 3-Tranche Scaling (0%, 33%, 66%, 100%)
            with Hard Breakdown Guard (-2% below EMA200) & Anti-Chop Guard.
 Data     : 4-layer resilient pipeline (Direct Yahoo v8 Chart API via curl_cffi,
            yfinance fallback, Stooq via curl_cffi, and Finnhub Candles).
 Layout   : Mobile-first Summary-at-Top for LINE/Discord.
================================================================================
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------------------------
# SECTION 0 -- LOGGING
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
LOG = logging.getLogger("quantbot")


# ------------------------------------------------------------------------------
# SECTION 1 -- CONFIGURATION
# ------------------------------------------------------------------------------

class CFG:
    """Immutable global configuration."""

    # Timezone
    TZ_BANGKOK = timezone(timedelta(hours=7))

    # Persistence
    STATE_FILE = "state.json"

    # Data pipeline
    HISTORY_PERIOD = "3y"
    MIN_ROWS = 240
    STALE_DAYS_MAX = 10
    HTTP_TIMEOUT = 25
    INTER_ASSET_SLEEP = 1.0

    # Indicator parameters
    EMA_FAST, EMA_MID, EMA_SLOW = 50, 100, 200
    RSI_LEN = 14
    ADX_LEN = 14
    ATR_LEN = 14
    MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
    HV_WINDOW = 20
    HV_PERCENTILE_LOOKBACK = 252
    TRADING_DAYS = 252

    # Regime guardrails
    ADX_CHOP_THRESHOLD = 20.0     # Below this => choppy market
    ADX_TREND_THRESHOLD = 25.0    # Above this => confirmed trend
    CHOP_SCORE_CAP = 65.0         # Hard ceiling on choppy market
    BREAKDOWN_EMA200_PCT = -2.0   # Price >2% below EMA200 => immediate hard exit

    # Hysteresis thresholds
    BUY_THRESHOLD = 75.0
    SELL_THRESHOLD = 55.0

    # Notifications
    DISCORD_CHUNK = 1900
    LINE_CHUNK = 4900
    LINE_TIMEOUT = 8.0            # Strict timeout for LINE gateway to avoid hanging pipeline
    LINE_PUSH_ENDPOINT = "https://api.line.me/v2/bot/message/push"

    # AI Explainer (Static fallback if dynamic discovery fails)
    MODEL_FALLBACK_LIST = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
    ]
    AI_TEMPERATURE = 0.0
    AI_MAX_OUTPUT_TOKENS = 800

    # News
    NEWS_LOOKBACK_DAYS = 3
    NEWS_MAX_HEADLINES = 3

    # Safe Haven & Macro Canary
    CASH_FUND = "SCBTMFPLUS-E"
    CANARY_TIP_TICKER = "TIP"

    # Pre-Market US Futures Guard (Circuit Breaker)
    FUTURES_ES_TICKER = "ES=F"
    FUTURES_NQ_TICKER = "NQ=F"
    FUTURES_ES_DROP_LIMIT_PCT = -1.2  # If ES <= -1.2% -> pause opening new tranches
    FUTURES_NQ_DROP_LIMIT_PCT = -1.5  # If NQ <= -1.5% -> pause opening new tranches


@dataclass(frozen=True)
class Asset:
    fund: str
    yahoo: str
    stooq: str
    label: str


UNIVERSE: List[Asset] = [
    Asset("SCBWORLDE",  "URTH",   "URTH.US", "MSCI World"),
    Asset("SCBS&P500E", "CSPX.L", "CSPX.UK", "S&P 500"),
    Asset("SCBNDQ(E)",  "QQQ",    "QQQ.US",  "Nasdaq 100"),
    Asset("SCBGOLDE",   "GLD",    "GLD.US",  "Gold"),
    Asset("SCBSEMI(E)", "SMH",    "SMH.US",  "Semiconductors"),
]

# ------------------------------------------------------------------------------
# SECTION 2 -- ENVIRONMENT & SECRETS
# ------------------------------------------------------------------------------

def env(key: str, default: str = "") -> str:
    return (os.environ.get(key) or default).strip()


SECRETS = {
    "GEMINI_API_KEY": env("GEMINI_API_KEY"),
    "DISCORD_WEBHOOK_URL": env("DISCORD_WEBHOOK_URL"),
    "FINNHUB_API_KEY": env("FINNHUB_API_KEY"),
    "LINE_CHANNEL_ACCESS_TOKEN": env("LINE_CHANNEL_ACCESS_TOKEN"),
    "LINE_USER_ID": env("LINE_USER_ID"),
}

OVERRIDE_ASSET = env("OVERRIDE_ASSET", "NONE")
OVERRIDE_POSITION = env("OVERRIDE_POSITION", "NO_CHANGE")


# ------------------------------------------------------------------------------
# SECTION 3 -- HARDENED DATA PIPELINE (curl_cffi + Multi-Source)
# ------------------------------------------------------------------------------

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def _get_curl_session():
    """Create a browser-fingerprinted session to bypass Cloudflare/Yahoo 429."""
    try:
        from curl_cffi import requests as cureq
        return cureq.Session(impersonate="chrome124")
    except Exception:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"})
        return sess


def _normalize_frame(df: pd.DataFrame, source: str) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for tup in df.columns:
            parts = [str(p) for p in tup if p is not None and str(p) != ""]
            chosen = parts[0] if parts else ""
            for p in parts:
                if p.lower().replace(" ", "_") in REQUIRED_COLS + ["adj_close"]:
                    chosen = p
                    break
            flat.append(chosen)
        df.columns = flat

    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]

    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("date", "datetime", "index"):
            if cand in df.columns:
                df.index = pd.to_datetime(df[cand], errors="coerce")
                df = df.drop(columns=[cand])
                break
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

    df.index = pd.to_datetime(df.index, errors="coerce")
    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)
    except (TypeError, AttributeError):
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass

    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if "close" not in df.columns:
        LOG.warning("[%s] normalization failed: no close column", source)
        return None
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[REQUIRED_COLS]
    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    return df if not df.empty else None


def _is_fresh(df: pd.DataFrame) -> bool:
    last = df.index[-1].to_pydatetime()
    age = (datetime.utcnow() - last).days
    if age > CFG.STALE_DAYS_MAX:
        LOG.warning("Stale data: last bar %s (%d days old)", last.date(), age)
        return False
    return True


def _validate(df: Optional[pd.DataFrame], source: str, ticker: str) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    if len(df) < CFG.MIN_ROWS:
        LOG.warning("[%s] %s rejected: %d rows < %d", source, ticker, len(df), CFG.MIN_ROWS)
        return None
    if not _is_fresh(df):
        return None
    LOG.info("[%s] %s OK -> %d rows, last=%s", source, ticker, len(df), df.index[-1].date())
    return df


def fetch_yahoo_chart_api(ticker: str) -> Optional[pd.DataFrame]:
    """Primary: Direct v8 Chart API using curl_cffi Chrome impersonation (no crumbs/cookies required)."""
    session = _get_curl_session()
    endpoints = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=3y&interval=1d",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=3y&interval=1d",
    ]

    for url in endpoints:
        try:
            resp = session.get(url, timeout=CFG.HTTP_TIMEOUT)
            if resp.status_code != 200:
                continue
            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                continue
            res = result[0]
            timestamps = res.get("timestamp", [])
            quote = res.get("indicators", {}).get("quote", [{}])[0]
            adjclose = res.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", quote.get("close", []))

            if not timestamps or not quote.get("close"):
                continue

            df = pd.DataFrame({
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": adjclose if adjclose else quote.get("close", []),
                "volume": quote.get("volume", [0] * len(timestamps)),
            }, index=pd.to_datetime(timestamps, unit="s"))

            norm = _normalize_frame(df, "yahoo_chart_api")
            valid = _validate(norm, "yahoo_chart_api", ticker)
            if valid is not None:
                return valid
        except Exception as exc:
            LOG.debug("Yahoo v8 failed for %s on %s: %s", ticker, url, exc)

    return None


def fetch_yfinance_lib(ticker: str) -> Optional[pd.DataFrame]:
    """Secondary: yfinance standard download."""
    try:
        import yfinance as yf
        raw = yf.download(
            tickers=ticker,
            period=CFG.HISTORY_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
        )
        norm = _normalize_frame(raw, "yfinance_lib")
        return _validate(norm, "yfinance_lib", ticker)
    except Exception:
        return None


def fetch_stooq(symbol: str) -> Optional[pd.DataFrame]:
    """Tertiary: Stooq CSV via curl_cffi with anti-HTML validation."""
    session = _get_curl_session()
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    try:
        resp = session.get(url, timeout=CFG.HTTP_TIMEOUT)
        if resp.status_code != 200:
            return None
        text = resp.text.strip()
        if not text or text.startswith(("<", "<!DOCTYPE", "<html")) or "No data" in text[:200]:
            LOG.warning("[stooq] blocked or empty response for %s", symbol)
            return None
        raw = pd.read_csv(io.StringIO(text))
        norm = _normalize_frame(raw, "stooq")
        return _validate(norm, "stooq", symbol)
    except Exception as exc:
        LOG.warning("[stooq] fetch failed for %s: %s", symbol, exc)
        return None


def fetch_finnhub_candles(ticker: str) -> Optional[pd.DataFrame]:
    """Quaternary: Finnhub daily candles for US ETFs."""
    key = SECRETS["FINNHUB_API_KEY"]
    if not key:
        return None

    clean_sym = ticker.split(".")[0].upper()
    now_ts = int(time.time())
    start_ts = now_ts - (3 * 365 * 86400)
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={clean_sym}&resolution=D&from={start_ts}&to={now_ts}&token={key}"

    try:
        resp = requests.get(url, timeout=CFG.HTTP_TIMEOUT)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("s") != "ok" or not data.get("t"):
            return None

        df = pd.DataFrame({
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data.get("v", [0] * len(data["t"])),
        }, index=pd.to_datetime(data["t"], unit="s"))

        norm = _normalize_frame(df, "finnhub_candles")
        return _validate(norm, "finnhub_candles", ticker)
    except Exception as exc:
        LOG.warning("[finnhub] candles failed for %s: %s", clean_sym, exc)
        return None


def get_price_history(asset: Asset) -> Tuple[Optional[pd.DataFrame], str]:
    # 1. Direct Yahoo v8 Chart API (Bypasses 429 via curl_cffi)
    df = fetch_yahoo_chart_api(asset.yahoo)
    if df is not None:
        return df, "yahoo_v8"

    # 2. Standard yfinance
    df = fetch_yfinance_lib(asset.yahoo)
    if df is not None:
        return df, "yfinance"

    # 3. Stooq
    LOG.info("Falling back to Stooq for %s (%s)", asset.fund, asset.stooq)
    df = fetch_stooq(asset.stooq)
    if df is not None:
        return df, "stooq"

    # 4. Finnhub Candles
    LOG.info("Falling back to Finnhub Candles for %s (%s)", asset.fund, asset.yahoo)
    df = fetch_finnhub_candles(asset.yahoo)
    if df is not None:
        return df, "finnhub"

    LOG.error("ALL 4 DATA SOURCES FAILED for %s", asset.fund)
    return None, "none"


# ------------------------------------------------------------------------------
# SECTION 4 -- TECHNICAL INDICATORS
# ------------------------------------------------------------------------------

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def wilder_smooth(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def rsi_wilder(close: pd.Series, length: int = CFG.RSI_LEN) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_smooth(gain, length)
    avg_loss = wilder_smooth(loss, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = ema(close, CFG.MACD_FAST)
    slow = ema(close, CFG.MACD_SLOW)
    line = fast - slow
    signal = line.ewm(span=CFG.MACD_SIGNAL, adjust=False, min_periods=CFG.MACD_SIGNAL).mean()
    hist = line - signal
    return line, signal, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, length: int = CFG.ATR_LEN) -> pd.Series:
    return wilder_smooth(true_range(df), length)


def adx_wilder(df: pd.DataFrame, length: int = CFG.ADX_LEN) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr_smooth = wilder_smooth(true_range(df), length)
    plus_di = 100.0 * wilder_smooth(plus_dm, length) / tr_smooth.replace(0.0, np.nan)
    minus_di = 100.0 * wilder_smooth(minus_dm, length) / tr_smooth.replace(0.0, np.nan)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = wilder_smooth(dx, length)
    return adx, plus_di, minus_di


def hist_volatility(close: pd.Series, window: int = CFG.HV_WINDOW) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(CFG.TRADING_DAYS) * 100.0


def percentile_rank(series: pd.Series, lookback: int = CFG.HV_PERCENTILE_LOOKBACK) -> float:
    tail = series.dropna().tail(lookback)
    if len(tail) < 30:
        return 50.0
    latest = float(tail.iloc[-1])
    return float((tail < latest).sum()) / float(len(tail)) * 100.0


def calc_momentum_13612(series: pd.Series, weighted: bool = False) -> float:
    """Computes 13612 momentum (21, 63, 126, 252 days). Unweighted mean for HAA TIP."""
    s = series.dropna()
    if len(s) < 253:
        return 0.0
    p0 = float(s.iloc[-1])
    r1 = (p0 / float(s.iloc[-22])) - 1.0
    r3 = (p0 / float(s.iloc[-64])) - 1.0
    r6 = (p0 / float(s.iloc[-127])) - 1.0
    r12 = (p0 / float(s.iloc[-253])) - 1.0
    if weighted:
        return 12.0 * r1 + 4.0 * r3 + 2.0 * r6 + 1.0 * r12
    return (r1 + r3 + r6 + r12) / 4.0


def evaluate_tip_canary() -> Tuple[bool, float, str]:
    """
    Evaluates Richman HAA TIP Canary Momentum.
    Returns (canary_ok, tip_mom, description)
    - If tip_mom <= 0: Force 100% Cash Park (Bear / Stagflation Defense)
    - If tip_mom > 0: Canary is GREEN, allow normal staged entry
    """
    dummy_asset = Asset("TIP_CANARY", CFG.CANARY_TIP_TICKER, "TIP.US", "iShares TIPS Bond ETF")
    df, src = get_price_history(dummy_asset)
    if df is None or len(df) < 253:
        LOG.warning("Could not fetch full TIP history (rows=%d), defaulting canary to NEUTRAL/ON", len(df) if df is not None else 0)
        return True, 0.0, "TIP Canary Data Unavailable (Defaulting Normal)"

    mom = calc_momentum_13612(df["close"], weighted=False)
    ok = (mom > 0.0)
    status_desc = f"TIP 13612 Mom: {mom * 100.0:+.2f}% ({'GREEN: Normal Operation' if ok else 'RED: 100% Cash Park Enforced'})"
    LOG.info("HAA Regime Canary Check: %s (source: %s)", status_desc, src)
    return ok, round(mom * 100.0, 2), status_desc


def evaluate_us_futures_guard() -> Tuple[bool, Dict[str, float], str]:
    """
    Evaluates Pre-Market US Futures Guard (Circuit Breaker) at 12:15 ICT.
    Fetches real-time / intraday price change of ES=F (S&P 500) and NQ=F (Nasdaq).
    Returns (circuit_triggered, {ticker: chg_pct}, alert_msg)
    - If ES=F <= -1.2% or NQ=F <= -1.5%: Circuit breaker triggered -> PAUSE opening new tranches
    """
    futures_data: Dict[str, float] = {}
    triggered = False
    alert_reasons = []

    try:
        import yfinance as yf
        tickers = [CFG.FUTURES_ES_TICKER, CFG.FUTURES_NQ_TICKER]
        for t_sym in tickers:
            try:
                t = yf.Ticker(t_sym)
                info = getattr(t, "fast_info", None)
                last_p = getattr(info, "last_price", None)
                prev_c = getattr(info, "previous_close", None)
                if last_p is None or prev_c is None or prev_c <= 0:
                    # Fallback to history 2d
                    h = t.history(period="2d")
                    if len(h) >= 2:
                        prev_c = float(h["Close"].iloc[-2])
                        last_p = float(h["Close"].iloc[-1])
                    elif len(h) == 1:
                        prev_c = float(h["Open"].iloc[0])
                        last_p = float(h["Close"].iloc[-1])

                if last_p is not None and prev_c is not None and prev_c > 0:
                    pct = ((last_p / prev_c) - 1.0) * 100.0
                    futures_data[t_sym] = round(pct, 2)
            except Exception as e:
                LOG.warning("Failed to fetch futures quote for %s: %s", t_sym, e)

        es_pct = futures_data.get(CFG.FUTURES_ES_TICKER, 0.0)
        nq_pct = futures_data.get(CFG.FUTURES_NQ_TICKER, 0.0)

        if es_pct <= CFG.FUTURES_ES_DROP_LIMIT_PCT:
            triggered = True
            alert_reasons.append(f"ES=F {es_pct:+.2f}% (Limit: {CFG.FUTURES_ES_DROP_LIMIT_PCT}%)")
        if nq_pct <= CFG.FUTURES_NQ_DROP_LIMIT_PCT:
            triggered = True
            alert_reasons.append(f"NQ=F {nq_pct:+.2f}% (Limit: {CFG.FUTURES_NQ_DROP_LIMIT_PCT}%)")

        if triggered:
            desc = f"🚨 CIRCUIT BREAKER TRIGGERED: {', '.join(alert_reasons)} — ระงับการเปิดไม้ใหม่ 1 วัน"
            LOG.warning("Pre-Market Futures Guard: %s", desc)
        else:
            es_str = f"ES=F {es_pct:+.2f}%" if CFG.FUTURES_ES_TICKER in futures_data else "ES=F N/A"
            nq_str = f"NQ=F {nq_pct:+.2f}%" if CFG.FUTURES_NQ_TICKER in futures_data else "NQ=F N/A"
            desc = f"NORMAL: {es_str} | {nq_str} (สภาวะตลาดล่วงหน้าปกติ)"
            LOG.info("Pre-Market Futures Guard: %s", desc)

        return triggered, futures_data, desc
    except Exception as exc:
        LOG.error("evaluate_us_futures_guard encountered error: %s", exc)
        return False, {}, f"Futures Check Error: {exc} (Defaulting Normal)"


# ------------------------------------------------------------------------------
# SECTION 5 -- METRICS SNAPSHOT
# ------------------------------------------------------------------------------

@dataclass
class Metrics:
    price: float = 0.0
    prev_close: float = 0.0
    chg_pct: float = 0.0
    ema50: float = 0.0
    ema100: float = 0.0
    ema200: float = 0.0
    dist_ema200_pct: float = 0.0
    dist_ema50_pct: float = 0.0
    rsi: float = 50.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    macd_hist_prev: float = 0.0
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    atr_pct: float = 0.0
    hv20: float = 0.0
    hv_pct_rank: float = 50.0
    bull_stack: bool = False
    data_source: str = "none"


def build_metrics(df: pd.DataFrame, source: str) -> Metrics:
    close = df["close"]

    e50, e100, e200 = ema(close, CFG.EMA_FAST), ema(close, CFG.EMA_MID), ema(close, CFG.EMA_SLOW)
    rsi_s = rsi_wilder(close)
    m_line, m_sig, m_hist = macd(close)
    adx_s, pdi_s, mdi_s = adx_wilder(df)
    atr_s = atr(df)
    hv_s = hist_volatility(close)

    def last(s: pd.Series, default: float = 0.0, offset: int = -1) -> float:
        try:
            val = s.iloc[offset]
            return float(val) if pd.notna(val) else default
        except Exception:
            return default

    price = last(close)
    prev = last(close, price, -2)
    ema200 = last(e200, price)
    ema50 = last(e50, price)

    m = Metrics(
        price=price,
        prev_close=prev,
        chg_pct=((price / prev - 1.0) * 100.0) if prev else 0.0,
        ema50=ema50,
        ema100=last(e100, price),
        ema200=ema200,
        dist_ema200_pct=((price / ema200 - 1.0) * 100.0) if ema200 else 0.0,
        dist_ema50_pct=((price / ema50 - 1.0) * 100.0) if ema50 else 0.0,
        rsi=last(rsi_s, 50.0),
        macd_line=last(m_line),
        macd_signal=last(m_sig),
        macd_hist=last(m_hist),
        macd_hist_prev=last(m_hist, 0.0, -2),
        adx=last(adx_s),
        plus_di=last(pdi_s),
        minus_di=last(mdi_s),
        atr_pct=(last(atr_s) / price * 100.0) if price else 0.0,
        hv20=last(hv_s),
        hv_pct_rank=percentile_rank(hv_s),
        data_source=source,
    )
    m.bull_stack = (m.ema50 > m.ema100 > m.ema200)
    return m


# ------------------------------------------------------------------------------
# SECTION 6 -- QUANT SCORING ENGINE (0-100)
# ------------------------------------------------------------------------------

@dataclass
class ScoreResult:
    score: float = 0.0
    raw_score: float = 0.0
    breakdown: bool = False
    breakdown_reason: str = ""
    chop_capped: bool = False
    components: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _score_trend_structure(m: Metrics, notes: List[str]) -> float:
    pts = 0.0
    if m.bull_stack:
        pts += 15.0
        notes.append("EMA bull stack (50>100>200)")
    elif m.ema50 > m.ema200:
        pts += 7.0
        notes.append("Partial bull stack (50>200)")
    else:
        notes.append("EMA stack broken")

    if m.price > m.ema200:
        pts += 15.0
        notes.append(f"Price > EMA200 (+{m.dist_ema200_pct:.1f}%)")
    else:
        notes.append(f"Price < EMA200 ({m.dist_ema200_pct:.1f}%)")

    if m.price > m.ema50:
        pts += 10.0
    elif m.dist_ema50_pct > -1.5:
        pts += 4.0
        notes.append("Price clinging to EMA50")
    return pts


def _score_trend_strength(m: Metrics, notes: List[str]) -> float:
    directional = m.plus_di > m.minus_di
    if m.adx >= CFG.ADX_TREND_THRESHOLD and directional:
        notes.append(f"Strong uptrend (ADX {m.adx:.1f})")
        return 15.0
    if m.adx >= CFG.ADX_CHOP_THRESHOLD and directional:
        notes.append(f"Developing uptrend (ADX {m.adx:.1f})")
        return 9.0
    if m.adx < CFG.ADX_CHOP_THRESHOLD:
        notes.append(f"Choppy sideways (ADX {m.adx:.1f})")
        return 3.0
    notes.append(f"Bearish trend (ADX {m.adx:.1f}, -DI dominant)")
    return 0.0


def _score_macd(m: Metrics, notes: List[str]) -> float:
    pts = 0.0
    if m.macd_line > m.macd_signal:
        pts += 8.0
        notes.append("MACD > Signal")
    if m.macd_hist > m.macd_hist_prev:
        pts += 4.0
    if m.macd_line > 0:
        pts += 3.0
    if pts == 0.0:
        notes.append("MACD bearish")
    return pts


def _score_rsi(m: Metrics, notes: List[str]) -> float:
    r = m.rsi
    if 55.0 <= r <= 70.0:
        notes.append(f"RSI optimal ({r:.1f})")
        return 15.0
    if 70.0 < r <= 78.0:
        return 11.0
    if r > 78.0:
        notes.append(f"RSI overbought ({r:.1f})")
        return 6.0
    if 50.0 <= r < 55.0:
        return 10.0
    if 45.0 <= r < 50.0:
        return 5.0
    notes.append(f"RSI weak ({r:.1f})")
    return 0.0


def _score_volatility(m: Metrics, notes: List[str]) -> float:
    p = m.hv_pct_rank
    if p <= 30.0:
        notes.append(f"Calm vol (pct {p:.0f})")
        return 15.0
    if p <= 50.0:
        return 12.0
    if p <= 70.0:
        return 8.0
    if p <= 85.0:
        notes.append(f"Elevated vol (pct {p:.0f})")
        return 4.0
    notes.append(f"Vol shock (pct {p:.0f})")
    return 0.0


def _detect_breakdown(m: Metrics) -> Tuple[bool, str]:
    if m.dist_ema200_pct < CFG.BREAKDOWN_EMA200_PCT:
        return True, f"Price {m.dist_ema200_pct:.1f}% below EMA200"
    if m.ema50 < m.ema200 and m.price < m.ema200:
        return True, "Death-cross: EMA50<EMA200 and Price<EMA200"
    if m.price < m.ema200 and m.adx >= CFG.ADX_TREND_THRESHOLD and m.minus_di > m.plus_di:
        return True, "Strong downward impulse below EMA200"
    return False, ""


def compute_score(m: Metrics) -> ScoreResult:
    notes: List[str] = []
    comp = {
        "trend_structure": _score_trend_structure(m, notes),
        "trend_strength": _score_trend_strength(m, notes),
        "momentum_macd": _score_macd(m, notes),
        "momentum_rsi": _score_rsi(m, notes),
        "volatility": _score_volatility(m, notes),
    }
    raw = float(sum(comp.values()))
    score = raw

    chop_capped = False
    if m.adx < CFG.ADX_CHOP_THRESHOLD and score > CFG.CHOP_SCORE_CAP:
        score = CFG.CHOP_SCORE_CAP
        chop_capped = True
        notes.append(f"Chop cap applied (max {CFG.CHOP_SCORE_CAP:.0f})")

    breakdown, reason = _detect_breakdown(m)
    if breakdown:
        score = min(score, 25.0)
        notes.append(f"BREAKDOWN: {reason}")

    return ScoreResult(
        score=round(max(0.0, min(100.0, score)), 1),
        raw_score=round(raw, 1),
        breakdown=breakdown,
        breakdown_reason=reason,
        chop_capped=chop_capped,
        components={k: round(v, 1) for k, v in comp.items()},
        notes=notes,
    )


# ------------------------------------------------------------------------------
# SECTION 7 -- PERSISTENCE & MANUAL OVERRIDE
# ------------------------------------------------------------------------------

def load_state() -> Dict[str, Any]:
    if not os.path.exists(CFG.STATE_FILE):
        LOG.info("No state file found -> Cold start (all positions OUT)")
        return {}
    try:
        with open(CFG.STATE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        return {}
    except Exception as exc:
        LOG.warning("Failed to load state (%s) -> resetting", exc)
        return {}


def save_state(state: Dict[str, Any]) -> None:
    tmp = f"{CFG.STATE_FILE}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, CFG.STATE_FILE)
        LOG.info("State saved atomically -> %s", CFG.STATE_FILE)
    except Exception as exc:
        LOG.error("Failed to save state: %s", exc)


def apply_manual_override(state: Dict[str, Any]) -> None:
    if OVERRIDE_ASSET != "NONE" and OVERRIDE_POSITION in ("IN", "OUT"):
        assets = state.setdefault("assets", {})
        entry = assets.setdefault(OVERRIDE_ASSET, {})
        old_pos = entry.get("position", "OUT")
        entry["position"] = OVERRIDE_POSITION
        LOG.info("MANUAL OVERRIDE APPLIED: %s position forced from %s -> %s", OVERRIDE_ASSET, old_pos, OVERRIDE_POSITION)


# ------------------------------------------------------------------------------
# SECTION 8 -- HYSTERESIS STATE MACHINE
# ------------------------------------------------------------------------------

@dataclass
class Decision:
    signal: str           # SWITCH IN / INVESTED / SWITCH OUT / AVOID / WATCH / TRANCHE SCALING
    position: str         # IN / OUT / TRANCHE_1 / TRANCHE_2 / FULL
    prev_position: str    # Previous position string
    action: str           # Badge tag
    icon: str             # 🟢 / 🔵 / 🔴 / ⚪ / 🟡
    action_detail: str    # Action description
    target_exposure: float = 0.0  # 0.0, 0.33, 0.66, 1.0
    tranche_stage: int = 0        # 0, 1, 2, 3
    tranche_label: str = "0% (Cash)"
    changed: bool = False


def decide(
    score: float,
    breakdown: bool,
    prev_position: str,
    fund: str,
    m: Metrics,
    canary_ok: bool = True,
    futures_guard_triggered: bool = False
) -> Decision:
    """
    AlphaShield V8.0 (Beta) Decision Engine:
    - If canary_ok is False (TIP 13612 Mom <= 0): Enforces 100% Cash Park (Exposure 0%)
    - Hard Breakdown: Price < EMA200 by -2% -> 0.0 (Cut 100% to Cash Park)
    - Tranche 3 (100% Full): Price > EMA200 and EMA50 > EMA100 > EMA200
    - Tranche 2 (66% Scaling): Price > EMA100 and EMA50 > EMA100
    - Tranche 1 (33% Starter): Price > EMA50
    - Pre-Market US Futures Guard: If futures drop heavily (ES <= -1.2% or NQ <= -1.5%),
      pause/skip opening new tranches (stage > prev_stage), maintaining prev exposure.
    - Else: 0% (Cash Park)
    """
    prev_pos_str = str(prev_position).upper()
    # Map previous position to stage integer
    if prev_pos_str in ("T3", "TRANCHE_3", "FULL"):
        prev_stage = 3
    elif prev_pos_str in ("T2", "TRANCHE_2"):
        prev_stage = 2
    elif prev_pos_str in ("T1", "TRANCHE_1", "IN"):
        prev_stage = 1
    else:
        prev_stage = 0

    # Determine staged tranche target
    if not canary_ok:
        target_exp = 0.0
        stage = 0
        tranche_label = "0% (Canary Cash)"
    elif breakdown or m.dist_ema200_pct < CFG.BREAKDOWN_EMA200_PCT:
        target_exp = 0.0
        stage = 0
        tranche_label = "0% (Breakdown Cut)"
    elif m.price > m.ema200 and (m.ema50 > m.ema100 > m.ema200):
        target_exp = 1.0
        stage = 3
        tranche_label = "100% (ไม้ 3/3 Full)"
    elif m.price > m.ema100 and (m.ema50 > m.ema100):
        target_exp = 0.66
        stage = 2
        tranche_label = "66% (ไม้ 2/3 Mid)"
    elif m.price > m.ema50:
        target_exp = 0.33
        stage = 1
        tranche_label = "33% (ไม้ 1/3 Starter)"
    else:
        target_exp = 0.0
        stage = 0
        tranche_label = "0% (Cash Park)"

    # Pre-Market US Futures Guard: Circuit breaker pauses opening new tranches
    futures_paused = False
    if futures_guard_triggered and stage > prev_stage:
        futures_paused = True
        stage = prev_stage
        if stage == 2:
            target_exp = 0.66
            tranche_label = "66% (Futures Hold T2)"
        elif stage == 1:
            target_exp = 0.33
            tranche_label = "33% (Futures Hold T1)"
        else:
            target_exp = 0.0
            tranche_label = "0% (Futures Cash Park)"

    pos_code = "OUT" if stage == 0 else f"T{stage}"
    is_in = stage > 0
    target_position = "IN" if is_in else "OUT"
    changed = (pos_code != prev_pos_str and target_position != prev_pos_str)

    if not canary_ok:
        signal = "CANARY DEFENSE"
        action = "[CANARY CASH]"
        icon = "🛡️"
        detail = f"HAA TIP Canary สั่งล็อกหลบภัย 100% ใน {CFG.CASH_FUND}"
    elif breakdown or m.dist_ema200_pct < CFG.BREAKDOWN_EMA200_PCT:
        signal = "HARD EXIT"
        action = "[HARD EXIT]"
        icon = "🧨"
        detail = f"หลุดแนวรับวิกฤต EMA200 เกิน -2% ตัดขาย 100% เข้า {CFG.CASH_FUND}"
    elif futures_paused:
        signal = "FUTURES PAUSE"
        action = f"[PAUSE: {pos_code}]"
        icon = "⚠️"
        detail = f"Futures Alert: ตลาดล่วงหน้าติดลบหนัก ชะลอการเปิดไม้ใหม่ คงน้ำหนักเดิม ({tranche_label})"
    elif stage == 3:
        signal = "FULL ALLOCATION"
        action = "[TRANCHE 3: 100%]"
        icon = "🟢"
        detail = f"โครงสร้าง Bullish เต็มรูปแบบ (EMA 50>100>200) จัดสรรเต็ม 100%"
    elif stage == 2:
        signal = "SCALE IN"
        action = "[TRANCHE 2: 66%]"
        icon = "🔵"
        detail = f"เทรนด์ระยะกลางแข็งแกร่ง (Price>EMA100 & EMA50>100) เพิ่มไม้เป็น 66%"
    elif stage == 1:
        signal = "STARTER"
        action = "[TRANCHE 1: 33%]"
        icon = "🟡"
        detail = f"เริ่มผ่าน EMA 50 ทยอยเปิดไม้ 1 ที่สัดส่วน 33% (คุมความเสี่ยง)"
    else:
        signal = "CASH PARK"
        action = "[CASH PARK]"
        icon = "⚪"
        detail = f"ราคาต่ำกว่า EMA 50 สแตนด์บายใน {CFG.CASH_FUND}"

    return Decision(
        signal=signal,
        position=target_position,
        prev_position=prev_pos_str,
        action=action,
        icon=icon,
        action_detail=detail,
        target_exposure=target_exp,
        tranche_stage=stage,
        tranche_label=tranche_label,
        changed=changed,
    )


# ------------------------------------------------------------------------------
# SECTION 9 -- NEWS DIGEST (Finnhub)
# ------------------------------------------------------------------------------

def fetch_news(symbol: str) -> List[str]:
    key = SECRETS["FINNHUB_API_KEY"]
    if not key:
        return []

    today = datetime.utcnow().date()
    start = today - timedelta(days=CFG.NEWS_LOOKBACK_DAYS)
    clean = symbol.split(".")[0].upper()

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": clean,
                "from": start.isoformat(),
                "to": today.isoformat(),
                "token": key,
            },
            timeout=CFG.HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        items = resp.json()
        if not isinstance(items, list):
            return []
        heads: List[str] = []
        for it in items[: CFG.NEWS_MAX_HEADLINES * 3]:
            h = str(it.get("headline", "")).strip()
            if h and h not in heads:
                heads.append(h[:140])
            if len(heads) >= CFG.NEWS_MAX_HEADLINES:
                break
        return heads
    except Exception:
        return []


# ------------------------------------------------------------------------------
# SECTION 10 -- AI EXPLAINER (Dynamic Cascade)
# ------------------------------------------------------------------------------

import re

AI_SYSTEM_PROMPT = (
    "คุณคือนักวิเคราะห์เชิงปริมาณสาย Trend-Following ที่เน้นการรักษาเงินต้นของพอร์ตกองทุนรวม SCB "
    "อธิบายเหตุผลของคะแนนและคำสั่งอย่างกระชับ มั่นใจ ตรงประเด็น ไม่ใช้คำพูดคลุมเครือ "
    "ตอบเป็นภาษาไทย 2-3 บรรทัดสั้นๆ สรุปภาพรวมทางเทคนิคและข่าว ไม่เกิน 90 คำ"
)


def get_dynamic_flash_models(client) -> List[str]:
    """Query Google API for available models and return Flash models sorted descending by version."""
    try:
        discovered = []
        flash_pattern = re.compile(r"gemini-(\d+(?:\.\d+)?)-flash(?:-latest|-preview)?$", re.IGNORECASE)

        for m in client.models.list():
            supported = getattr(m, "supported_actions", []) or []
            if "generateContent" not in supported:
                continue

            name = m.name.replace("models/", "")
            match = flash_pattern.search(name)
            if match:
                version = float(match.group(1))
                discovered.append((version, name))

        if not discovered:
            return CFG.MODEL_FALLBACK_LIST

        discovered.sort(key=lambda x: x[0], reverse=True)
        unique = []
        for _, name in discovered:
            if name not in unique:
                unique.append(name)
        return unique
    except Exception as exc:
        LOG.warning("Dynamic model discovery failed (%s) -> using fallback list", exc)
        return CFG.MODEL_FALLBACK_LIST


def ai_explain(asset: Asset, m: Metrics, sc: ScoreResult, dec: Decision, news: List[str]) -> str:
    key = SECRETS["GEMINI_API_KEY"]
    if not key:
        return "_(AI explainer disabled: No API key)_"

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return "_(AI explainer unavailable: google-genai missing)_"

    news_text = "\n".join(f"- {h}" for h in news) if news else "ไม่มีข่าวสำคัญใน 3 วันนี้"
    prompt = f"""
สินทรัพย์: {asset.fund} ({asset.label})
คำสั่งของระบบ: {dec.action} ({dec.action_detail})
คะแนน Quant: {sc.score}/100 | ทรงตัวหรือหลุด EMA200: {m.dist_ema200_pct:+.1f}%
RSI: {m.rsi:.1f} | ADX: {m.adx:.1f} (+DI {m.plus_di:.1f} / -DI {m.minus_di:.1f})
ภาวะตลาดแกว่งตัว: {sc.chop_capped} | โครงสร้างพัง: {sc.breakdown} ({sc.breakdown_reason})
หัวข้อข่าวเด่น:
{news_text}

ช่วยสรุปสั้นๆ ว่าทำไมวันนี้ถึงต้องเลือกทำตามคำสั่งนี้ และเงื่อนไขสำคัญที่ต้องจับตาคืออะไร
""".strip()

    try:
        client = genai.Client(api_key=key)
    except Exception:
        return "_(AI explainer unavailable)_"

    model_cascade = get_dynamic_flash_models(client)

    for model in model_cascade:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=AI_SYSTEM_PROMPT,
                    temperature=CFG.AI_TEMPERATURE,
                    max_output_tokens=CFG.AI_MAX_OUTPUT_TOKENS,
                ),
            )
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                LOG.info("[%s] AI generated successfully with model: %s", asset.fund, model)
                return text
        except Exception as err:
            LOG.warning("[%s] Model %s failed (%s) -> cascading down", asset.fund, model, err)
            continue

    return "_(AI explainer unavailable: fallback cascade failed)_"


# ------------------------------------------------------------------------------
# SECTION 11 -- NOTIFICATIONS
# ------------------------------------------------------------------------------

def _chunk_text(text: str, limit: int) -> List[str]:
    chunks: List[str] = []
    buf = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(buf) + len(line) + 1 > limit:
            chunks.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf.strip():
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def send_to_discord(message: str) -> bool:
    url = SECRETS["DISCORD_WEBHOOK_URL"]
    if not url:
        return False
    ok = True
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    for chunk in _chunk_text(message, CFG.DISCORD_CHUNK):
        for attempt in range(3):
            try:
                r = requests.post(url, json={"content": chunk}, headers=headers, timeout=CFG.HTTP_TIMEOUT)
                if r.status_code == 429:
                    wait = float(r.json().get("retry_after", 2))
                    time.sleep(wait + 0.5)
                    continue
                r.raise_for_status()
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        else:
            ok = False
        time.sleep(0.5)
    return ok


def send_to_line(message: str) -> bool:
    # ── สวิตช์ปิด LINE ชั่วคราวเพื่อประหยัด QUOTA ──
    enable_line = os.getenv("ENABLE_LINE", "false").lower() in ("true", "1", "yes")
    if not enable_line:
        LOG.info("MOCK: ข้ามการส่ง LINE ชั่วคราวเพื่อประหยัดโควตา (ENABLE_LINE=false)")
        return True

    token = SECRETS["LINE_CHANNEL_ACCESS_TOKEN"]
    user_id = SECRETS["LINE_USER_ID"]
    if not token or not user_id:
        return False

    chunks = _chunk_text(message, CFG.LINE_CHUNK)
    if not chunks:
        return False

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    ok = True
    for start in range(0, len(chunks), 5):
        batch = chunks[start:start + 5]
        payload = {
            "to": user_id,
            "messages": [{"type": "text", "text": c} for c in batch],
        }
        for attempt in range(2):
            try:
                r = requests.post(
                    CFG.LINE_PUSH_ENDPOINT,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=CFG.LINE_TIMEOUT,
                )
                if r.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
            except Exception as e:
                LOG.warning("LINE push failed (attempt %d/2, timeout=%.1fs): %s", attempt + 1, CFG.LINE_TIMEOUT, e)
                time.sleep(1.0 * (attempt + 1))
        else:
            ok = False
        time.sleep(0.3)
    return ok


def broadcast(message: str) -> None:
    try:
        send_to_discord(message)
    except Exception as exc:
        LOG.error("Discord error: %s", exc)
    try:
        send_to_line(message)
    except Exception as exc:
        LOG.error("LINE error: %s", exc)


# ------------------------------------------------------------------------------
# SECTION 12 -- RENDERING (SUMMARY AT TOP)
# ------------------------------------------------------------------------------

def render_executive_summary(
    summary_rows: List[Tuple[str, float, str, str, str, str, float, str]],
    canary_ok: bool,
    tip_mom: float,
    canary_desc: str,
    futures_triggered: bool = False,
    futures_desc: str = ""
) -> str:
    lines = [
        "──────── ⚡ **สรุปคำสั่งพอร์ต AlphaShield V8.0 (Beta)** ────────",
    ]
    # Macro Canary status banner
    canary_icon = "🟢" if canary_ok else "🚨"
    canary_badge = "CANARY GREEN" if canary_ok else "CANARY RED (100% CASH DEFENSE)"
    lines.append(f"{canary_icon} **HAA Regime Canary:** `{canary_badge}` (TIP 13612 Mom: `{tip_mom:+.2f}%`)")

    # Pre-Market US Futures Guard banner
    if futures_triggered:
        lines.append(f"⚠️ **Pre-Market Futures Alert:** `{futures_desc}`")
    else:
        lines.append(f"🌐 **Pre-Market US Futures:** `{futures_desc}`")
    lines.append("")

    total_equity_weight = 0.0
    active_allocations = []

    for item in summary_rows:
        fund, score, signal, position, icon, detail, exp, tranche_lbl = item
        total_equity_weight += (exp / len(UNIVERSE)) * 100.0
        if exp > 0:
            active_allocations.append(f"{fund} ({tranche_lbl})")
        lines.append(f"{icon} **{fund}** | `{score:4.1f}` | **{signal}** | ไม้: `{tranche_lbl}`")
        lines.append(f"   ↳ _{detail}_")

    cash_weight = max(0.0, 100.0 - total_equity_weight)
    lines.append("")
    if active_allocations:
        lines.append(f"💼 **สัดส่วนพอร์ตจริง:** หุ้น/ทอง `{total_equity_weight:.1f}%` ({', '.join(active_allocations)}) | เงินสด `{cash_weight:.1f}%` ({CFG.CASH_FUND})")
    else:
        lines.append(f"💼 **สัดส่วนพอร์ตจริง:** ⚪ เงินสด `100.0%` พักหลุมหลบภัยใน `{CFG.CASH_FUND}`")
    lines.append("────────────────────────────────────────────────────")
    return "\n".join(lines)


def render_asset_block(asset: Asset, m: Metrics, sc: ScoreResult, dec: Decision, news: List[str], ai_text: str) -> str:
    lines = [
        f"{dec.icon} **{asset.fund}** — {asset.label} (`{asset.yahoo}`)",
        f"**ไม้จัดสรร:** `{dec.tranche_label}` | **คำสั่ง:** `{dec.action}` | **สถานะ:** `{dec.prev_position} → {dec.position}`",
        f"**ราคา:** {m.price:,.2f} ({m.chg_pct:+.2f}%) | **EMA Stack:** {'50>100>200 (สมบูรณ์)' if m.bull_stack else 'Broken'}",
        f"**vs EMA50:** {m.dist_ema50_pct:+.1f}% | **vs EMA200:** {m.dist_ema200_pct:+.1f}% | **RSI:** {m.rsi:.1f} | **ADX:** {m.adx:.1f}",
        f"**คะแนน Quant:** `{sc.score}/100` | **แหล่งข้อมูล:** `{m.data_source}`",
    ]
    if sc.chop_capped:
        lines.append("🌀 _สภาวะตลาดแกว่งตัว (ADX < 20) — ล็อกคะแนนห้ามเข้าซื้อ_")
    if sc.breakdown:
        lines.append(f"🧨 _โครงสร้างหลุดแนวรับวิกฤต: {sc.breakdown_reason}_")
    if news:
        lines.append("**พาดหัวข่าว:** " + " • ".join(news[:2]))
    lines.append(f"**วิเคราะห์:**\n{ai_text}")
    return "\n".join(lines)


def render_header(now_th: datetime) -> str:
    return (
        "═══════════════════════════════════════════════════════════════\n"
        "🛡️ **บอทเฝ้าดอย V8.0 (Beta) — STAGED TRANCHE & HAA CANARY**\n"
        f"🕛 {now_th.strftime('%Y-%m-%d %H:%M')} ICT | หลุมหลบภัย: `{CFG.CASH_FUND}`\n"
        f"กติกา: HAA TIP Canary + 3-Tranche Scaling + Pre-Market Futures Guard\n"
        "═══════════════════════════════════════════════════════════════"
    )


# ------------------------------------------------------------------------------
# SECTION 13 -- MAIN PIPELINE & MASTER ENCRYPTED EXPORT
# ------------------------------------------------------------------------------
import base64
import secrets
from pathlib import Path
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Crypto Constants (ตรงกับ index.html 100%)
PBKDF2_ITERATIONS = 100_000
KEY_LENGTH = 32
SALT_BYTES = 16
IV_BYTES = 12
TAG_BITS = 128

DEFAULT_PASSWORD = "AlphaShield@2026"
MASTER_PASSWORD = (os.getenv("DASHBOARD_PASSWORD") or DEFAULT_PASSWORD).strip()
DASHBOARD_URL = "https://abbuckyo.github.io/Bucky-Trading-Bot/"

_b64e = lambda b: base64.b64encode(b).decode("ascii")
_b64d = lambda s: base64.b64decode(str(s).strip())


def derive_key(password: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LENGTH,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(password.strip().encode("utf-8"))


def encrypt_payload(payload: dict, password: str = MASTER_PASSWORD) -> dict:
    salt = secrets.token_bytes(SALT_BYTES)
    iv = secrets.token_bytes(IV_BYTES)
    key = derive_key(password, salt)

    plaintext = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(iv, plaintext, None)

    return {
        "v": 2,
        "alg": "AES-GCM",
        "key_bits": KEY_LENGTH * 8,
        "tag_bits": TAG_BITS,
        "kdf": "PBKDF2-SHA256",
        "iterations": PBKDF2_ITERATIONS,
        "salt": _b64e(salt),
        "iv": _b64e(iv),
        "ciphertext": _b64e(ciphertext),
        "issued_at_ict": datetime.now(CFG.TZ_BANGKOK).strftime("%Y-%m-%dT%H:%M:%S"),
        "hint": "Master Password (คงที่)",
    }


def selftest_decrypt(envelope: dict, password: str, expect: dict) -> None:
    key = derive_key(password, _b64d(envelope["salt"]), envelope["iterations"])
    decrypted = AESGCM(key).decrypt(_b64d(envelope["iv"]), _b64d(envelope["ciphertext"]), None)
    back = json.loads(decrypted.decode("utf-8"))
    if len(back["assets"]) != len(expect.get("assets", [])):
        raise ValueError("Self-test: จำนวน assets ไม่ตรงกับต้นฉบับ")
    LOG.info("Self-test ผ่าน ✓ ถอดรหัสได้ %d assets", len(back["assets"]))


def write_encrypted_dashboard(payload: dict, password: str = MASTER_PASSWORD) -> Path:
    out_path = Path("docs/data.enc")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    envelope = encrypt_payload(payload, password)
    selftest_decrypt(envelope, password, payload)

    tmp = out_path.with_suffix(".enc.tmp")
    tmp.write_text(json.dumps(envelope, separators=(",", ":")), encoding="utf-8")
    tmp.replace(out_path)

    legacy_json = Path("docs/data.json")
    if legacy_json.exists():
        legacy_json.unlink()
        LOG.warning("ลบ docs/data.json ทิ้งแล้ว — กันข้อมูลรั่วไหลผ่าน Direct URL")

    LOG.info("เขียน %s สำเร็จ (%d bytes)", out_path, out_path.stat().st_size)
    return out_path


def process_asset(
    asset: Asset,
    state: Dict[str, Any],
    canary_ok: bool = True,
    futures_guard_triggered: bool = False
) -> Tuple[str, Dict[str, Any], Optional[Tuple]]:
    prev_entry = state.get("assets", {}).get(asset.fund, {})
    prev_position = prev_entry.get("position", "OUT")

    df, source = get_price_history(asset)
    if df is None:
        block = f"⚠️ **{asset.fund}** — ไม่สามารถดึงข้อมูลราคาได้จากทุกแหล่ง (คงสถานะเดิม `{prev_position}`)"
        return block, prev_entry or {"position": prev_position, "signal": "NO DATA"}, None

    m = build_metrics(df, source)
    sc = compute_score(m)
    dec = decide(sc.score, sc.breakdown, prev_position, asset.fund, m, canary_ok=canary_ok, futures_guard_triggered=futures_guard_triggered)
    news = fetch_news(asset.yahoo)
    ai_text = ai_explain(asset, m, sc, dec, news)

    entry = {
        "fund": asset.fund,
        "label": asset.label,
        "ticker": asset.yahoo,
        "position": dec.position,
        "signal": dec.signal,
        "icon": dec.icon,
        "action_detail": dec.action_detail,
        "target_exposure": dec.target_exposure,
        "tranche_stage": dec.tranche_stage,
        "tranche_label": dec.tranche_label,
        "score": sc.score,
        "raw_score": sc.raw_score,
        "price": round(m.price, 4),
        "chg_pct": round(m.chg_pct, 2),
        "dist_ema50_pct": round(m.dist_ema50_pct, 2),
        "dist_ema200_pct": round(m.dist_ema200_pct, 2),
        "bull_stack": m.bull_stack,
        "rsi": round(m.rsi, 1),
        "adx": round(m.adx, 1),
        "hv20": round(m.hv20, 1),
        "chop_capped": sc.chop_capped,
        "breakdown": sc.breakdown,
        "breakdown_reason": sc.breakdown_reason,
        "news": news[:3],
        "ai_analysis": ai_text,
        "data_source": source,
        "components": sc.components,
        "updated_at": datetime.now(CFG.TZ_BANGKOK).isoformat(timespec="seconds"),
    }

    block = render_asset_block(asset, m, sc, dec, news, ai_text)
    summary_tuple = (
        asset.fund,
        sc.score,
        dec.signal,
        dec.position,
        dec.icon,
        dec.action_detail,
        dec.target_exposure,
        dec.tranche_label
    )
    return block, entry, summary_tuple


def main() -> int:
    now_th = datetime.now(CFG.TZ_BANGKOK)
    LOG.info("=== QUANT BOT V8.0 (Beta) START @ %s ICT ===", now_th.strftime("%Y-%m-%d %H:%M:%S"))

    # 1. Macro Regime Canary (Richman HAA TIP Momentum)
    canary_ok, tip_mom, canary_desc = evaluate_tip_canary()

    # 2. Pre-Market US Futures Guard (Circuit Breaker)
    futures_triggered, futures_data, futures_desc = evaluate_us_futures_guard()

    state = load_state()
    apply_manual_override(state)
    assets_state: Dict[str, Any] = dict(state.get("assets", {}))

    detail_blocks: List[str] = []
    summary_rows: List[Tuple[str, float, str, str, str, str, float, str]] = []
    failures = 0

    for idx, asset in enumerate(UNIVERSE):
        try:
            block, entry, s_tuple = process_asset(
                asset,
                {"assets": assets_state},
                canary_ok=canary_ok,
                futures_guard_triggered=futures_triggered
            )
            detail_blocks.append(block)
            assets_state[asset.fund] = entry
            if s_tuple:
                summary_rows.append(s_tuple)
            else:
                failures += 1
        except Exception as exc:
            failures += 1
            LOG.exception("Error on %s: %s", asset.fund, exc)
            detail_blocks.append(f"⚠️ **{asset.fund}** — ระบบวิเคราะห์ขัดข้อง: `{exc}`")

        if idx < len(UNIVERSE) - 1:
            time.sleep(CFG.INTER_ASSET_SLEEP)

    # 3. State ข้ามวัน
    new_state = {
        "version": 3,
        "last_run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_run_ict": now_th.isoformat(timespec="seconds"),
        "canary": {
            "ticker": CFG.CANARY_TIP_TICKER,
            "tip_mom_13612_pct": tip_mom,
            "canary_ok": canary_ok,
            "description": canary_desc,
        },
        "futures_guard": {
            "triggered": futures_triggered,
            "data": futures_data,
            "description": futures_desc,
        },
        "assets": assets_state,
    }
    save_state(new_state)

    # 4. Payload และเขียนไฟล์ data.enc (Master Password)
    total_eq_weight = 0.0
    invested_funds = []
    for item in summary_rows:
        fund, _, _, _, _, _, exp, tr_lbl = item
        total_eq_weight += (exp / len(UNIVERSE)) * 100.0
        if exp > 0:
            invested_funds.append(f"{fund} ({tr_lbl})")

    eq_pct = round(total_eq_weight, 1)
    cash_pct = round(max(0.0, 100.0 - eq_pct), 1)

    dashboard_data = {
        "version": 3,
        "engine": "AlphaShield V8.0 (Beta)",
        "last_run_ict": now_th.isoformat(timespec="seconds"),
        "safe_haven": CFG.CASH_FUND,
        "macro_canary": {
            "model": "Richman HAA (13612 Momentum)",
            "ticker": CFG.CANARY_TIP_TICKER,
            "momentum_pct": tip_mom,
            "status": "GREEN" if canary_ok else "RED",
            "description": canary_desc,
        },
        "futures_guard": {
            "triggered": futures_triggered,
            "data": futures_data,
            "description": futures_desc,
        },
        "portfolio_status": {
            "invested_funds": invested_funds,
            "cash_park": CFG.CASH_FUND,
            "equity_weight_pct": eq_pct,
            "cash_weight_pct": cash_pct,
        },
        "assets": [
            assets_state.get(a.fund, {
                "fund": a.fund,
                "label": a.label,
                "ticker": a.yahoo,
                "score": 0.0,
                "signal": "NO DATA",
                "icon": "⚪",
                "action_detail": "ไม่มีข้อมูล",
                "position": "OUT",
                "target_exposure": 0.0,
                "tranche_stage": 0,
                "tranche_label": "0% (No Data)",
                "price": 0.0,
                "chg_pct": 0.0,
                "dist_ema50_pct": 0.0,
                "dist_ema200_pct": 0.0,
                "bull_stack": False,
                "rsi": 50.0,
                "adx": 0.0,
                "hv20": 0.0,
                "chop_capped": False,
                "news": [],
                "ai_analysis": "ไม่สามารถดึงข้อมูลได้ในรอบนี้",
            })
            for a in UNIVERSE
        ],
    }

    write_encrypted_dashboard(dashboard_data)

    # 5. ข้อความแจ้งเตือน Discord / LINE
    report_sections = [
        render_header(now_th),
        render_executive_summary(
            summary_rows,
            canary_ok=canary_ok,
            tip_mom=tip_mom,
            canary_desc=canary_desc,
            futures_triggered=futures_triggered,
            futures_desc=futures_desc
        ),
        "──────── 🔍 **รายละเอียดทางเทคนิครายสินทรัพย์** ────────\n",
        "\n\n".join(detail_blocks),
        "────────────────────────────────────────",
        "🔐 **Dashboard เข้ารหัส AES-GCM เรียบร้อยแล้ว**",
        "🔑 **เข้าดูด้วย Master Password ประจำตัว**",
        f"🌐 **Web Dashboard:** {DASHBOARD_URL}",
        f"_AlphaShield Engine v8.0 (Beta) • HAA TIP Regime: {'GREEN' if canary_ok else 'RED'} • สมบูรณ์ {len(summary_rows)}/{len(UNIVERSE)} • ล้มเหลว {failures}_"
    ]

    final_report = "\n\n".join(report_sections)
    broadcast(final_report)
    LOG.info("=== QUANT BOT COMPLETED (Failures: %d) ===", failures)
    return 1 if failures == len(UNIVERSE) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as fatal:
        LOG.exception("FATAL: %s", fatal)
        try:
            broadcast(f"🚨 **QUANT BOT FATAL ERROR**\n```\n{fatal}\n```")
        except Exception:
            pass
        sys.exit(1)
