#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 QUANT ASSET ALLOCATION BOT V8.1 -- AlphaShield Tactical Switcher
================================================================================
 Strategy : Long-only mutual fund switcher (SCB Easy App).
            Safe-haven cash park = SCBTMFPLUS-E (Money Market Fund).
 Engine   : HAA TIP Regime Canary + 2-Strike Corrupt Data Engine + 3-Tranche Scaling
            with Pre-Market Futures Guard & Anti-Chop Protection.
 Data     : 4-layer resilient pipeline (Direct Yahoo v8 Chart API via curl_cffi,
            yfinance fallback, Stooq via curl_cffi, and Finnhub Candles).
 Layout   : Mobile-first Summary-at-Top for LINE/Discord.
================================================================================
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from alphashield.data.provider import MarketDataProvider
from alphashield.engine.decision import (
    DecisionEngine,
    MarketSnapshot,
    StrategyOutcome,
)
from alphashield.notification.gateway import NotificationGateway
from alphashield.strategy.data_integrity import (
    DataHealth,
    IntegrityAction,
    IntegrityVerdict,
    assess_data_health,
    evaluate_integrity,
    is_nan,
)
from alphashield.strategy.signals import (
    TrancheDecision,
    resolve_stage,
    TRANCHE_EXPOSURE_MAP,
    TRANCHE_LABEL_MAP,
    STAGE_TO_POS,
)
from alphashield.execution.approval import (
    GateOutcome,
    evaluate_gate,
    render_approval_alert,
)
from tools.divergence_log import log_daily_divergence

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

    # SCB Order & Execution Limits
    SCB_MIN_SWITCH_THB = 1000.0       # Minimum order size required by SCB fund platform
    DEFAULT_CAPITAL_THB = 100_000.0   # Default capital for order sizing if not provided


def is_nan(val: Any) -> bool:
    """Check if a numeric value is NaN, None, or infinite."""
    if val is None:
        return True
    try:
        f = float(val)
        return math.isnan(f) or math.isinf(f)
    except (ValueError, TypeError):
        return True


def validate_sizing_for_scb(
    fund: str,
    prev_stage: int,
    target_stage: int,
    portfolio_total_thb: float,
    sleeve_pct: float = 0.20,
    min_switch_thb: float = CFG.SCB_MIN_SWITCH_THB
) -> Tuple[bool, int, str]:
    """
    SCB Minimum Switch Guard:
    Checks if the delta THB value of the tranche step meets the minimum SCB order threshold (1,000 THB).
    - If total port value is too small and delta < min_switch_thb:
      Blocks the trade/switch to prevent state.json from drifting away from the actual SCB portfolio.
    Returns: (is_valid, allowed_stage, reason)
    """
    if target_stage == prev_stage:
        return True, target_stage, "NO CHANGE"

    # Full exit to cash (stage -> 0) is always permitted by SCB fund redemptions
    if target_stage == 0:
        return True, 0, "FULL EXIT PERMITTED"

    # Compute tranche delta value
    # Each sleeve is sleeve_pct (20%) of total portfolio
    stage_step_fraction = abs(target_stage - prev_stage) / 3.0
    delta_thb = stage_step_fraction * sleeve_pct * portfolio_total_thb

    if delta_thb < min_switch_thb:
        reason = (
            f"BLOCKED: มูลค่าไม้ {delta_thb:,.2f} บาท ต่ำกว่าเกณฑ์ขั้นต่ำ SCB ({min_switch_thb:,.0f} บาท) "
            f"สำหรับพอร์ตขนาด {portfolio_total_thb:,.2f} บาท (คงสถานะเดิม T{prev_stage})"
        )
        LOG.warning("SCB Minimum Switch Guard on %s: %s", fund, reason)
        return False, prev_stage, reason

    return True, target_stage, f"VALID: มูลค่าไม้ {delta_thb:,.2f} บาท >= {min_switch_thb:,.0f} บาท"


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
    "LINE_CHANNEL_ACCESS_TOKEN": env("LINE_CHANNEL_ACCESS_TOKEN") or env("LINE_TOKEN") or env("LINE_ACCESS_TOKEN") or env("LINE_NOTIFY_TOKEN"),
    "LINE_USER_ID": env("LINE_USER_ID") or env("LINE_TARGET_ID") or env("LINE_TO"),
}

OVERRIDE_ASSET = env("OVERRIDE_ASSET", "NONE")
OVERRIDE_POSITION = env("OVERRIDE_POSITION", "NO_CHANGE")


# ------------------------------------------------------------------------------
# SECTION 3 -- HARDENED DATA PIPELINE (MarketDataProvider Deep Module)
# ------------------------------------------------------------------------------

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

def _get_curl_session():
    """Compatibility hook for curl session (preserves existing test patch points)."""
    try:
        from curl_cffi import requests as cureq
        return cureq.Session(impersonate="chrome124")
    except Exception:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"})
        return sess


DATA_PROVIDER = MarketDataProvider(
    finnhub_key=SECRETS["FINNHUB_API_KEY"],
    http_timeout=CFG.HTTP_TIMEOUT,
    history_period=CFG.HISTORY_PERIOD,
    session_factory=lambda: _get_curl_session(),
)

DECISION_ENGINE = DecisionEngine(
    sleeve_weight=0.20,
    min_order_thb=CFG.SCB_MIN_SWITCH_THB,
    default_capital_thb=CFG.DEFAULT_CAPITAL_THB,
)

NOTIFICATION_GATEWAY = NotificationGateway(
    discord_webhook_url=SECRETS["DISCORD_WEBHOOK_URL"],
    line_access_token=SECRETS["LINE_CHANNEL_ACCESS_TOKEN"],
    line_user_id=SECRETS["LINE_USER_ID"],
    discord_chunk_limit=CFG.DISCORD_CHUNK,
    line_chunk_limit=CFG.LINE_CHUNK,
    http_timeout=CFG.HTTP_TIMEOUT,
    line_timeout=CFG.LINE_TIMEOUT,
)


def _normalize_frame(df: pd.DataFrame, source: str) -> Optional[pd.DataFrame]:
    return DATA_PROVIDER.normalize_frame(df, source)


def _is_fresh(df: pd.DataFrame) -> bool:
    return DATA_PROVIDER.is_fresh(df, stale_days_max=CFG.STALE_DAYS_MAX)


def _validate(df: Optional[pd.DataFrame], source: str, ticker: str) -> Optional[pd.DataFrame]:
    return DATA_PROVIDER.validate(df, source, ticker, min_rows=CFG.MIN_ROWS, stale_days_max=CFG.STALE_DAYS_MAX)


def fetch_yahoo_chart_api(ticker: str) -> Optional[pd.DataFrame]:
    return DATA_PROVIDER.fetch_yahoo_chart_api(ticker, min_rows=CFG.MIN_ROWS, stale_days_max=CFG.STALE_DAYS_MAX)


def fetch_yfinance_lib(ticker: str) -> Optional[pd.DataFrame]:
    return DATA_PROVIDER.fetch_yfinance_lib(ticker, min_rows=CFG.MIN_ROWS, stale_days_max=CFG.STALE_DAYS_MAX)


def fetch_stooq(symbol: str) -> Optional[pd.DataFrame]:
    return DATA_PROVIDER.fetch_stooq(symbol, min_rows=CFG.MIN_ROWS, stale_days_max=CFG.STALE_DAYS_MAX)


def fetch_finnhub_candles(ticker: str) -> Optional[pd.DataFrame]:
    return DATA_PROVIDER.fetch_finnhub_candles(ticker, min_rows=CFG.MIN_ROWS, stale_days_max=CFG.STALE_DAYS_MAX)


def get_price_history(asset: Asset) -> Tuple[Optional[pd.DataFrame], str]:
    return DATA_PROVIDER.get_series(
        ticker=asset.yahoo,
        stooq_symbol=asset.stooq,
        min_rows=CFG.MIN_ROWS,
        stale_days_max=CFG.STALE_DAYS_MAX,
    )


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
    - If tip_mom <= 0 or NaN: Force 100% Cash Park (Bear / Stagflation Defense)
    - If tip_mom > 0: Canary is GREEN, allow normal staged entry
    """
    dummy_asset = Asset("TIP_CANARY", CFG.CANARY_TIP_TICKER, "TIP.US", "iShares TIPS Bond ETF")
    df, src = get_price_history(dummy_asset)
    if df is None or len(df) < 253:
        LOG.warning("Could not fetch full TIP history (rows=%d), enforcing defensive cash", len(df) if df is not None else 0)
        return False, 0.0, "TIP Canary Data Unavailable (Enforcing Defensive Cash)"

    mom = calc_momentum_13612(df["close"], weighted=False)
    if is_nan(mom):
        LOG.warning("TIP Canary computed as NaN -> Enforcing Defensive Cash Park")
        return False, 0.0, "TIP Canary NaN Detected (Enforcing Defensive Cash)"

    ok = (mom > 0.0)
    status_desc = f"TIP 13612 Mom: {mom * 100.0:+.2f}% ({'GREEN: Normal Operation' if ok else 'RED: 100% Cash Park Enforced'})"
    LOG.info("HAA Regime Canary Check: %s (source: %s)", status_desc, src)
    return ok, round(mom * 100.0, 2), status_desc


def evaluate_us_futures_guard() -> Tuple[bool, Dict[str, float], str, str]:
    """
    Evaluates Pre-Market US Futures Guard (Circuit Breaker) at 12:15 ICT.
    Fetches real-time / intraday price change of ES=F (S&P 500) and NQ=F (Nasdaq).
    Returns (circuit_triggered, {ticker: chg_pct}, alert_msg, guard_status)
    - If ES=F <= -1.2% or NQ=F <= -1.5%: Circuit breaker triggered -> PAUSE opening new tranches
    - If data cannot be retrieved: guard_status="degraded", allows normal bot execution (no silent crash)
    """
    futures_data: Dict[str, float] = {}
    triggered = False
    alert_reasons = []
    guard_status = "ok"

    try:
        tickers = [CFG.FUTURES_ES_TICKER, CFG.FUTURES_NQ_TICKER]

        for t_sym in tickers:
            quote_info = DATA_PROVIDER.get_quote(t_sym)
            if quote_info and not is_nan(quote_info.get("chg_pct")):
                futures_data[t_sym] = round(float(quote_info["chg_pct"]), 2)

        if not futures_data:
            guard_status = "degraded"
            desc = "DEGRADED: ไม่สามารถดึงข้อมูล US Futures ได้ (ทำงานตามตรรกะปกติ)"
            LOG.warning("Pre-Market Futures Guard: %s", desc)
            return False, {}, desc, guard_status

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

        return triggered, futures_data, desc, guard_status
    except Exception as exc:
        LOG.error("evaluate_us_futures_guard encountered error: %s", exc)
        return False, {}, f"Futures Check Error: {exc} (Degraded: Defaulting Normal)", "degraded"


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
        macd_hist_prev=last(m_hist, offset=-2),
        adx=last(adx_s),
        plus_di=last(pdi_s),
        minus_di=last(mdi_s),
        atr_pct=(last(atr_s) / price * 100.0) if price else 0.0,
        hv20=last(hv_s),
        hv_pct_rank=percentile_rank(hv_s),
        bull_stack=(ema50 > last(e100, price) > ema200),
        data_source=source,
    )
    return m


# ------------------------------------------------------------------------------
# SECTION 6 -- COMPOSITE SCORING ENGINE
# ------------------------------------------------------------------------------

@dataclass
class ScoreResult:
    score: float
    raw_score: float
    breakdown: bool
    breakdown_reason: str
    chop_capped: bool
    components: Dict[str, float]
    notes: List[str]


def _score_trend_structure(m: Metrics, notes: List[str]) -> float:
    pts = 0.0
    if m.bull_stack:
        pts += 20.0
        notes.append("Full Bull Stack: EMA50>100>200")
    elif m.ema50 > m.ema200:
        pts += 12.0
        notes.append("EMA50 > EMA200 (Golden structure)")
    elif m.price > m.ema200:
        pts += 6.0
        notes.append("Price above EMA200 only")

    if m.price > m.ema50:
        pts += 15.0
        notes.append("Price > EMA50")
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
    # Verify portfolio weights sum to 1.0 (Exposure denominator & cash weight assertion)
    if "portfolio" in state:
        port = state["portfolio"]
        eq_w = port.get("equity_weight", 0.0)
        cash_w = port.get("cash_weight", 0.0)
        assert abs(eq_w + cash_w - 1.0) < 1e-6, (
            f"State portfolio weights invalid: eq={eq_w}, cash={cash_w}, sum={eq_w + cash_w}"
        )

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
        raise


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
    target_exposure: float = 0.0  # Derived via lookup table {0: 0.0, 1: 1/3, 2: 2/3, 3: 1.0}
    tranche_stage: int = 0        # 0, 1, 2, 3 (Source of Truth)
    tranche_label: str = "0% (Cash Park)"
    changed: bool = False

# Lookup table for tranche exposures
TRANCHE_EXPOSURE_MAP: Dict[int, float] = {
    0: 0.0,
    1: 1.0 / 3.0,
    2: 2.0 / 3.0,
    3: 1.0,
}
TRANCHE_LABEL_MAP: Dict[int, str] = {
    0: "0% (Cash Park)",
    1: "33% (ไม้ 1/3 Starter)",
    2: "66% (ไม้ 2/3 Mid)",
    3: "100% (ไม้ 3/3 Full)",
}


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
    AlphaShield V8.0 (Beta) Decision Engine -- Strict Deterministic Precedence Order:
    0. NaN Guard -> หากตัวแปรชี้วัดทางเทคนิคเป็น NaN ให้บังคับ Force Cash Park 100% ทันที
    1. TIP Canary OFF (<= 0) -> บังคับ Force Cash 100% (ทุกสินทรัพย์ reset เป็น stage 0)
    2. Hard Breakdown (-2% EMA 200) -> บังคับ Force Exit (reset เป็น stage 0)
    3. Normal Exit/Trim (หลุด EMA 50/100) -> ลดขั้นบันได (ลดได้มากกว่า 1 ขั้นในวันเดียว: Asymmetric De-risking)
    4. Futures Guard PAUSE -> บล็อกเฉพาะการ 'เปิดไม้ใหม่หรือเพิ่มไม้' เท่านั้น (ห้ามบล็อกขาขาย/ขาลดความเสี่ยงเด็ดขาด)
    5. Tranche Entry Ladder -> ขยับขึ้นทีละ 1 ขั้น/วัน (Max +1 stage per day ป้องกันการกระโดดข้ามบันได)
    """
    prev_pos_str = str(prev_position).upper()
    # Map previous position string/code to stage integer
    if prev_pos_str in ("3", "T3", "TRANCHE_3", "FULL"):
        prev_stage = 3
    elif prev_pos_str in ("2", "T2", "TRANCHE_2"):
        prev_stage = 2
    elif prev_pos_str in ("1", "T1", "TRANCHE_1", "IN"):
        prev_stage = 1
    else:
        prev_stage = 0

    # --------------------------------------------------------------------------
    # Step 0: Precondition NaN Guard (Critical Data Integrity)
    # --------------------------------------------------------------------------
    nan_fields = []
    if is_nan(m.price): nan_fields.append("price")
    if is_nan(m.ema50): nan_fields.append("ema50")
    if is_nan(m.ema100): nan_fields.append("ema100")
    if is_nan(m.ema200): nan_fields.append("ema200")
    if is_nan(m.rsi): nan_fields.append("rsi")
    if is_nan(score): nan_fields.append("score")

    if nan_fields:
        LOG.error("CRITICAL: NaN detected in %s indicators (%s) -> Forcing 100%% Cash Park", fund, ", ".join(nan_fields))
        return Decision(
            signal="CORRUPT DATA EXIT",
            position="OUT",
            prev_position=f"T{prev_stage}" if prev_stage > 0 else "OUT",
            action="[CORRUPT CASH]",
            icon="🚨",
            action_detail=f"ตรวจพบค่า NaN ใน {', '.join(nan_fields)} บังคับตัดเข้า {CFG.CASH_FUND} เพื่อความปลอดภัย",
            target_exposure=0.0,
            tranche_stage=0,
            tranche_label="0% (Corrupt Data Cash)",
            changed=(prev_stage != 0),
        )

    # --------------------------------------------------------------------------
    # Step 1: TIP Canary OFF (<= 0) -> Atomic Reset to Stage 0
    # --------------------------------------------------------------------------
    if not canary_ok:
        stage = 0
        target_exp = TRANCHE_EXPOSURE_MAP[0]
        tranche_label = "0% (Canary Cash)"
        pos_code = "OUT"
        target_position = "OUT"
        changed = (prev_stage != 0)
        return Decision(
            signal="CANARY DEFENSE",
            position=target_position,
            prev_position=f"T{prev_stage}" if prev_stage > 0 else "OUT",
            action="[CANARY CASH]",
            icon="🛡️",
            action_detail=f"HAA TIP Canary สั่งล็อกหลบภัย 100% ใน {CFG.CASH_FUND}",
            target_exposure=target_exp,
            tranche_stage=stage,
            tranche_label=tranche_label,
            changed=changed,
        )

    # --------------------------------------------------------------------------
    # Step 2: Hard Breakdown (-2% EMA 200) -> Atomic Reset to Stage 0
    # --------------------------------------------------------------------------
    is_hard_breakdown = breakdown or (m.dist_ema200_pct < CFG.BREAKDOWN_EMA200_PCT)
    if is_hard_breakdown:
        stage = 0
        target_exp = TRANCHE_EXPOSURE_MAP[0]
        tranche_label = "0% (Breakdown Cut)"
        pos_code = "OUT"
        target_position = "OUT"
        changed = (prev_stage != 0)
        return Decision(
            signal="HARD EXIT",
            position=target_position,
            prev_position=f"T{prev_stage}" if prev_stage > 0 else "OUT",
            action="[HARD EXIT]",
            icon="🧨",
            action_detail=f"หลุดแนวรับวิกฤต EMA200 เกิน -2% ตัดขาย 100% เข้า {CFG.CASH_FUND}",
            target_exposure=target_exp,
            tranche_stage=stage,
            tranche_label=tranche_label,
            changed=changed,
        )

    # --------------------------------------------------------------------------
    # Determine raw technical stage based on EMA structure
    # --------------------------------------------------------------------------
    if m.price > m.ema200 and (m.ema50 > m.ema100 > m.ema200):
        raw_stage = 3
    elif m.price > m.ema100 and (m.ema50 > m.ema100):
        raw_stage = 2
    elif m.price > m.ema50:
        raw_stage = 1
    else:
        raw_stage = 0

    # --------------------------------------------------------------------------
    # Step 3: Normal Exit/Trim (Asymmetric De-risking: drop immediately)
    # --------------------------------------------------------------------------
    if raw_stage < prev_stage:
        stage = raw_stage
        target_exp = TRANCHE_EXPOSURE_MAP[stage]
        pos_code = "OUT" if stage == 0 else f"T{stage}"
        target_position = "IN" if stage > 0 else "OUT"
        changed = True
        if stage == 0:
            signal = "CASH PARK"
            action = "[CASH PARK]"
            icon = "⚪"
            detail = f"ราคาหลุด EMA 50 ลดพอร์ตเข้า {CFG.CASH_FUND} (Asymmetric De-risk)"
            tranche_label = "0% (Cash Park)"
        else:
            signal = "TRIM RISK"
            action = f"[TRIM: {pos_code}]"
            icon = "🔵" if stage == 2 else "🟡"
            detail = f"โครงสร้างชะลอตัว ปรับลดพอร์ตลงมาที่ไม้ {stage} ({TRANCHE_LABEL_MAP[stage]})"
            tranche_label = TRANCHE_LABEL_MAP[stage]

        return Decision(
            signal=signal,
            position=target_position,
            prev_position=f"T{prev_stage}" if prev_stage > 0 else "OUT",
            action=action,
            icon=icon,
            action_detail=detail,
            target_exposure=target_exp,
            tranche_stage=stage,
            tranche_label=tranche_label,
            changed=changed,
        )

    # --------------------------------------------------------------------------
    # Step 4: Futures Guard PAUSE (Blocks ONLY opening/increasing tranches)
    # --------------------------------------------------------------------------
    if futures_guard_triggered and raw_stage > prev_stage:
        stage = prev_stage
        target_exp = TRANCHE_EXPOSURE_MAP[stage]
        pos_code = "OUT" if stage == 0 else f"T{stage}"
        target_position = "IN" if stage > 0 else "OUT"
        changed = False
        return Decision(
            signal="FUTURES PAUSE",
            position=target_position,
            prev_position=f"T{prev_stage}" if prev_stage > 0 else "OUT",
            action=f"[PAUSE: {pos_code}]",
            icon="⚠️",
            action_detail=f"Futures Alert: ตลาดล่วงหน้าติดลบหนัก ชะลอการเปิดไม้ใหม่ คงน้ำหนักเดิม ({TRANCHE_LABEL_MAP[stage]})",
            target_exposure=target_exp,
            tranche_stage=stage,
            tranche_label=f"{TRANCHE_LABEL_MAP[stage]} (Futures Hold)",
            changed=changed,
        )

    # --------------------------------------------------------------------------
    # Step 5: Tranche Entry Ladder (Max +1 stage per day)
    # --------------------------------------------------------------------------
    if raw_stage > prev_stage:
        stage = prev_stage + 1  # Climb at most 1 stage per day
    else:
        stage = prev_stage

    target_exp = TRANCHE_EXPOSURE_MAP[stage]
    pos_code = "OUT" if stage == 0 else f"T{stage}"
    target_position = "IN" if stage > 0 else "OUT"
    changed = (stage != prev_stage)

    if stage == 3:
        signal = "FULL ALLOCATION"
        action = "[TRANCHE 3: 100%]"
        icon = "🟢"
        detail = "โครงสร้าง Bullish เต็มรูปแบบ (EMA 50>100>200) จัดสรรเต็ม 100%"
        tranche_label = TRANCHE_LABEL_MAP[3]
    elif stage == 2:
        signal = "SCALE IN" if changed else "INVESTED"
        action = "[TRANCHE 2: 66%]"
        icon = "🔵"
        detail = "เทรนด์ระยะกลางแข็งแกร่ง (Price>EMA100 & EMA50>100) ถือครองสัดส่วน 66%"
        tranche_label = TRANCHE_LABEL_MAP[2]
    elif stage == 1:
        signal = "STARTER" if changed else "INVESTED"
        action = "[TRANCHE 1: 33%]"
        icon = "🟡"
        detail = "เริ่มผ่าน EMA 50 ทยอยเปิดไม้ 1 ที่สัดส่วน 33% (คุมความเสี่ยง)"
        tranche_label = TRANCHE_LABEL_MAP[1]
    else:
        signal = "CASH PARK"
        action = "[CASH PARK]"
        icon = "⚪"
        detail = f"ราคาต่ำกว่า EMA 50 สแตนด์บายใน {CFG.CASH_FUND}"
        tranche_label = TRANCHE_LABEL_MAP[0]

    return Decision(
        signal=signal,
        position=target_position,
        prev_position=f"T{prev_stage}" if prev_stage > 0 else "OUT",
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
# SECTION 11 -- NOTIFICATIONS (NotificationGateway Deep Module)
# ------------------------------------------------------------------------------

def _chunk_text(text: str, limit: int) -> List[str]:
    return NOTIFICATION_GATEWAY.chunk_text(text, limit)


def send_to_discord(message: str) -> bool:
    return NOTIFICATION_GATEWAY.send_discord(message)


def send_to_line(message: str) -> bool:
    return NOTIFICATION_GATEWAY.send_line(message)


def broadcast(message: str) -> None:
    NOTIFICATION_GATEWAY.dispatch(title="", message=message)


# ------------------------------------------------------------------------------
# SECTION 12 -- RENDERING (SUMMARY AT TOP)
# ------------------------------------------------------------------------------

def render_executive_summary(
    summary_rows: List[Tuple[str, float, str, str, str, str, int, str]],
    canary_ok: bool,
    tip_mom: float,
    canary_desc: str,
    futures_triggered: bool = False,
    futures_desc: str = ""
) -> str:
    lines = [
        "──────── ⚡ **สรุปคำสั่งพอร์ต AlphaShield V8.1** ────────",
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
        fund, score, signal, position, icon, detail, stage, tranche_lbl = item
        sleeve_weight = (stage / 3.0) * 20.0  # Sleeve 20% of total portfolio
        total_equity_weight += sleeve_weight
        if stage > 0:
            active_allocations.append(f"{fund} ({sleeve_weight:.1f}%)")
        lines.append(f"{icon} **{fund}** | `{score:4.1f}` | **{signal}** | ไม้: `{tranche_lbl}` ({sleeve_weight:.1f}%)")
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
    futures_guard_triggered: bool = False,
    run_date: Optional[date] = None,
    now_iso: Optional[str] = None,
    capital_thb: float = CFG.DEFAULT_CAPITAL_THB,
    require_manual_confirm: bool = True,
    submitted_token: str = "",
    pending_approvals: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any], Optional[Tuple], Optional[str]]:
    prev_entry = state.get("assets", {}).get(asset.fund, {})
    prev_stage = prev_entry.get("tranche_stage", 0)
    prev_integ = prev_entry.get("integrity", {
        "corrupt_strike": prev_entry.get("corrupt_strike", 0),
        "corrupt_last_date": prev_entry.get("corrupt_last_date"),
        "corrupt_first_seen_ict": prev_entry.get("corrupt_first_seen_ict"),
    })

    if run_date is None:
        run_date = datetime.now(CFG.TZ_BANGKOK).date()

    df, source = get_price_history(asset)
    last_bar_date = None
    if df is not None and not df.empty:
        try:
            last_bar_date = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
        except Exception:
            last_bar_date = None

    m = build_metrics(df, source) if df is not None else None
    sc = compute_score(m) if m is not None else ScoreResult(score=0.0, raw_score=0.0, chop_capped=False, breakdown=False, breakdown_reason="", components={})

    snapshot = MarketSnapshot(
        fund=asset.fund,
        price=m.price if m else 0.0,
        ema50=m.ema50 if m else 0.0,
        ema100=m.ema100 if m else 0.0,
        ema200=m.ema200 if m else 0.0,
        score=sc.score,
        rsi=m.rsi if m else 50.0,
        adx=m.adx if m else 0.0,
        hv20=m.hv20 if m else 0.0,
        chg_pct=m.chg_pct if m else 0.0,
        dist_ema50_pct=m.dist_ema50_pct if m else 0.0,
        dist_ema200_pct=m.dist_ema200_pct if m else 0.0,
        bull_stack=m.bull_stack if m else False,
        breakdown=sc.breakdown,
        breakdown_reason=sc.breakdown_reason,
        canary_ok=canary_ok,
        canary_healthy=True,
        futures_guard_triggered=futures_guard_triggered,
        last_bar_date=last_bar_date,
        run_date=run_date,
        now_iso=now_iso,
        is_data_available=(df is not None),
    )

    outcome = DECISION_ENGINE.evaluate(
        snapshot=snapshot,
        current_state=state.get("assets", {}),
        config={
            "capital_thb": capital_thb,
            "require_manual_confirm": require_manual_confirm,
            "submitted_token": submitted_token,
            "pending_approvals": pending_approvals,
            "now_dt": datetime.now(CFG.TZ_BANGKOK),
        }
    )

    dec = Decision(
        signal=outcome.signal,
        position=outcome.position,
        prev_position=f"T{prev_stage}" if prev_stage > 0 else "OUT",
        action=outcome.action,
        icon=outcome.icon,
        action_detail=outcome.action_detail,
        target_exposure=outcome.exposure,
        tranche_stage=outcome.stage,
        tranche_label=outcome.tranche_label,
        changed=outcome.changed,
    )

    if df is None:
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
            "integrity": outcome.integrity.to_state() if outcome.integrity else prev_integ,
            "score": 0.0,
            "raw_score": 0.0,
            "price": 0.0,
            "chg_pct": 0.0,
            "dist_ema50_pct": 0.0,
            "dist_ema200_pct": 0.0,
            "bull_stack": False,
            "rsi": 50.0,
            "adx": 0.0,
            "hv20": 0.0,
            "chop_capped": False,
            "breakdown": False,
            "breakdown_reason": "",
            "news": [],
            "ai_analysis": "ไม่สามารถดึงข้อมูลได้ในรอบนี้",
            "data_source": "NONE",
            "components": {},
            "updated_at": datetime.now(CFG.TZ_BANGKOK).isoformat(timespec="seconds"),
        }
        strike_num = outcome.integrity.strike if outcome.integrity else 0
        block = f"⚠️ **{asset.fund}** — ไม่สามารถดึงข้อมูลราคาได้ ({dec.signal} • Strike {strike_num})"
        summary_tuple = (
            asset.fund,
            0.0,
            dec.signal,
            dec.position,
            dec.icon,
            dec.action_detail,
            dec.tranche_stage,
            dec.tranche_label,
        )
        return block, entry, summary_tuple

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
        "integrity": outcome.integrity.to_state() if outcome.integrity else prev_integ,
        "score": sc.score,
        "raw_score": sc.raw_score,
        "price": round(m.price, 4) if not is_nan(m.price) else 0.0,
        "chg_pct": round(m.chg_pct, 2) if not is_nan(m.chg_pct) else 0.0,
        "dist_ema50_pct": round(m.dist_ema50_pct, 2) if not is_nan(m.dist_ema50_pct) else 0.0,
        "dist_ema200_pct": round(m.dist_ema200_pct, 2) if not is_nan(m.dist_ema200_pct) else 0.0,
        "bull_stack": m.bull_stack,
        "rsi": round(m.rsi, 1) if not is_nan(m.rsi) else 50.0,
        "adx": round(m.adx, 1) if not is_nan(m.adx) else 0.0,
        "hv20": round(m.hv20, 1) if not is_nan(m.hv20) else 0.0,
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
        dec.tranche_stage,
        dec.tranche_label
    )
    return block, entry, summary_tuple


def main() -> int:
    now_th = datetime.now(CFG.TZ_BANGKOK)
    LOG.info("=== QUANT BOT V8.1 START @ %s ICT ===", now_th.strftime("%Y-%m-%d %H:%M:%S"))

    # 1. Macro Regime Canary (Richman HAA TIP Momentum)
    canary_ok, tip_mom, canary_desc = evaluate_tip_canary()

    # 2. Pre-Market US Futures Guard (Circuit Breaker with degraded fallback)
    futures_triggered, futures_data, futures_desc, guard_status = evaluate_us_futures_guard()

    state = load_state()
    apply_manual_override(state)
    assets_state: Dict[str, Any] = dict(state.get("assets", {}))

    # Manual Confirmation Gate settings
    require_manual_confirm: bool = state.get("require_manual_confirm", True)
    pending_approvals: Dict[str, Any] = dict(state.get("pending_approvals", {}))
    submitted_token = (os.getenv("APPROVE_TOKEN") or "").strip()
    try:
        capital_thb = float(os.getenv("CAPITAL_THB") or CFG.DEFAULT_CAPITAL_THB)
    except (ValueError, TypeError):
        capital_thb = CFG.DEFAULT_CAPITAL_THB
    approval_alerts: List[str] = []

    detail_blocks: List[str] = []
    summary_rows: List[Tuple[str, float, str, str, str, str, int, str]] = []
    failures = 0

    run_date = now_th.date()
    now_iso = now_th.isoformat(timespec="seconds")

    for idx, asset in enumerate(UNIVERSE):
        try:
            prev_st = assets_state.get(asset.fund, {}).get("tranche_stage", 0)
            block, entry, s_tuple = process_asset(
                asset,
                {"assets": assets_state},
                canary_ok=canary_ok,
                futures_guard_triggered=futures_triggered,
                run_date=run_date,
                now_iso=now_iso,
            )

            # ── Execution Approval Gate (Phase 1 & 2) ──────────────────────────
            prop_stage = entry.get("tranche_stage", 0)
            sig = entry.get("signal", "")

            # Check if approval gate is required (first SWITCH IN: prev_stage == 0 -> target_stage == 1)
            is_first_switch_in = (prev_st == 0 and prop_stage == 1 and sig == "SWITCH IN")
            if is_first_switch_in and require_manual_confirm:
                pending_dict = pending_approvals.get(asset.fund)
                verdict = evaluate_gate(
                    fund=asset.fund,
                    prev_stage=prev_st,
                    proposed_stage=prop_stage,
                    signal=sig,
                    capital_thb=capital_thb,
                    require_manual_confirm=require_manual_confirm,
                    pending=pending_dict,
                    submitted_token=submitted_token,
                    now=now_th,
                )

                if verdict.outcome == GateOutcome.APPROVED:
                    LOG.info("Execution Gate: APPROVED for %s with token (committed_stage=%d)", asset.fund, verdict.committed_stage)
                    entry["tranche_stage"] = verdict.committed_stage
                    entry["target_exposure"] = (verdict.committed_stage / 3.0) * 0.20
                    pending_approvals.pop(asset.fund, None)
                    if verdict.unlock_auto:
                        require_manual_confirm = False
                        LOG.info("Execution Gate: Auto-mode unlocked (require_manual_confirm=False)")
                elif verdict.outcome == GateOutcome.PENDING:
                    LOG.warning("Execution Gate: PENDING for %s (Token: %s)", asset.fund, verdict.ticket.token if verdict.ticket else "N/A")
                    entry["tranche_stage"] = verdict.committed_stage  # 0
                    entry["target_exposure"] = 0.0
                    entry["position"] = "OUT"
                    entry["signal"] = "PENDING APPROVAL"
                    entry["icon"] = "🔐"
                    entry["action_detail"] = f"รอการยืนยันคำสั่งจากมนุษย์ (Token: {verdict.ticket.token if verdict.ticket else 'N/A'})"
                    if verdict.ticket:
                        pending_approvals[asset.fund] = verdict.ticket.to_state()
                        alert_text = render_approval_alert(verdict, repo_url=os.getenv("GITHUB_SERVER_URL", "") + "/" + os.getenv("GITHUB_REPOSITORY", ""))
                        if alert_text:
                            approval_alerts.append(alert_text)
                    # Update summary row tuple for pending state
                    if s_tuple:
                        s_tuple = (
                            asset.fund,
                            s_tuple[1],
                            "PENDING APPROVAL",
                            "OUT",
                            "🔐",
                            entry["action_detail"],
                            0,
                            "0% (Pending Confirmation)"
                        )
                elif verdict.outcome == GateOutcome.REJECTED_SIZE:
                    LOG.warning("Execution Gate: REJECTED_SIZE for %s (%s)", asset.fund, verdict.reasons[0] if verdict.reasons else "")
                    entry["tranche_stage"] = verdict.committed_stage  # 0
                    entry["target_exposure"] = 0.0
                    entry["position"] = "OUT"
                    entry["action_detail"] = verdict.reasons[0] if verdict.reasons else "Order size too small"
                elif verdict.outcome in (GateOutcome.EXPIRED, GateOutcome.SUPERSEDED):
                    LOG.warning("Execution Gate: %s for %s (New Token: %s)", verdict.outcome, asset.fund, verdict.ticket.token if verdict.ticket else "N/A")
                    entry["tranche_stage"] = verdict.committed_stage
                    entry["target_exposure"] = 0.0
                    entry["position"] = "OUT"
                    entry["signal"] = "PENDING APPROVAL"
                    entry["icon"] = "🔐"
                    entry["action_detail"] = f"{verdict.outcome.value}: ออก Token ใหม่ ({verdict.ticket.token if verdict.ticket else 'N/A'})"
                    if verdict.ticket:
                        pending_approvals[asset.fund] = verdict.ticket.to_state()
                        alert_text = render_approval_alert(verdict, repo_url=os.getenv("GITHUB_SERVER_URL", "") + "/" + os.getenv("GITHUB_REPOSITORY", ""))
                        if alert_text:
                            approval_alerts.append(alert_text)
                    if s_tuple:
                        s_tuple = (
                            asset.fund,
                            s_tuple[1],
                            "PENDING APPROVAL",
                            "OUT",
                            "🔐",
                            entry["action_detail"],
                            0,
                            "0% (Pending Confirmation)"
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

    # 3. Compute Net Portfolio Weights & Assertion (Sleeve 20% each)
    # Net Portfolio Weight = (tranche_stage / 3.0) * 0.20
    portfolio_weights: Dict[str, float] = {}
    invested_funds = []
    for a in UNIVERSE:
        st = assets_state.get(a.fund, {}).get("tranche_stage", 0)
        w = (st / 3.0) * 0.20
        portfolio_weights[a.fund] = w
        if st > 0:
            lbl = TRANCHE_LABEL_MAP.get(st, f"Stage {st}")
            invested_funds.append(f"{a.fund} ({lbl} = {w * 100:.1f}%)")

    total_eq_weight = sum(portfolio_weights.values())
    cash_weight = 1.0 - total_eq_weight

    # Critical Assertion: sum(weights) + cash_weight == 1.0
    assert abs(sum(portfolio_weights.values()) + cash_weight - 1.0) < 1e-6, (
        f"Weight sum mismatch: eq={total_eq_weight}, cash={cash_weight}, sum={total_eq_weight + cash_weight}"
    )

    eq_pct = round(total_eq_weight * 100.0, 1)
    cash_pct = round(cash_weight * 100.0, 1)

    # 4. State ข้ามวัน (Schema v3.1 - tranche_stage as single source of truth)
    new_state = {
        "version": 3.1,
        "last_run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_run_ict": now_th.isoformat(timespec="seconds"),
        "require_manual_confirm": require_manual_confirm,
        "pending_approvals": pending_approvals,
        "canary": {
            "ticker": CFG.CANARY_TIP_TICKER,
            "tip_mom_13612_pct": tip_mom,
            "canary_ok": canary_ok,
            "description": canary_desc,
        },
        "futures_guard": {
            "triggered": futures_triggered,
            "guard_status": guard_status,
            "data": futures_data,
            "description": futures_desc,
        },
        "portfolio": {
            "equity_weight": round(total_eq_weight, 4),
            "cash_weight": round(cash_weight, 4),
            "weights": {k: round(v, 4) for k, v in portfolio_weights.items()},
        },
        "assets": assets_state,
    }
    save_state(new_state)

    # 5. Payload และเขียนไฟล์ data.enc (Master Password)
    dashboard_data = {
        "version": 3.1,
        "engine": "AlphaShield V8.1",
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
            "guard_status": guard_status,
            "data": futures_data,
            "description": futures_desc,
        },
        "execution": {
            "require_manual_confirm": require_manual_confirm,
            "pending_approvals": pending_approvals,
            "capital_thb": capital_thb,
            "gate_status": "PENDING TICKET" if pending_approvals else ("MANUAL CONFIRM" if require_manual_confirm else "AUTO"),
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

    # 5. บันทึกผลเปรียบเทียบ Divergence Monitor (Shadow Mode: Legacy V7.2 vs AlphaShield V8.1)
    try:
        log_daily_divergence(
            run_date=now_th.strftime("%Y-%m-%d"),
            assets_state=assets_state,
            tip_mom=tip_mom,
            canary_ok=canary_ok,
            futures_guard=futures_triggered,
        )
        LOG.info("Divergence Monitor: Logged daily dual-engine comparison to logs/divergence_log.csv")
    except Exception as exc:
        LOG.error("Failed to log divergence comparison: %s", exc)

    # 6. ข้อความแจ้งเตือน Discord / LINE
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
        f"_AlphaShield Engine v8.1 • HAA TIP Regime: {'GREEN' if canary_ok else 'RED'} • สมบูรณ์ {len(summary_rows)}/{len(UNIVERSE)} • ล้มเหลว {failures}_"
    ]

    final_report = "\n\n".join(report_sections)
    broadcast(final_report)

    # Send separate critical approval alerts if any tickets are pending confirmation
    for alert in approval_alerts:
        try:
            send_to_discord(alert)
        except Exception as e:
            LOG.error("Failed to send approval alert: %s", e)

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
