\"\"\"
Quant Asset Allocation Bot - Backtest Core Engine
Calculates indicators and quant scoring identically to bot.py without lookahead bias.
\"\"\"
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    \"\"\"Exponential Moving Average matching bot.py.\"\"\"
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def wilder_smooth(series: pd.Series, length: int) -> pd.Series:
    \"\"\"Wilder smoothing function used for RSI, ATR, and ADX.\"\"\"
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    \"\"\"Wilder's RSI (14-period).\"\"\"
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_smooth(gain, length)
    avg_loss = wilder_smooth(loss, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    \"\"\"MACD Line, Signal Line, and Histogram (12, 26, 9).\"\"\"
    fast_s = ema(close, fast)
    slow_s = ema(close, slow)
    line = fast_s - slow_s
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return line, sig, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    \"\"\"True Range calculation for ATR and ADX.\"\"\"
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def adx_wilder(df: pd.DataFrame, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    \"\"\"Wilder's Directional Movement Index (ADX, +DI, -DI).\"\"\"
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr_smooth = wilder_smooth(true_range(df), length)
    plus_di = 100.0 * wilder_smooth(plus_dm, length) / tr_smooth.replace(0.0, np.nan)
    minus_di = 100.0 * wilder_smooth(minus_dm, length) / tr_smooth.replace(0.0, np.nan)
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = wilder_smooth(dx, length)
    return adx, plus_di, minus_di


def hist_volatility(close: pd.Series, window: int = 20, trading_days: int = 252) -> pd.Series:
    \"\"\"20-day annualized historical volatility.\"\"\"
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(trading_days) * 100.0


def precompute_asset_indicators(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Vectorized computation of all indicators across the entire price history.\"\"\"
    df = df.copy()
    close = df['close']
    df['ema50'] = ema(close, 50)
    df['ema100'] = ema(close, 100)
    df['ema200'] = ema(close, 200)
    df['rsi'] = rsi_wilder(close, 14)
    m_line, m_sig, m_hist = macd(close)
    df['macd_line'] = m_line
    df['macd_sig'] = m_sig
    df['macd_hist'] = m_hist
    df['macd_hist_prev'] = m_hist.shift(1)
    adx_s, pdi_s, mdi_s = adx_wilder(df, 14)
    df['adx'] = adx_s
    df['plus_di'] = pdi_s
    df['minus_di'] = mdi_s
    hv_s = hist_volatility(close, 20)
    df['hv20'] = hv_s

    def p_rank(w: np.ndarray) -> float:
        if len(w) < 30:
            return 50.0
        latest = w[-1]
        return float((np.sum(w < latest) / len(w)) * 100.0)

    df['hv_pct_rank'] = hv_s.rolling(252, min_periods=30).apply(p_rank, raw=True)
    return df


def score_bar(
    row: pd.Series,
    buy_th: float = 75.0,
    sell_th: float = 55.0,
    chop_th: float = 20.0,
    chop_cap: float = 65.0,
    breakdown_pct: float = -2.0,
    use_anti_chop: bool = True,
    use_hard_breakdown: bool = True
) -> Tuple[float, bool]:
    \"\"\"Calculates quant composite score (0-100) and detects hard breakdowns for a single bar.\"\"\"
    p = row['close']
    e50, e100, e200 = row['ema50'], row['ema100'], row['ema200']
    if pd.isna(e200) or pd.isna(row['adx']) or pd.isna(row['rsi']):
        return 0.0, False

    dist_ema200 = (p / e200 - 1.0) * 100.0 if e200 > 0 else 0.0
    dist_ema50 = (p / e50 - 1.0) * 100.0 if e50 > 0 else 0.0

    # Trend structure
    pts = 0.0
    if e50 > e100 > e200:
        pts += 15.0
    elif e50 > e200:
        pts += 7.0
    if p > e200:
        pts += 15.0
    if p > e50:
        pts += 10.0
    elif dist_ema50 > -1.5:
        pts += 4.0

    # Trend strength
    directional = row['plus_di'] > row['minus_di']
    adx_val = row['adx']
    if adx_val >= 25.0 and directional:
        pts += 15.0
    elif adx_val >= chop_th and directional:
        pts += 9.0
    elif adx_val < chop_th:
        pts += 3.0

    # MACD
    if row['macd_line'] > row['macd_sig']:
        pts += 8.0
    if row['macd_hist'] > row['macd_hist_prev']:
        pts += 4.0
    if row['macd_line'] > 0:
        pts += 3.0

    # RSI
    r = row['rsi']
    if 55.0 <= r <= 70.0:
        pts += 15.0
    elif 70.0 < r <= 78.0:
        pts += 11.0
    elif r > 78.0:
        pts += 6.0
    elif 50.0 <= r < 55.0:
        pts += 10.0
    elif 45.0 <= r < 50.0:
        pts += 5.0

    # Volatility
    hv_p = row['hv_pct_rank']
    if hv_p <= 30.0:
        pts += 15.0
    elif hv_p <= 50.0:
        pts += 12.0
    elif hv_p <= 70.0:
        pts += 8.0
    elif hv_p <= 85.0:
        pts += 4.0

    score = pts
    if use_anti_chop and adx_val < chop_th and score > chop_cap:
        score = chop_cap

    breakdown = False
    if use_hard_breakdown:
        if dist_ema200 < breakdown_pct:
            breakdown = True
        elif e50 < e200 and p < e200:
            breakdown = True
        elif p < e200 and adx_val >= 25.0 and row['minus_di'] > row['plus_di']:
            breakdown = True

    if breakdown:
        score = min(score, 25.0)

    score = round(max(0.0, min(100.0, score)), 1)
    return score, breakdown
