# -*- coding: utf-8 -*-
"""Individual Trend Guard, ATR(20), and Chandelier Trailing Exit Engine.

1. Individual Trend Guard:
   - Stock ``Close`` must be strictly above its own ``EMA 100`` (``Close > EMA 100``).
2. ATR(20) & Chandelier Trailing Exit:
   - True Range: ``TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)``
   - ``ATR_20``: 20-day rolling average of True Range.
   - Normal Chandelier Stop Price:
       ``Stop_normal(t) = max_{22}(High)(t) - 3.0 * ATR_20(t)``
   - Pre-Earnings Tightened Chandelier Stop Price (when within 5 trading days of earnings):
       ``Stop_tight(t)  = max_{22}(High)(t) - 1.5 * ATR_20(t)``
   - Exit Trigger:
       If daily ``Close(t) < Stop Price(t)``, trigger complete position exit at
       next session's open (strict T+1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RiskGuardState:
    """Precomputed technical risk guard series across the asset universe."""

    ema100: pd.DataFrame                # 100-day EMA of Close
    trend_eligible: pd.DataFrame        # True where Close > EMA 100
    atr20: pd.DataFrame                 # 20-day Average True Range
    high22: pd.DataFrame                # 22-day rolling maximum of High
    chandelier_stop_normal: pd.DataFrame  # High_22 - 3.0 * ATR_20
    chandelier_stop_tight: pd.DataFrame   # High_22 - 1.5 * ATR_20
    active_stop_price: pd.DataFrame     # Effective stop price combining earnings blackout
    stop_breached: pd.DataFrame         # True where Close < active_stop_price


def compute_ema(closes: pd.DataFrame, span: int = 100) -> pd.DataFrame:
    """Compute Exponential Moving Average along the time axis."""
    if span < 1:
        raise ValueError("EMA span must be >= 1")
    return closes.astype(float).ewm(span=span, adjust=False, min_periods=span).mean()


def compute_true_range(
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    closes: pd.DataFrame,
) -> pd.DataFrame:
    """Compute daily True Range across a multi-asset DataFrame."""
    h = highs.astype(float)
    l = lows.astype(float)
    prev_c = closes.astype(float).shift(1)

    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()

    tr = np.maximum(tr1, np.maximum(tr2.fillna(tr1), tr3.fillna(tr1)))
    return pd.DataFrame(tr, index=closes.index, columns=closes.columns)


def compute_atr(
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    closes: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Compute rolling N-day Average True Range (ATR)."""
    if window < 1:
        raise ValueError("ATR window must be >= 1")
    tr = compute_true_range(highs, lows, closes)
    return tr.rolling(window=window, min_periods=window).mean()


def compute_chandelier_stops(
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    closes: pd.DataFrame,
    atr_window: int = 20,
    high_window: int = 22,
    normal_mult: float = 3.0,
    tight_mult: float = 1.5,
    earnings_blackout: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute ATR_20, High_22, normal stop, tightened pre-earnings stop, and active stop."""
    atr20 = compute_atr(highs, lows, closes, window=atr_window)
    high22 = highs.astype(float).rolling(window=high_window, min_periods=high_window).max()
    stop_normal = high22 - normal_mult * atr20
    stop_tight = high22 - tight_mult * atr20

    if earnings_blackout is not None:
        mask = earnings_blackout.reindex_like(stop_normal).fillna(False).astype(bool)
        active_stop = stop_normal.where(~mask, stop_tight)
    else:
        active_stop = stop_normal.copy()

    return atr20, high22, stop_normal, stop_tight, active_stop


class RiskGuards:
    """Evaluates individual stock trend filter (Close > EMA100) and Chandelier stops."""

    def __init__(
        self,
        ema_window: int = 100,
        atr_window: int = 20,
        chandelier_high_window: int = 22,
        chandelier_atr_mult: float = 3.0,
        earnings_tight_atr_mult: float = 1.5,
    ) -> None:
        self._ema_window = ema_window
        self._atr_window = atr_window
        self._high_window = chandelier_high_window
        self._normal_mult = chandelier_atr_mult
        self._tight_mult = earnings_tight_atr_mult

    def evaluate(
        self,
        highs: pd.DataFrame,
        lows: pd.DataFrame,
        closes: pd.DataFrame,
        earnings_blackout: pd.DataFrame | None = None,
    ) -> RiskGuardState:
        ema100 = compute_ema(closes, span=self._ema_window)
        trend_ok = (closes.astype(float) > ema100).fillna(False)

        atr20, high22, stop_normal, stop_tight, active_stop = compute_chandelier_stops(
            highs=highs,
            lows=lows,
            closes=closes,
            atr_window=self._atr_window,
            high_window=self._high_window,
            normal_mult=self._normal_mult,
            tight_mult=self._tight_mult,
            earnings_blackout=earnings_blackout,
        )
        stop_breached = (closes.astype(float) < active_stop).fillna(False)

        return RiskGuardState(
            ema100=ema100,
            trend_eligible=trend_ok,
            atr20=atr20,
            high22=high22,
            chandelier_stop_normal=stop_normal,
            chandelier_stop_tight=stop_tight,
            active_stop_price=active_stop,
            stop_breached=stop_breached,
        )
