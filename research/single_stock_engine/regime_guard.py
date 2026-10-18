# -*- coding: utf-8 -*-
"""Macro regime filter (SPY > EMA 200) and Earnings Blackout guard.

1. Broad Market Gate:
   - Only allow opening NEW long positions if ``SPY Close > EMA 200``.
   - When ``SPY Close <= EMA 200``: freeze new entries (existing valid winners
     may be held until their individual trend or Chandelier stop exits them).
2. Earnings Blackout Guard:
   - If an asset's next earnings announcement is within ``blackout_days``
     (default 5) trading days forward from session ``t`` (i.e. 1..5 trading
     days ahead, or same day ``0..5``), new entries are blocked and any
     already-held position tightens its Chandelier stop from 3.0x ATR to
     1.5x ATR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RegimeState:
    """Time-series macro gate and cross-sectional earnings blackout masks."""

    spy_ema200: pd.Series
    macro_bullish: pd.Series          # True when SPY Close > EMA 200 (new entries allowed)
    earnings_blackout: pd.DataFrame   # True when ticker is within 5 trading days pre-earnings


def compute_spy_regime(
    spy_close: pd.Series,
    ema_window: int = 200,
) -> tuple[pd.Series, pd.Series]:
    """Compute SPY EMA(200) and boolean macro gate ``SPY Close > EMA 200``."""
    if ema_window < 1:
        raise ValueError("ema_window must be >= 1")
    s = spy_close.astype(float)
    ema = s.ewm(span=ema_window, adjust=False, min_periods=ema_window).mean()
    bullish = (s > ema).fillna(False)
    return ema.rename(f"SPY_EMA{ema_window}"), bullish.rename("macro_bullish")


def is_in_earnings_blackout(
    current_date: pd.Timestamp,
    trading_calendar: pd.DatetimeIndex,
    ticker_earnings_dates: Sequence[pd.Timestamp | str] | pd.DatetimeIndex,
    blackout_trading_days: int = 5,
) -> bool:
    """Return True if an earnings announcement falls within ``[0, blackout_trading_days]``
    trading sessions from ``current_date`` on ``trading_calendar``.
    """
    if len(ticker_earnings_dates) == 0:
        return False
    dt = pd.Timestamp(current_date).normalize()
    cal = trading_calendar.normalize()
    pos = int(cal.searchsorted(dt, side="left"))
    if pos >= len(cal):
        return False
    window_end_pos = min(len(cal) - 1, pos + blackout_trading_days)
    window_start = cal[pos]
    window_end = cal[window_end_pos]

    ed_idx = pd.to_datetime(list(ticker_earnings_dates)).normalize()
    return bool(((ed_idx >= window_start) & (ed_idx <= window_end)).any())


def compute_earnings_blackout_matrix(
    trading_calendar: pd.DatetimeIndex,
    tickers: Sequence[str],
    earnings_dates: Mapping[str, Sequence[pd.Timestamp | str] | pd.DatetimeIndex] | None = None,
    blackout_trading_days: int = 5,
) -> pd.DataFrame:
    """Build a boolean DataFrame ``(date x ticker)`` flagging the pre-earnings blackout window.

    For any trading day ``t`` at calendar index ``i``, a ticker is in blackout if it
    has an earnings announcement date ``E`` such that ``cal[i] <= E <= cal[min(N-1, i + blackout_trading_days)]``,
    meaning the announcement occurs within the next ``blackout_trading_days`` trading sessions
    (inclusive of ``t``).
    """
    cal = pd.DatetimeIndex(trading_calendar).normalize()
    n_days = len(cal)
    out = pd.DataFrame(False, index=trading_calendar, columns=list(tickers), dtype=bool)
    if not earnings_dates or n_days == 0:
        return out

    for sym in tickers:
        raw_dates = earnings_dates.get(sym, [])
        if raw_dates is None or len(raw_dates) == 0:
            continue
        ed_index = pd.to_datetime(list(raw_dates))
        if getattr(ed_index, "tz", None) is not None:
            ed_index = ed_index.tz_localize(None)
        ed_norm = ed_index.normalize().unique()

        mask = np.zeros(n_days, dtype=bool)
        for ed in ed_norm:
            # Find first trading session on or after the earnings date
            e_pos = int(cal.searchsorted(ed, side="left"))
            if e_pos >= n_days:
                continue
            # If the earnings date is on a weekend/holiday, e_pos is the next trading day;
            # the 5 trading days leading up to `ed` correspond to indices
            # [max(0, e_pos - blackout_trading_days), e_pos] if cal[e_pos] == ed,
            # or [max(0, e_pos - blackout_trading_days), e_pos - 1] if ed < cal[e_pos].
            end_idx = e_pos if cal[e_pos] == ed else e_pos - 1
            if end_idx < 0:
                continue
            start_idx = max(0, e_pos - blackout_trading_days)
            mask[start_idx : end_idx + 1] = True
        out[sym] = mask

    return out


class RegimeGuard:
    """Evaluates broad market SPY > EMA200 regime and ticker-level earnings blackouts."""

    def __init__(
        self,
        spy_ema_window: int = 200,
        earnings_blackout_days: int = 5,
    ) -> None:
        self._spy_ema_window = spy_ema_window
        self._earnings_blackout_days = earnings_blackout_days

    def evaluate(
        self,
        spy_close: pd.Series,
        tickers: Sequence[str],
        earnings_dates: Mapping[str, Sequence[pd.Timestamp | str] | pd.DatetimeIndex] | None = None,
    ) -> RegimeState:
        ema200, macro_bullish = compute_spy_regime(spy_close, self._spy_ema_window)
        blackout = compute_earnings_blackout_matrix(
            trading_calendar=spy_close.index,
            tickers=tickers,
            earnings_dates=earnings_dates,
            blackout_trading_days=self._earnings_blackout_days,
        )
        return RegimeState(
            spy_ema200=ema200,
            macro_bullish=macro_bullish,
            earnings_blackout=blackout,
        )
