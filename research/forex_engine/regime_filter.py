# -*- coding: utf-8 -*-
"""Orthogonal Regime Detection Filter: Wilder's ADX(14) & Rolling Hurst Exponent (H_100).

1. Wilder's ADX (``adx_wilder``, period=14):
   - Exact Wilder smoothing (alpha = 1/14) on True Range, +DM, -DM, and DX.
   - Strictly bounded in [0.0, 100.0] and handles flat/constant prices cleanly
     (returns 0.0 without NaN or ZeroDivisionError).
2. Rolling Hurst Exponent (``rolling_hurst``, window=100):
   - Multi-scale Rescaled Range (R/S) estimator over a 100-day rolling window
     computed strictly up to day ``t``.
   - Bounded in [0.0, 1.0]; flat prices default cleanly to 0.50 without NaN.
3. Regime Classification:
   - Trending Regime: ``ADX_14 > 22.0`` OR ``H_100 > 0.53``
   - Choppy / Mean-Reverting Regime: ``ADX_14 < 18.0`` AND ``H_100 < 0.48``
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ForexRegimeState:
    """Computed ADX(14), +DI(14), -DI(14), Hurst(100), and regime masks."""

    adx14: pd.DataFrame
    plus_di14: pd.DataFrame
    minus_di14: pd.DataFrame
    hurst100: pd.DataFrame
    is_trending: pd.DataFrame   # (ADX_14 > 22) | (H_100 > 0.53)
    is_choppy: pd.DataFrame     # (ADX_14 < 18) & (H_100 < 0.48)


def _wilder_smooth_1d(arr: np.ndarray, period: int) -> np.ndarray:
    """Apply Wilder's recursive smoothing (alpha = 1 / period) on a 1D float array."""
    n = len(arr)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    alpha = 1.0 / float(period)
    one_minus_alpha = 1.0 - alpha

    first_val = float(arr[0]) if np.isfinite(arr[0]) else 0.0
    out[0] = first_val
    for t in range(1, n):
        v = float(arr[t]) if np.isfinite(arr[t]) else 0.0
        out[t] = one_minus_alpha * out[t - 1] + alpha * v
    return out


def directional_indicators_wilder(
    high: pd.Series | pd.DataFrame,
    low: pd.Series | pd.DataFrame,
    close: pd.Series | pd.DataFrame,
    period: int = 14,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """Compute exact Wilder ``(adx, plus_di, minus_di)`` with alpha = 1 / period.

    Handles both ``pd.Series`` and ``pd.DataFrame`` inputs. Guaranteed free of
    ``NaN`` / ``ZeroDivisionError`` on flat prices and strictly bounded in ``[0, 100]``.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    is_series = isinstance(close, pd.Series)
    h_df = high.to_frame() if isinstance(high, pd.Series) else high.copy()
    l_df = low.to_frame() if isinstance(low, pd.Series) else low.copy()
    c_df = close.to_frame() if isinstance(close, pd.Series) else close.copy()

    h_arr = h_df.to_numpy(dtype=float)
    l_arr = l_df.to_numpy(dtype=float)
    c_arr = c_df.to_numpy(dtype=float)

    n_rows, n_cols = c_arr.shape
    adx_out = np.zeros((n_rows, n_cols), dtype=float)
    pdi_out = np.zeros((n_rows, n_cols), dtype=float)
    mdi_out = np.zeros((n_rows, n_cols), dtype=float)

    for j in range(n_cols):
        hj = h_arr[:, j]
        lj = l_arr[:, j]
        cj = c_arr[:, j]

        tr = np.zeros(n_rows, dtype=float)
        pdm = np.zeros(n_rows, dtype=float)
        mdm = np.zeros(n_rows, dtype=float)

        if n_rows > 0:
            tr[0] = max(0.0, float(hj[0] - lj[0]))
        for t in range(1, n_rows):
            up_move = float(hj[t] - hj[t - 1])
            down_move = float(lj[t - 1] - lj[t])
            pdm[t] = up_move if (up_move > down_move and up_move > 0.0) else 0.0
            mdm[t] = down_move if (down_move > up_move and down_move > 0.0) else 0.0
            tr[t] = max(
                float(hj[t] - lj[t]),
                abs(float(hj[t] - cj[t - 1])),
                abs(float(lj[t] - cj[t - 1])),
            )

        sm_tr = _wilder_smooth_1d(tr, period)
        sm_pdm = _wilder_smooth_1d(pdm, period)
        sm_mdm = _wilder_smooth_1d(mdm, period)

        with np.errstate(divide="ignore", invalid="ignore"):
            pdi = np.where(sm_tr > 1e-15, 100.0 * (sm_pdm / sm_tr), 0.0)
            mdi = np.where(sm_tr > 1e-15, 100.0 * (sm_mdm / sm_tr), 0.0)
            di_sum = pdi + mdi
            dx = np.where(di_sum > 1e-15, 100.0 * np.abs(pdi - mdi) / di_sum, 0.0)

        pdi = np.clip(np.nan_to_num(pdi, nan=0.0, posinf=100.0, neginf=0.0), 0.0, 100.0)
        mdi = np.clip(np.nan_to_num(mdi, nan=0.0, posinf=100.0, neginf=0.0), 0.0, 100.0)
        dx = np.clip(np.nan_to_num(dx, nan=0.0, posinf=100.0, neginf=0.0), 0.0, 100.0)
        adx = np.clip(_wilder_smooth_1d(dx, period), 0.0, 100.0)

        pdi_out[:, j] = pdi
        mdi_out[:, j] = mdi
        adx_out[:, j] = adx

    if is_series:
        return (
            pd.Series(adx_out[:, 0], index=c_df.index, name="ADX"),
            pd.Series(pdi_out[:, 0], index=c_df.index, name="PLUS_DI"),
            pd.Series(mdi_out[:, 0], index=c_df.index, name="MINUS_DI"),
        )
    return (
        pd.DataFrame(adx_out, index=c_df.index, columns=c_df.columns),
        pd.DataFrame(pdi_out, index=c_df.index, columns=c_df.columns),
        pd.DataFrame(mdi_out, index=c_df.index, columns=c_df.columns),
    )


def adx_wilder(
    high: pd.Series | pd.DataFrame,
    low: pd.Series | pd.DataFrame,
    close: pd.Series | pd.DataFrame,
    period: int = 14,
) -> pd.Series | pd.DataFrame:
    """Compute Wilder's ADX(period) bounded in ``[0.0, 100.0]``."""
    adx, _, _ = directional_indicators_wilder(high, low, close, period=period)
    return adx


def _hurst_rs_1d(prices: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute rolling 100-day Hurst exponent via multi-lag variance-ratio scaling."""
    n = len(prices)
    out = np.full(n, 0.50, dtype=float)
    if n < window or window < 20:
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(np.maximum(prices, 1e-12))

    lags = [k for k in (2, 5, 10, 20) if k < window // 2]
    log_lags = np.log(np.asarray([1] + lags, dtype=float))
    x_mean = float(log_lags.mean())
    x_denom = float(((log_lags - x_mean) ** 2).sum())

    for t in range(window - 1, n):
        seg = log_p[t - window + 1 : t + 1]
        d1 = np.diff(seg)
        s1 = float(np.std(d1, ddof=1))
        if not np.isfinite(s1) or s1 <= 1e-12 or x_denom <= 1e-15:
            out[t] = 0.50
            continue

        log_stds = np.empty(len(lags) + 1, dtype=float)
        log_stds[0] = math.log(s1)
        valid = True
        for idx_k, lag in enumerate(lags, start=1):
            dk = seg[lag:] - seg[:-lag]
            sk = float(np.std(dk, ddof=1))
            if not np.isfinite(sk) or sk <= 1e-12:
                valid = False
                break
            log_stds[idx_k] = math.log(sk)

        if not valid:
            out[t] = 0.50
            continue

        y_mean = float(log_stds.mean())
        h_val = float(((log_lags - x_mean) * (log_stds - y_mean)).sum() / x_denom)
        out[t] = float(np.clip(h_val, 0.0, 1.0)) if np.isfinite(h_val) else 0.50

    return out


def rolling_hurst(
    close: pd.Series | pd.DataFrame,
    window: int = 100,
) -> pd.Series | pd.DataFrame:
    """Compute rolling Hurst exponent (H_100) strictly up to day ``t``."""
    if window < 20:
        raise ValueError("Hurst rolling window must be >= 20")
    if isinstance(close, pd.Series):
        arr = _hurst_rs_1d(close.to_numpy(dtype=float), window=window)
        return pd.Series(arr, index=close.index, name=f"H_{window}")

    c_arr = close.to_numpy(dtype=float)
    n_rows, n_cols = c_arr.shape
    h_mat = np.full((n_rows, n_cols), 0.50, dtype=float)
    for j in range(n_cols):
        h_mat[:, j] = _hurst_rs_1d(c_arr[:, j], window=window)
    return pd.DataFrame(h_mat, index=close.index, columns=close.columns)


class ForexRegimeFilter:
    """Orthogonal Wilder ADX(14) + Rolling Hurst(100) regime classifier."""

    def __init__(
        self,
        adx_period: int = 14,
        hurst_window: int = 100,
        adx_trend_threshold: float = 22.0,
        hurst_trend_threshold: float = 0.53,
        adx_choppy_threshold: float = 18.0,
        hurst_choppy_threshold: float = 0.48,
    ) -> None:
        self._adx_period = adx_period
        self._hurst_window = hurst_window
        self._adx_trend = adx_trend_threshold
        self._hurst_trend = hurst_trend_threshold
        self._adx_choppy = adx_choppy_threshold
        self._hurst_choppy = hurst_choppy_threshold

    def evaluate(
        self,
        highs: pd.DataFrame,
        lows: pd.DataFrame,
        closes: pd.DataFrame,
    ) -> ForexRegimeState:
        adx14, pdi14, mdi14 = directional_indicators_wilder(
            highs, lows, closes, period=self._adx_period
        )
        hurst100 = rolling_hurst(closes, window=self._hurst_window)

        is_choppy = (adx14 < self._adx_choppy) & (hurst100 < self._hurst_choppy)
        is_trending = ((adx14 > self._adx_trend) | (hurst100 > self._hurst_trend)) & (~is_choppy)

        return ForexRegimeState(
            adx14=adx14,
            plus_di14=pdi14,
            minus_di14=mdi14,
            hurst100=hurst100,
            is_trending=is_trending,
            is_choppy=is_choppy,
        )
