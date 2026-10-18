# -*- coding: utf-8 -*-
"""G10 Forex Signal & Carry Proxy Engine with Choppy Guard and ATR Trailing Stops.

1. Time-Series Trend Signal (Active in Trending Regime):
   - Long (+1):  Close > EMA_50 AND EMA_20 > EMA_50 AND +DI_14 > -DI_14
   - Short (-1): Close < EMA_50 AND EMA_20 < EMA_50 AND -DI_14 > +DI_14
2. Medium-Term Momentum / Carry Confirmation:
   - Strength score: ``S_mom = r_63 / sigma_20 + r_126 / sigma_60``
   - Confirms breakout direction and scales signal conviction in ``[-1.0, +1.0]``.
3. Choppy / Mean-Reverting Guard:
   - When ``is_choppy`` (``ADX_14 < 18`` and ``H_100 < 0.48``), flatten exposure
     to ``0.0`` immediately to prevent whipsaw bleed.
4. Trailing Stop Guard:
   - Exit Long if ``Close < max_20(Close) - 2.5 * ATR_14``
   - Exit Short if ``Close > min_20(Close) + 2.5 * ATR_14``
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ForexConfig, TRADING_DAYS_PER_YEAR
from .regime_filter import ForexRegimeState


@dataclass(slots=True)
class ForexSignalOutput:
    """Computed trend directions, momentum confirmation, trailing stops, and target directions."""

    ema20: pd.DataFrame
    ema50: pd.DataFrame
    atr14: pd.DataFrame
    vol20_annual: pd.DataFrame
    vol60_annual: pd.DataFrame
    mom_carry_score: pd.DataFrame    # r_63 / sigma_20 + r_126 / sigma_60
    raw_breakout_dir: pd.DataFrame   # {-1.0, 0.0, +1.0}
    long_stop_level: pd.DataFrame    # max_20(Close) - 2.5 * ATR_14
    short_stop_level: pd.DataFrame   # min_20(Close) + 2.5 * ATR_14
    direction_signal: pd.DataFrame   # final regime-filtered & stop-guarded direction in [-1.0, +1.0]


def compute_atr(
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    closes: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """Compute rolling N-day Average True Range (ATR)."""
    h = highs.astype(float)
    l = lows.astype(float)
    prev_c = closes.astype(float).shift(1)
    tr = np.maximum(
        h - l,
        np.maximum((h - prev_c).abs().fillna(h - l), (l - prev_c).abs().fillna(h - l)),
    )
    return pd.DataFrame(tr, index=closes.index, columns=closes.columns).rolling(
        window=window, min_periods=1
    ).mean()


class ForexSignalEngine:
    """Generates symmetric Long/Short G10 FX directional signals at Close(t)."""

    def __init__(self, config: ForexConfig | None = None) -> None:
        self._cfg = config or ForexConfig()

    def compute(
        self,
        highs: pd.DataFrame,
        lows: pd.DataFrame,
        closes: pd.DataFrame,
        regime: ForexRegimeState,
    ) -> ForexSignalOutput:
        cfg = self._cfg
        c = closes.astype(float)
        idx = c.index
        cols = list(c.columns)

        ema20 = c.ewm(span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast).mean()
        ema50 = c.ewm(span=cfg.ema_slow, adjust=False, min_periods=cfg.ema_slow).mean()
        atr14 = compute_atr(highs, lows, c, window=cfg.atr_window)

        daily_ret = c.pct_change(fill_method=None)
        vol20 = (
            daily_ret.rolling(window=cfg.vol_fast_window, min_periods=cfg.vol_fast_window)
            .std(ddof=1)
            * math.sqrt(TRADING_DAYS_PER_YEAR)
        )
        vol60 = (
            daily_ret.rolling(window=cfg.vol_slow_window, min_periods=cfg.vol_slow_window)
            .std(ddof=1)
            * math.sqrt(TRADING_DAYS_PER_YEAR)
        )

        r63 = c.pct_change(cfg.mom_fast_window, fill_method=None)
        r126 = c.pct_change(cfg.mom_slow_window, fill_method=None)

        vol20_safe = vol20.clip(lower=0.02)
        vol60_safe = vol60.clip(lower=0.02)
        mom_score = (r63 / vol20_safe) + (r126 / vol60_safe)

        # Raw breakout + EMA stack + DI stack
        long_cond = (c > ema50) & (ema20 > ema50) & (regime.plus_di14 > regime.minus_di14)
        short_cond = (c < ema50) & (ema20 < ema50) & (regime.minus_di14 > regime.plus_di14)

        raw_dir = pd.DataFrame(0.0, index=idx, columns=cols)
        raw_dir = raw_dir.where(~long_cond, 1.0)
        raw_dir = raw_dir.where(~short_cond, -1.0)

        # Trailing Stop Levels
        max20_close = c.rolling(window=cfg.trailing_stop_window, min_periods=1).max()
        min20_close = c.rolling(window=cfg.trailing_stop_window, min_periods=1).min()
        long_stop = max20_close - cfg.trailing_stop_atr_mult * atr14
        short_stop = min20_close + cfg.trailing_stop_atr_mult * atr14

        long_stop_hit = c < long_stop
        short_stop_hit = c > short_stop

        # Stateful path across sessions per currency pair
        raw_arr = raw_dir.to_numpy(dtype=float)
        mom_arr = mom_score.fillna(0.0).to_numpy(dtype=float)
        vol20_arr = vol20.to_numpy(dtype=float)
        trend_arr = regime.is_trending.to_numpy(dtype=bool)
        chop_arr = regime.is_choppy.to_numpy(dtype=bool)
        l_stop_arr = long_stop_hit.fillna(False).to_numpy(dtype=bool)
        s_stop_arr = short_stop_hit.fillna(False).to_numpy(dtype=bool)
        is_rebal_day = (idx.weekday == 0)

        n_dates, n_pairs = raw_arr.shape
        dir_out = np.zeros((n_dates, n_pairs), dtype=float)
        eff_vol20 = vol20.to_numpy(dtype=float).copy()
        cur_dir = np.zeros(n_pairs, dtype=float)
        locked_vol = np.full(n_pairs, np.nan, dtype=float)

        for t in range(n_dates):
            rebal = bool(is_rebal_day[t])
            for j in range(n_pairs):
                # 1. Choppy / Mean-Reverting Guard -> immediate flatten to 0.0 on any day
                if chop_arr[t, j]:
                    cur_dir[j] = 0.0
                    locked_vol[j] = np.nan
                    dir_out[t, j] = 0.0
                    continue

                # 2. Trailing Stop Guard on active positions (instant exit on any day)
                if cur_dir[j] > 0.0 and l_stop_arr[t, j]:
                    cur_dir[j] = 0.0
                    locked_vol[j] = np.nan
                elif cur_dir[j] < 0.0 and s_stop_arr[t, j]:
                    cur_dir[j] = 0.0
                    locked_vol[j] = np.nan

                # 3. Trending Regime signal evaluation & momentum/carry confirmation
                b_dir = raw_arr[t, j]
                m_val = mom_arr[t, j]
                if (cur_dir[j] > 0.0 and b_dir < 0.0) or (cur_dir[j] < 0.0 and b_dir > 0.0):
                    cur_dir[j] = 0.0
                    locked_vol[j] = np.nan

                if trend_arr[t, j] and b_dir != 0.0:
                    if cur_dir[j] == 0.0 or rebal:
                        if b_dir > 0.0 and m_val > 0.25 and (not l_stop_arr[t, j]):
                            cur_dir[j] = 1.0 if m_val >= 1.0 else 0.5
                            locked_vol[j] = vol20_arr[t, j]
                        elif b_dir < 0.0 and m_val < -0.25 and (not s_stop_arr[t, j]):
                            cur_dir[j] = -1.0 if m_val <= -1.0 else -0.5
                            locked_vol[j] = vol20_arr[t, j]
                        elif cur_dir[j] != 0.0 and (b_dir * m_val <= 0.0):
                            cur_dir[j] = 0.0
                            locked_vol[j] = np.nan

                dir_out[t, j] = cur_dir[j]
                if cur_dir[j] != 0.0 and np.isfinite(locked_vol[j]):
                    eff_vol20[t, j] = locked_vol[j]

        return ForexSignalOutput(
            ema20=ema20,
            ema50=ema50,
            atr14=atr14,
            vol20_annual=pd.DataFrame(eff_vol20, index=idx, columns=cols),
            vol60_annual=vol60,
            mom_carry_score=mom_score,
            raw_breakout_dir=raw_dir,
            long_stop_level=long_stop,
            short_stop_level=short_stop,
            direction_signal=pd.DataFrame(dir_out, index=idx, columns=cols),
        )
