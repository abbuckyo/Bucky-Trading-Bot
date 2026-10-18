# -*- coding: utf-8 -*-
"""Cross-Sectional Volatility-Adjusted Momentum Ranking Engine.

Formulation:
  * 12M-1M Return:  r_{12M-1M}(t) = Close(t - 21) / Close(t - 252) - 1.0
    (strictly skips the most recent 21 trading days to avoid short-term
    mean-reversion drag).
  * Realized Volatility: 60-day annualized sample std of daily log returns:
    sigma_{60D}(t) = std_{60}(ln(Close_t / Close_{t-1})) * sqrt(252)
  * Momentum Score:
    Score(t) = r_{12M-1M}(t) / max(sigma_{60D}(t), 0.05)
  * Ranking & Weekly Rebalance:
    Ranked descending across the universe on weekly rebalance dates (every
    Monday, or the first trading session of the week if Monday is a holiday).
    Eligible pool = Top 20% (Top Quintile / Decile up to max_active_holdings).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR


@dataclass(slots=True)
class RankingOutput:
    """Computed cross-sectional momentum scores, ranks, and weekly eligibility."""

    ret_12m_1m: pd.DataFrame       # r_{12M - 1M} = Close(t-21) / Close(t-252) - 1
    realized_vol_60d: pd.DataFrame # annualized 60d log-return volatility
    momentum_score: pd.DataFrame   # r_{12M - 1M} / max(vol_60d, 0.05)
    ranks: pd.DataFrame            # 1 = highest momentum score on date t
    eligible_pool: pd.DataFrame    # boolean mask of top-ranked stocks per date
    rebalance_mask: pd.Series      # boolean Series flagging Monday / weekly rebalance dates


def compute_12m_1m_return(
    closes: pd.DataFrame,
    lookback_long: int = 252,
    lookback_skip: int = 21,
) -> pd.DataFrame:
    """Compute Close(t - lookback_skip) / Close(t - lookback_long) - 1.0.

    Notice that prices inside ``(t - lookback_skip + 1 .. t]`` do NOT enter
    this calculation at all.
    """
    if lookback_long <= lookback_skip or lookback_skip < 0:
        raise ValueError("Require lookback_long > lookback_skip >= 0.")
    c = closes.astype(float)
    return c.shift(lookback_skip) / c.shift(lookback_long) - 1.0


def compute_realized_volatility(
    closes: pd.DataFrame,
    vol_lookback: int = 60,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Compute 60-day annualized standard deviation of daily log returns."""
    if vol_lookback < 2:
        raise ValueError("vol_lookback must be >= 2.")
    c = closes.astype(float)
    log_rets = np.log(c / c.shift(1))
    return log_rets.rolling(window=vol_lookback, min_periods=vol_lookback).std(ddof=1) * math.sqrt(
        annualization
    )


def compute_momentum_score(
    closes: pd.DataFrame,
    lookback_long: int = 252,
    lookback_skip: int = 21,
    vol_lookback: int = 60,
    vol_floor: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(ret_12m_1m, realized_vol_60d, momentum_score)``."""
    ret_12m_1m = compute_12m_1m_return(closes, lookback_long=lookback_long, lookback_skip=lookback_skip)
    vol_60d = compute_realized_volatility(closes, vol_lookback=vol_lookback)
    denom = vol_60d.clip(lower=vol_floor)
    score = ret_12m_1m / denom
    return ret_12m_1m, vol_60d, score


def weekly_rebalance_schedule(calendar: pd.DatetimeIndex) -> pd.Series:
    """Identify weekly rebalance sessions (Mondays, or first session of ISO week)."""
    if len(calendar) == 0:
        return pd.Series(dtype=bool, index=calendar)
    is_monday = calendar.weekday == 0
    # Also catch holiday-shortened weeks where Monday was closed (first session of ISO week)
    iso = calendar.isocalendar()
    year_week = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    first_of_week = ~year_week.duplicated(keep="first")
    rebalance = pd.Series(is_monday | first_of_week.to_numpy(), index=calendar, name="rebalance_day")
    return rebalance


class RankingEngine:
    """Cross-sectional momentum ranking engine."""

    def __init__(
        self,
        lookback_long: int = 252,
        lookback_skip: int = 21,
        vol_lookback: int = 60,
        vol_floor: float = 0.05,
        top_quantile: float = 0.20,
        max_active_holdings: int = 10,
    ) -> None:
        self._lookback_long = lookback_long
        self._lookback_skip = lookback_skip
        self._vol_lookback = vol_lookback
        self._vol_floor = vol_floor
        self._top_quantile = top_quantile
        self._max_active_holdings = max_active_holdings

    def compute(self, closes: pd.DataFrame) -> RankingOutput:
        ret_12m_1m, vol_60d, score = compute_momentum_score(
            closes,
            lookback_long=self._lookback_long,
            lookback_skip=self._lookback_skip,
            vol_lookback=self._vol_lookback,
            vol_floor=self._vol_floor,
        )
        # Rank descending across available stocks on each date (1.0 = best)
        ranks = score.rank(axis=1, ascending=False, method="first", na_option="bottom")
        valid_counts = score.notna().sum(axis=1)
        # Top quantile cutoff (e.g. top 20% of 30 = 6 stocks, capped by max_active_holdings)
        cutoff = (
            np.ceil(valid_counts * self._top_quantile)
            .clip(lower=1, upper=self._max_active_holdings)
        )
        eligible = (
            score.notna()
            & (score > 0.0)
            & ranks.le(cutoff, axis=0)
        )
        rebal_mask = weekly_rebalance_schedule(closes.index)
        return RankingOutput(
            ret_12m_1m=ret_12m_1m,
            realized_vol_60d=vol_60d,
            momentum_score=score,
            ranks=ranks,
            eligible_pool=eligible,
            rebalance_mask=rebal_mask,
        )
