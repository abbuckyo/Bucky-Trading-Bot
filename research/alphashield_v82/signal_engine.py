# -*- coding: utf-8 -*-
"""Module B — Signal & canary engine.

Vectorized EMA(50/100/200) computation, daily unweighted TIP 13612 momentum
canary, and a pure path-dependent state machine implementing the sequential
tranche ladder.

Invariants (not configurable):
* TIP 13612 momentum <= 0  ->  ALL risky assets forced to 0% exposure.
* Hard exit: Close < EMA200 * 0.98 -> exposure 0%.
* Accumulation: max +1 tranche step per trading day.
* De-risking: instant drop to the highest currently valid rung.
* Equal-cap: per-asset weight <= 20% (tranche / 3 * 20%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EQUAL_CAP, SignalConfig
from .data_engine import MarketData

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Pure signal primitives
# ---------------------------------------------------------------------------

def tip_momentum_13612(close: pd.Series) -> pd.Series:
    """Unweighted 13612 momentum on trading days: r21 + r63 + r126 + r252.

    Pre-warm-up values are NaN (unknown), which the canary treats as unsafe.
    """
    windows = (21, 63, 126, 252)
    parts = [close.pct_change(w, fill_method=None) for w in windows]
    momentum = pd.concat(parts, axis=1).sum(axis=1, skipna=True)
    valid = pd.concat(parts, axis=1).notna().all(axis=1)
    return momentum.where(valid)


def resolve_desired_state(
    canary_safe: bool,
    close: float,
    ema_fast: float,
    ema_mid: float,
    ema_long: float,
    hard_exit_buffer: float = 0.98,
) -> int:
    """Highest valid rung of the sequential ladder for one session.

    Tranche 1: Close > EMA50
    Tranche 2: Close > EMA100 AND EMA50 > EMA100      (sequential: includes T1)
    Tranche 3: Close > EMA200 AND EMA100 > EMA200     (sequential: includes T2)

    Canary OFF or hard breakdown (Close < EMA200 * buffer) -> 0.
    """
    if not canary_safe or not np.isfinite(close):
        return 0
    if close < ema_long * hard_exit_buffer:
        return 0  # hard exit guard: instant 0%
    if close > ema_fast:  # T1 condition, required for every higher rung
        if close > ema_mid and ema_fast > ema_mid:  # T2 (sequential)
            if close > ema_long and ema_mid > ema_long:  # T3 (sequential)
                return 3
            return 2
        return 1
    return 0


def advance_tranche_ladder(desired: "list[int] | np.ndarray", initial: int = 0) -> np.ndarray:
    """Path-dependent rung state: at most +1 step per day, instant drops.

    ``desired`` is the per-session highest valid rung from
    ``resolve_desired_state``. Returns the executed state path (0..3).
    """
    states = np.empty(len(desired), dtype=np.int8)
    state = int(initial)
    for i, d in enumerate(desired):
        d = int(d)
        if d > state:  # accumulation: at most one step per session
            state = min(state + 1, d)
        else:  # de-risking: instant drop to highest valid state
            state = d
        states[i] = state
    return states


@dataclass
class SignalFrame:
    """Everything the signal layer produced (weights + diagnostics)."""

    weights: pd.DataFrame  # target weights as of Close(t), per asset
    tranche_states: pd.DataFrame  # rung path 0..3
    canary_safe: pd.Series  # TIP momentum > 0
    tip_momentum: pd.Series


class SignalEngine:
    """Compose the V8.1 macro canary with the sequential tranche ladder."""

    def __init__(self, market: MarketData, config: SignalConfig) -> None:
        self._market = market
        self._config = config
        missing = [
            t for t in (*config.universe, config.canary_ticker)
            if t not in market.closes.columns
        ]
        if missing:
            raise KeyError(f"missing tickers in market data: {missing}")

    # -- public API ---------------------------------------------------------

    def compute(self) -> SignalFrame:
        closes = self._market.closes
        assets = list(self._config.universe)
        asset_close = closes[assets].astype(float)

        spans = self._config.ema_spans
        ema_fast = asset_close.ewm(span=spans[0], adjust=False).mean()
        ema_mid = asset_close.ewm(span=spans[1], adjust=False).mean()
        ema_long = asset_close.ewm(span=spans[2], adjust=False).mean()

        score = tip_momentum_13612(closes[self._config.canary_ticker])
        canary_safe = score > 0.0  # NaN -> False (unsafe until proven)

        close = asset_close.to_numpy(dtype=float)
        e_fast = ema_fast.to_numpy(dtype=float)
        e_mid = ema_mid.to_numpy(dtype=float)
        e_long = ema_long.to_numpy(dtype=float)
        canary = canary_safe.reindex(closes.index).fillna(False).to_numpy(dtype=bool)[:, None]

        hard_breakdown = close < e_long * self._config.hard_exit_buffer

        # Vectorized sequential (nested) ladder conditions.
        t1 = close > e_fast
        t2 = t1 & (close > e_mid) & (e_fast > e_mid)
        t3 = t2 & (close > e_long) & (e_mid > e_long)
        desired = np.where(t3, 3, np.where(t2, 2, np.where(t1, 1, 0)))
        desired = np.where(canary & ~hard_breakdown, desired, 0).astype(np.int8)

        states = np.empty_like(desired)
        for col in range(len(assets)):
            states[:, col] = advance_tranche_ladder(desired[:, col])

        weights_df = pd.DataFrame(
            states.astype(float) / 3.0 * EQUAL_CAP,
            index=closes.index,
            columns=assets,
        )
        self._verify_invariants(weights_df, canary_safe)

        return SignalFrame(
            weights=weights_df,
            tranche_states=pd.DataFrame(states, index=closes.index, columns=assets),
            canary_safe=canary_safe.reindex(closes.index),
            tip_momentum=score,
        )

    # -- invariant audit ----------------------------------------------------

    def _verify_invariants(self, weights: pd.DataFrame, canary_safe: pd.Series) -> None:
        arr = weights.to_numpy(dtype=float)
        if (arr < -_EPS).any():
            raise AssertionError("negative weights produced")
        cap = self._config.max_weight_per_asset
        if (arr > cap + _EPS).any():
            raise AssertionError(f"per-asset weight exceeds equal-cap {cap:.1%}")
        if (arr.sum(axis=1) > 1.0 + _EPS).any():
            raise AssertionError("gross exposure exceeds 100%")
        # STRICT canary: momentum <= 0 (or unknown) -> no risky exposure at all.
        unsafe = ~canary_safe.to_numpy(dtype=bool)
        if unsafe.any() and not np.allclose(arr[unsafe], 0.0, atol=_EPS):
            raise AssertionError("canary invariant violated: exposure while TIP <= 0")
