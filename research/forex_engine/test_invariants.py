# -*- coding: utf-8 -*-
r"""Offline deterministic unit tests for the G10 Forex Quantitative Engine (Engine 3).

Verifies all 4 required invariants using synthetic deterministic DataFrames:
  1. ``adx_wilder(14)`` is strictly bounded in ``[0, 100]`` and handles flat
     prices without ``NaN`` or ``ZeroDivisionError``.
  2. Symmetric Long/Short PnL accounting and gross exposure cap ``<= 1.0``.
  3. Choppy regime filter (``ADX_14 < 18`` and ``H_100 < 0.48``) flattens exposure
     to ``0.0``.
  4. Strict ``T+1`` execution lag with zero look-ahead bias.

Run:
  E:\AI\Quant-Bot\.venv\Scripts\python.exe -m research.forex_engine.main --self-test
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.forex_engine.config import (  # noqa: E402
    ForexConfig,
    G10_UNIVERSE,
    MAX_GROSS_LEVERAGE,
    MAX_PAIR_WEIGHT,
)
from research.forex_engine.portfolio_engine import (  # noqa: E402
    ForexPortfolioEngine,
    compute_target_weights,
)
from research.forex_engine.regime_filter import (  # noqa: E402
    ForexRegimeFilter,
    ForexRegimeState,
    adx_wilder,
    directional_indicators_wilder,
    rolling_hurst,
)
from research.forex_engine.signal_engine import ForexSignalEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Invariant 1: adx_wilder(14) bounded in [0, 100] and safe on flat prices
# ---------------------------------------------------------------------------

def test_invariant_1_adx_wilder_bounds_and_flat_prices() -> None:
    """Verify adx_wilder(14) stays in [0, 100] and handles flat prices without NaN/ZeroDivision."""
    idx = pd.bdate_range("2022-01-03", periods=250)

    # Case A: Completely flat prices (High == Low == Close == 1.1000)
    flat_s = pd.Series(1.1000, index=idx)
    adx_flat, pdi_flat, mdi_flat = directional_indicators_wilder(flat_s, flat_s, flat_s, period=14)
    assert not adx_flat.isna().any(), "NaN produced on flat price series"
    assert not pdi_flat.isna().any()
    assert not mdi_flat.isna().any()
    np.testing.assert_allclose(adx_flat.to_numpy(), 0.0, atol=1e-12)

    # Also verify Hurst on flat prices returns 0.50 without NaN
    h_flat = rolling_hurst(flat_s, window=100)
    assert not h_flat.isna().any()
    np.testing.assert_allclose(h_flat.to_numpy(), 0.50, atol=1e-12)

    # Case B: Extreme volatile & trending DataFrames across all 7 G10 pairs
    rng = np.random.default_rng(123)
    closes_dict: dict[str, np.ndarray] = {}
    for k, pair in enumerate(G10_UNIVERSE):
        drift = 0.002 if k % 2 == 0 else -0.002
        closes_dict[pair] = 1.0 * np.cumprod(1.0 + drift + rng.normal(0.0, 0.008, len(idx)))
    closes = pd.DataFrame(closes_dict, index=idx)
    highs = closes * 1.01
    lows = closes * 0.99

    adx_df = adx_wilder(highs, lows, closes, period=14)
    assert not adx_df.isna().any().any()
    assert bool((adx_df.to_numpy() >= 0.0).all() and (adx_df.to_numpy() <= 100.0).all())
    # Strong monotonic drift must push ADX well above 22
    assert float(adx_df.iloc[-1].mean()) > 22.0


# ---------------------------------------------------------------------------
# Invariant 2: Symmetric Long/Short PnL accounting & gross exposure <= 1.0
# ---------------------------------------------------------------------------

def test_invariant_2_symmetric_long_short_pnl_and_gross_cap() -> None:
    """Verify exact symmetry between Long (+w) on +r and Short (-w) on -r, and gross cap <= 1.0."""
    idx = pd.bdate_range("2023-01-02", periods=60)
    # Pair A rises +1% per day; Pair B falls -1% per day (exact mirror return -1%)
    r_step = 0.01
    close_up = 1.0 * ((1.0 + r_step) ** np.arange(len(idx)))
    close_dn = 1.0 * ((1.0 - r_step) ** np.arange(len(idx)))

    pairs = list(G10_UNIVERSE)
    closes_up = pd.DataFrame({p: close_up for p in pairs}, index=idx)
    closes_dn = pd.DataFrame({p: close_dn for p in pairs}, index=idx)

    cfg_zero_cost = ForexConfig(cost_bps=0.0, cash_yield_annual=0.0)

    # Portfolio 1: Long +0.15 on EURUSD=X in rising market
    w_long = pd.DataFrame(0.0, index=idx, columns=pairs)
    w_long["EURUSD=X"] = +0.15
    led_long = ForexPortfolioEngine(closes_up, cfg_zero_cost).run(w_long)

    # Portfolio 2: Short -0.15 on EURUSD=X in falling market (-1% daily return)
    w_short = pd.DataFrame(0.0, index=idx, columns=pairs)
    w_short["EURUSD=X"] = -0.15
    led_short = ForexPortfolioEngine(closes_dn, cfg_zero_cost).run(w_short)

    # Both must earn +0.15 * 0.01 = +0.0015 per session from session 1 onwards!
    np.testing.assert_allclose(
        led_long.returns.iloc[1:].to_numpy(),
        0.15 * r_step,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        led_long.returns.to_numpy(),
        led_short.returns.to_numpy(),
        atol=1e-12,
        err_msg="Symmetric Long/Short PnL violated!",
    )

    # Verify gross exposure normalization when all 7 pairs demand max weight (7 * 0.20 = 1.40 > 1.0)
    dir_all = pd.DataFrame(
        [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]] * len(idx),
        index=idx,
        columns=pairs,
    )
    low_vol = pd.DataFrame(0.05, index=idx, columns=pairs)  # 0.025 / 0.05 = 0.50 -> clipped to 0.20 each
    sized_w = compute_target_weights(dir_all, low_vol)
    assert float(sized_w.abs().max().max()) <= MAX_PAIR_WEIGHT + 1e-12
    np.testing.assert_allclose(sized_w.abs().sum(axis=1).to_numpy(), MAX_GROSS_LEVERAGE, atol=1e-12)

    # Also verify ForexPortfolioEngine caps any oversized mixed Long/Short input to gross <= 1.0
    oversized = pd.DataFrame(
        [[0.20, -0.20, 0.20, -0.20, 0.20, -0.20, 0.20]] * len(idx),
        index=idx,
        columns=pairs,
    )
    led_gross = ForexPortfolioEngine(closes_up).run(oversized)
    assert float(led_gross.gross_exposure.max()) <= MAX_GROSS_LEVERAGE + 1e-9
    assert float(led_gross.weights.abs().to_numpy().max()) <= MAX_PAIR_WEIGHT + 1e-9


# ---------------------------------------------------------------------------
# Invariant 3: Choppy regime filter flattens exposure to zero
# ---------------------------------------------------------------------------

def test_invariant_3_choppy_regime_flattens_exposure() -> None:
    """Verify that when ADX_14 < 18 and H_100 < 0.48, active positions flatten to 0.0."""
    n = 220
    idx = pd.bdate_range("2022-01-03", periods=n)
    pairs = list(G10_UNIVERSE)

    # Strong uptrend so breakout + momentum signal wants to be +1 Long
    trend_curve = 1.0 * np.cumprod(np.full(n, 1.0015))
    closes = pd.DataFrame({p: trend_curve for p in pairs}, index=idx)
    highs = closes * 1.003
    lows = closes * 0.997

    base_regime = ForexRegimeFilter().evaluate(highs, lows, closes)
    sig_engine = ForexSignalEngine()
    out_trending = sig_engine.compute(highs, lows, closes, base_regime)
    # Confirm active long signal on the final window in trending regime
    assert float(out_trending.direction_signal.iloc[-1].abs().sum()) > 0.0

    # Now force Choppy Regime (ADX_14 = 12.0 < 18.0 and H_100 = 0.40 < 0.48) on the last 20 days
    adx_chop = base_regime.adx14.copy()
    hurst_chop = base_regime.hurst100.copy()
    adx_chop.iloc[-20:, :] = 12.0
    hurst_chop.iloc[-20:, :] = 0.40
    is_choppy = (adx_chop < 18.0) & (hurst_chop < 0.48)
    is_trending = ((adx_chop > 22.0) | (hurst_chop > 0.53)) & (~is_choppy)

    choppy_state = ForexRegimeState(
        adx14=adx_chop,
        plus_di14=base_regime.plus_di14,
        minus_di14=base_regime.minus_di14,
        hurst100=hurst_chop,
        is_trending=is_trending,
        is_choppy=is_choppy,
    )
    out_choppy = sig_engine.compute(highs, lows, closes, choppy_state)
    weights_choppy = compute_target_weights(out_choppy.direction_signal, out_choppy.vol20_annual)

    # Every single pair during the final 20 choppy sessions must be strictly 0.0
    np.testing.assert_allclose(
        out_choppy.direction_signal.iloc[-20:].to_numpy(),
        0.0,
        atol=1e-12,
        err_msg="Choppy regime failed to flatten direction_signal to 0.0!",
    )
    np.testing.assert_allclose(
        weights_choppy.iloc[-20:].to_numpy(),
        0.0,
        atol=1e-12,
        err_msg="Choppy regime failed to flatten portfolio weights to 0.0!",
    )


# ---------------------------------------------------------------------------
# Invariant 4: Strict T+1 execution lag & zero look-ahead bias
# ---------------------------------------------------------------------------

def test_invariant_4_strict_t_plus_1_lag() -> None:
    """Verify lag=0 is rejected, executed weights == signal(t-1), and future mutations do not leak."""
    idx = pd.bdate_range("2023-01-02", periods=120)
    pairs = list(G10_UNIVERSE)
    rng = np.random.default_rng(99)
    closes = pd.DataFrame(
        1.0 + np.cumsum(rng.normal(0.0, 0.004, (len(idx), len(pairs))), axis=0),
        index=idx,
        columns=pairs,
    ).clip(lower=0.2)

    try:
        ForexConfig(execution_lag=0)
        raise AssertionError("ForexConfig(execution_lag=0) must raise ValueError")
    except ValueError:
        pass

    engine = ForexPortfolioEngine(closes)
    try:
        engine.run(pd.DataFrame(0.0, index=idx, columns=pairs), execution_lag=0)
        raise AssertionError("engine.run(execution_lag=0) must raise ValueError")
    except ValueError:
        pass

    sig = pd.DataFrame(0.0, index=idx, columns=pairs)
    sig.iloc[30:60, 0] = +0.18   # Long EURUSD
    sig.iloc[60:90, 2] = -0.15   # Short USDJPY

    ledger = engine.run(sig)
    # Signal at t=30 deploys at t=31 (still 0.0 at t=30)
    assert float(ledger.weights.iloc[30, 0]) == 0.0
    assert abs(float(ledger.weights.iloc[31, 0]) - 0.18) < 1e-12
    # Signal at t=60 deploys at t=61 (still 0.0 at t=60 for USDJPY)
    assert float(ledger.weights.iloc[60, 2]) == 0.0
    assert abs(float(ledger.weights.iloc[61, 2]) - (-0.15)) < 1e-12

    # Prefix invariance check: mutating signals after t=80 never changes returns on [0..80]
    sig_shock = sig.copy()
    sig_shock.iloc[81:, :] = -0.10
    ledger_shock = engine.run(sig_shock)
    np.testing.assert_allclose(
        ledger.returns.iloc[:81].to_numpy(),
        ledger_shock.returns.iloc[:81].to_numpy(),
        atol=1e-12,
    )


def run_all_tests() -> bool:
    tests = [
        test_invariant_1_adx_wilder_bounds_and_flat_prices,
        test_invariant_2_symmetric_long_short_pnl_and_gross_cap,
        test_invariant_3_choppy_regime_flattens_exposure,
        test_invariant_4_strict_t_plus_1_lag,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} G10 Forex engine invariants passed.")
    return failed == 0


if __name__ == "__main__":
    if not run_all_tests():
        sys.exit(1)
