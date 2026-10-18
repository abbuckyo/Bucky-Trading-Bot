# -*- coding: utf-8 -*-
"""Offline deterministic unit tests for the Cross-Sectional Single-Stock Momentum Engine.

Verifies the 5 core invariants using synthetic deterministic DataFrames (zero network calls):
  1. Cross-sectional ranking logic strictly skips the last 21-day window (r_{12M-1M}).
  2. Earnings Blackout suppresses new trade entries within the 5-day pre-earnings window
     and tightens the Chandelier trailing stop to 1.5x ATR for held positions.
  3. Chandelier Trailing Stop triggers immediately when Close < High_22 - 3.0 * ATR_20.
  4. Max allocation per single stock never breaches 15.0% under extreme sizing inputs.
  5. Strict T+1 signal-to-execution lag is maintained in the ledger with zero look-ahead.

Run:
  python -m research.single_stock_engine.test_invariants
  or pytest research/single_stock_engine/test_invariants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running both as module and direct script
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.single_stock_engine.audit_suite import (  # noqa: E402
    performance_metrics,
    rf_series,
    spanning_regression,
    verify_invariants_report,
)
from research.single_stock_engine.config import (  # noqa: E402
    EngineConfig,
    MAX_SINGLE_STOCK_WEIGHT,
    TARGET_UNIVERSE,
)
from research.single_stock_engine.portfolio_engine import (  # noqa: E402
    PortfolioEngine,
    build_signal_weights,
    compute_position_weight,
    compute_target_notional,
)
from research.single_stock_engine.ranking_engine import (  # noqa: E402
    RankingEngine,
    compute_12m_1m_return,
    compute_momentum_score,
)
from research.single_stock_engine.regime_guard import (  # noqa: E402
    RegimeGuard,
    compute_earnings_blackout_matrix,
)
from research.single_stock_engine.risk_guards import (  # noqa: E402
    RiskGuards,
    compute_atr,
    compute_chandelier_stops,
)

TEST_UNIVERSE: tuple[str, ...] = TARGET_UNIVERSE[:10]


def _make_synthetic_ohlc(
    n_days: int = 320,
    tickers: tuple[str, ...] = TEST_UNIVERSE,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Create deterministic synthetic OHLC DataFrames + bullish SPY series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    closes_dict: dict[str, np.ndarray] = {}
    for idx, sym in enumerate(tickers):
        # Distinct positive drift per stock so cross-sectional ranking is deterministic
        drift = 0.0004 + 0.00015 * (len(tickers) - idx)
        noise = rng.normal(0.0, 0.006, size=n_days)
        closes_dict[sym] = 100.0 * np.cumprod(1.0 + drift + noise)

    closes = pd.DataFrame(closes_dict, index=dates)
    opens = closes * 0.999
    highs = closes * 1.008
    lows = closes * 0.992
    spy_close = pd.Series(
        400.0 * np.cumprod(np.full(n_days, 1.0008)),
        index=dates,
        name="SPY",
    )
    return opens, highs, lows, closes, spy_close


# ---------------------------------------------------------------------------
# Invariant 1: 12M-1M Momentum strictly skips the most recent 21 trading days
# ---------------------------------------------------------------------------

def test_invariant_1_ranking_skips_21_day_window() -> None:
    """Verify r_{12M-1M} = Close(t-21)/Close(t-252) - 1 ignores [t-20 .. t]."""
    _, _, _, closes, _ = _make_synthetic_ohlc(n_days=300)
    ret_base = compute_12m_1m_return(closes, lookback_long=252, lookback_skip=21)

    # Corrupt the last 21 trading observations (indices -20 .. -1) for AAPL
    closes_shocked = closes.copy()
    closes_shocked.iloc[-20:, closes_shocked.columns.get_loc("AAPL")] *= 5.0
    ret_shocked = compute_12m_1m_return(closes_shocked, lookback_long=252, lookback_skip=21)

    # On the final date t = -1, Close(t-21) is at index -22, which is untouched!
    expected_final = closes.iloc[-22] / closes.iloc[-253] - 1.0
    np.testing.assert_allclose(
        ret_base.iloc[-1].to_numpy(),
        expected_final.to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ret_base.iloc[-1].to_numpy(),
        ret_shocked.iloc[-1].to_numpy(),
        atol=1e-12,
        err_msg="r_{12M-1M} leaked information from the skipped 21-day window!",
    )

    # Also verify volatility floor 0.05 in Momentum Score
    const_dates = pd.bdate_range("2022-01-03", periods=300)
    low_vol_closes = pd.DataFrame(
        {"A": np.linspace(100.0, 150.0, 300), "B": np.linspace(100.0, 120.0, 300)},
        index=const_dates,
    )
    r12_1, vol60, score = compute_momentum_score(low_vol_closes, vol_floor=0.05)
    assert (vol60.dropna() < 0.05).all().all()
    np.testing.assert_allclose(
        score.iloc[-1].to_numpy(),
        (r12_1.iloc[-1] / 0.05).to_numpy(),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Invariant 2: Earnings Blackout suppresses entries within 5 trading days
#              and tightens Chandelier stop to 1.5x ATR for held stocks
# ---------------------------------------------------------------------------

def test_invariant_2_earnings_blackout_guard() -> None:
    """Verify 5-day pre-earnings entry freeze and 1.5x ATR stop tightening."""
    opens, highs, lows, closes, spy_close = _make_synthetic_ohlc(n_days=310)
    cfg = EngineConfig(universe=TEST_UNIVERSE, max_active_holdings=2, top_quantile=0.20)
    ranking = RankingEngine(
        lookback_long=cfg.mom_lookback_long,
        lookback_skip=cfg.mom_lookback_skip,
        vol_lookback=cfg.vol_lookback,
        vol_floor=cfg.vol_floor,
        top_quantile=cfg.top_quantile,
        max_active_holdings=cfg.max_active_holdings,
    ).compute(closes)

    # Find the very first weekly rebalance day with valid 252d momentum (index >= 252)
    rebal_dates = closes.index[ranking.rebalance_mask & (np.arange(len(closes)) >= 252)]
    first_rebal = rebal_dates[0]
    first_rebal_pos = int(closes.index.get_loc(first_rebal))

    # Top ranked stock on first_rebal (not yet held prior to first_rebal)
    top_stock = str(ranking.ranks.loc[first_rebal].idxmin())

    # Place an earnings announcement for `top_stock` 4 trading days after `first_rebal` (within 5-day window)
    earnings_dt = closes.index[first_rebal_pos + 4]
    blackout_mat = compute_earnings_blackout_matrix(
        closes.index,
        TEST_UNIVERSE,
        earnings_dates={top_stock: [earnings_dt]},
        blackout_trading_days=5,
    )
    # Must be True at [first_rebal_pos - 1 .. first_rebal_pos + 4] (5 trading days prior + earnings day)
    assert bool(blackout_mat.loc[first_rebal, top_stock]) is True
    assert bool(blackout_mat.iloc[first_rebal_pos - 2][top_stock]) is False

    # Control run (no earnings blackout) -> enters `top_stock` on `first_rebal`
    regime_no_ed = RegimeGuard().evaluate(spy_close, TEST_UNIVERSE, earnings_dates={})
    guards_no_ed = RiskGuards().evaluate(highs, lows, closes, earnings_blackout=regime_no_ed.earnings_blackout)
    weights_no_ed = build_signal_weights(closes, ranking, regime_no_ed, guards_no_ed, cfg)
    assert float(weights_no_ed.loc[first_rebal, top_stock]) > 0.0

    # Blackout run -> blocks new entry in `top_stock` on `first_rebal`
    regime_ed = RegimeGuard().evaluate(spy_close, TEST_UNIVERSE, earnings_dates={top_stock: [earnings_dt]})
    guards_ed = RiskGuards().evaluate(highs, lows, closes, earnings_blackout=regime_ed.earnings_blackout)
    weights_ed = build_signal_weights(closes, ranking, regime_ed, guards_ed, cfg)
    assert float(weights_ed.loc[first_rebal, top_stock]) == 0.0

    # Verify stop tightening: during blackout window, active_stop_price == High_22 - 1.5 * ATR_20
    expected_tight = (
        guards_ed.high22.loc[first_rebal, top_stock]
        - 1.5 * guards_ed.atr20.loc[first_rebal, top_stock]
    )
    np.testing.assert_allclose(
        guards_ed.active_stop_price.loc[first_rebal, top_stock],
        expected_tight,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Invariant 3: Chandelier Trailing Stop triggers instantly on breach
# ---------------------------------------------------------------------------

def test_invariant_3_chandelier_trailing_stop_exit() -> None:
    """Verify Close < High_22 - 3.0 * ATR_20 triggers immediate signal exit and T+1 execution exit."""
    opens, highs, lows, closes, spy_close = _make_synthetic_ohlc(n_days=310)
    cfg = EngineConfig(universe=TEST_UNIVERSE, max_active_holdings=2, top_quantile=0.20)
    ranking = RankingEngine(
        top_quantile=cfg.top_quantile,
        max_active_holdings=cfg.max_active_holdings,
    ).compute(closes)

    rebal_dates = closes.index[ranking.rebalance_mask & (np.arange(len(closes)) >= 260)]
    entry_rebal = rebal_dates[0]
    entry_pos = int(closes.index.get_loc(entry_rebal))
    top_stock = str(ranking.ranks.loc[entry_rebal].idxmin())

    # Force a Chandelier stop breach 2 trading days after entry_rebal (a non-Monday session)
    breach_pos = entry_pos + 2
    breach_date = closes.index[breach_pos]
    col_idx = closes.columns.get_loc(top_stock)

    # Compute baseline ATR20 and High22 at breach_date, then set Close just below Stop Price
    atr_pre = compute_atr(highs, lows, closes, window=20)
    high22_pre = highs.rolling(22, min_periods=22).max()
    stop_level = float(high22_pre.iloc[breach_pos, col_idx] - 3.0 * atr_pre.iloc[breach_pos, col_idx])

    closes_mod = closes.copy()
    lows_mod = lows.copy()
    # Set close slightly below the stop while keeping High and Low consistent so stop is breached
    closes_mod.iloc[breach_pos, col_idx] = stop_level * 0.97
    lows_mod.iloc[breach_pos, col_idx] = min(float(lows_mod.iloc[breach_pos, col_idx]), stop_level * 0.96)

    _, _, stop_normal, _, active_stop = compute_chandelier_stops(highs, lows_mod, closes_mod)
    assert float(closes_mod.iloc[breach_pos, col_idx]) < float(stop_normal.iloc[breach_pos, col_idx])

    regime = RegimeGuard().evaluate(spy_close, TEST_UNIVERSE)
    guards = RiskGuards().evaluate(highs, lows_mod, closes_mod, earnings_blackout=regime.earnings_blackout)
    assert bool(guards.stop_breached.loc[breach_date, top_stock]) is True

    sig_weights = build_signal_weights(closes_mod, ranking, regime, guards, cfg)
    # Held on entry_rebal and entry_pos + 1, then immediately 0.0 on breach_date
    assert float(sig_weights.iloc[entry_pos, col_idx]) > 0.0
    assert float(sig_weights.iloc[entry_pos + 1, col_idx]) > 0.0
    assert float(sig_weights.iloc[breach_pos, col_idx]) == 0.0

    # Executed ledger exits completely at T+1 (breach_pos + 1)
    ledger = PortfolioEngine(opens, closes_mod, cfg).run(sig_weights)
    assert float(ledger.weights.iloc[breach_pos, col_idx]) > 0.0      # still holding at Open(breach_date)
    assert float(ledger.weights.iloc[breach_pos + 1, col_idx]) == 0.0  # exited at Open(breach_date + 1)


# ---------------------------------------------------------------------------
# Invariant 4: Max allocation per single stock never breaches 15%
# ---------------------------------------------------------------------------

def test_invariant_4_max_allocation_cap_15_percent() -> None:
    """Verify single-stock weight never exceeds 15.0% even under extreme volatility/ATR inputs."""
    # Extreme scalar & array inputs: ultra-tiny ATR, zero ATR, huge price
    w_tiny_atr = compute_position_weight(close=500.0, atr20=1e-9)
    w_zero_atr = compute_position_weight(close=500.0, atr20=0.0)
    assert abs(float(w_tiny_atr) - MAX_SINGLE_STOCK_WEIGHT) < 1e-12
    assert abs(float(w_zero_atr) - MAX_SINGLE_STOCK_WEIGHT) < 1e-12

    # Exact formula check for moderate ATR where uncapped weight < 0.15:
    # Close = 100, ATR_20 = 5.0 -> Stop Distance = 15.0 -> Stop % = 0.15
    # Target Weight = 0.0075 / 0.15 = 0.05 (5.0%)
    w_exact = compute_position_weight(close=100.0, atr20=5.0)
    assert abs(float(w_exact) - 0.05) < 1e-12
    notional_exact = compute_target_notional(portfolio_nav=2_000_000.0, close=100.0, atr20=5.0)
    assert abs(notional_exact - 100_000.0) < 1e-9

    # Full pipeline check with ultra-low ATR across all 10 stocks
    opens, _, _, closes, spy_close = _make_synthetic_ohlc(n_days=300)
    highs_tight = closes * 1.0001
    lows_tight = closes * 0.9999
    cfg = EngineConfig(universe=TEST_UNIVERSE, max_active_holdings=10, top_quantile=1.0)
    ranking = RankingEngine(top_quantile=1.0, max_active_holdings=10).compute(closes)
    regime = RegimeGuard().evaluate(spy_close, TEST_UNIVERSE)
    guards = RiskGuards().evaluate(highs_tight, lows_tight, closes)
    sig_w = build_signal_weights(closes, ranking, regime, guards, cfg)

    assert float(sig_w.to_numpy().max()) <= MAX_SINGLE_STOCK_WEIGHT + 1e-12
    assert float(sig_w.sum(axis=1).max()) <= 1.0 + 1e-12

    # Even if oversized weights (e.g. 0.40) are passed into PortfolioEngine, they are capped at 0.15
    oversized = pd.DataFrame(0.40, index=closes.index, columns=closes.columns)
    ledger = PortfolioEngine(opens, closes, cfg).run(oversized)
    assert float(ledger.weights.to_numpy().max()) <= MAX_SINGLE_STOCK_WEIGHT + 1e-12
    assert float(ledger.risky_exposure.max()) <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Invariant 5: Strict T+1 signal-to-execution lag & zero look-ahead bias
# ---------------------------------------------------------------------------

def test_invariant_5_strict_t_plus_1_lag_and_ledger_integrity() -> None:
    """Verify strict T+1 lag, rejection of lag=0, prefix invariance, and friction/cash accounting."""
    opens, _, _, closes, _ = _make_synthetic_ohlc(n_days=280)
    cfg = EngineConfig(universe=TEST_UNIVERSE)

    # Lag 0 must be rejected
    try:
        EngineConfig(execution_lag=0)
        raise AssertionError("EngineConfig(execution_lag=0) should have raised ValueError")
    except ValueError:
        pass

    engine = PortfolioEngine(opens, closes, cfg)
    try:
        engine.run(pd.DataFrame(0.0, index=closes.index, columns=closes.columns), execution_lag=0)
        raise AssertionError("engine.run(execution_lag=0) should have raised ValueError")
    except ValueError:
        pass

    # Deterministic step-change signal weights on session t = 100
    sig = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    sig.iloc[100:, 0] = 0.12
    sig.iloc[150:, 1] = 0.15

    ledger = engine.run(sig)
    # On session 100 (signal day), executed weight must still be 0.0; on session 101 (T+1), it becomes 0.12
    assert float(ledger.weights.iloc[100, 0]) == 0.0
    assert abs(float(ledger.weights.iloc[101, 0]) - 0.12) < 1e-12
    assert float(ledger.weights.iloc[150, 1]) == 0.0
    assert abs(float(ledger.weights.iloc[151, 1]) - 0.15) < 1e-12

    # Prefix invariance (zero look-ahead): mutating future weights/prices after t=200 never alters t <= 200
    sig_future_shock = sig.copy()
    sig_future_shock.iloc[201:, :] = 0.10
    ledger_shock = engine.run(sig_future_shock)
    np.testing.assert_allclose(
        ledger.returns.iloc[:201].to_numpy(),
        ledger_shock.returns.iloc[:201].to_numpy(),
        atol=1e-12,
    )

    # Audit report verification
    report = verify_invariants_report(ledger, lag=1)
    assert all(v == "PASS" for v in report.values())

    # Spanning regression vs SPY and synthetic V8.1
    rf = rf_series(ledger.returns.index, cfg.cash_yield_annual)
    spy_ret = closes.iloc[:, 0].pct_change().fillna(0.0)
    v81_ret = closes.iloc[:, 1].pct_change().fillna(0.0)
    metrics = performance_metrics(ledger.returns, rf, benchmark_returns=spy_ret, v81_returns=v81_ret)
    assert "alpha_annual" in metrics and "v81_alpha_annual" in metrics
    span_multi = spanning_regression(
        ledger.returns,
        pd.DataFrame({"SPY": spy_ret, "V8_1": v81_ret}),
        rf,
    )
    assert "beta_SPY" in span_multi and "beta_V8_1" in span_multi


def run_all_tests() -> bool:
    tests = [
        test_invariant_1_ranking_skips_21_day_window,
        test_invariant_2_earnings_blackout_guard,
        test_invariant_3_chandelier_trailing_stop_exit,
        test_invariant_4_max_allocation_cap_15_percent,
        test_invariant_5_strict_t_plus_1_lag_and_ledger_integrity,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} single-stock engine invariants passed.")
    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    if not ok:
        sys.exit(1)
