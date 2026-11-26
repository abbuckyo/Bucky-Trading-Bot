import numpy as np
import pandas as pd

import bot


def _make_close_series(daily_vol: float, periods: int = 120, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=0.0003, scale=daily_vol, size=periods)
    prices = 100.0 * np.cumprod(1.0 + rets)
    idx = pd.date_range("2026-07-01", periods=periods, freq="B")
    return pd.Series(prices, index=idx)


def test_lower_vol_gets_higher_cap():
    closes = {
        "SCBWORLDE": _make_close_series(0.010, seed=1),
        "SCBS&P500E": _make_close_series(0.011, seed=2),
        "SCBNDQ(E)": _make_close_series(0.015, seed=3),
        "SCBGOLDE": _make_close_series(0.006, seed=4),
        "SCBSEMI(E)": _make_close_series(0.025, seed=5),
    }
    caps = bot.compute_inv_vol_caps(closes)
    assert caps["SCBGOLDE"] > caps["SCBSEMI(E)"]
    assert abs(sum(caps.values()) - 1.0) < 1e-9


def test_caps_bounds_and_sum_under_extreme_spreads():
    closes = {
        "SCBWORLDE": _make_close_series(0.0005, seed=1),
        "SCBS&P500E": _make_close_series(0.0005, seed=2),
        "SCBNDQ(E)": _make_close_series(0.050, seed=3),
        "SCBGOLDE": _make_close_series(0.060, seed=4),
        "SCBSEMI(E)": _make_close_series(0.080, seed=5),
    }
    caps = bot.compute_inv_vol_caps(closes)
    assert abs(sum(caps.values()) - 1.0) < 1e-9
    for w in caps.values():
        assert 0.08 - 1e-9 <= w <= 0.35 + 1e-9


def test_fallback_to_equal_when_missing_or_disabled(monkeypatch):
    closes = {
        "SCBWORLDE": _make_close_series(0.010, periods=10, seed=1),
        "SCBS&P500E": _make_close_series(0.010, periods=100, seed=2),
        "SCBNDQ(E)": _make_close_series(0.015, periods=100, seed=3),
        "SCBGOLDE": _make_close_series(0.006, periods=100, seed=4),
        "SCBSEMI(E)": _make_close_series(0.025, periods=100, seed=5),
    }
    caps_short = bot.compute_inv_vol_caps(closes)
    assert all(abs(w - 0.20) < 1e-9 for w in caps_short.values())

    closes["SCBWORLDE"] = _make_close_series(0.010, periods=100, seed=1)
    monkeypatch.setattr(bot.CFG, "ENABLE_INV_VOL_ALLOCATOR", False)
    caps_disabled = bot.compute_inv_vol_caps(closes)
    assert all(abs(w - 0.20) < 1e-9 for w in caps_disabled.values())


def test_total_portfolio_weight_bounded_across_tranches():
    closes = {
        "SCBWORLDE": _make_close_series(0.010, seed=1),
        "SCBS&P500E": _make_close_series(0.012, seed=2),
        "SCBNDQ(E)": _make_close_series(0.016, seed=3),
        "SCBGOLDE": _make_close_series(0.007, seed=4),
        "SCBSEMI(E)": _make_close_series(0.028, seed=5),
    }
    caps = bot.compute_inv_vol_caps(closes)
    for stage in range(4):
        total = sum((stage / 3.0) * w for w in caps.values())
        assert total <= 1.0 + 1e-9
