# -*- coding: utf-8 -*-
"""Invariant & accounting self-checks for the AlphaShield core engines.

Run:  python research/alphashield_v82/test_invariants.py   (no pytest required)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])  # .../Bucky-Trading-Bot/research
)

from alphashield_v82.audit_suite import max_drawdown, nav_curve, spanning_regression  # noqa: E402
from alphashield_v82.config import DataConfig, PortfolioConfig, SignalConfig  # noqa: E402
from alphashield_v82.data_engine import DataEngine, MarketData  # noqa: E402
from alphashield_v82.portfolio_engine import PortfolioEngine  # noqa: E402
from alphashield_v82.signal_engine import (  # noqa: E402
    SignalEngine,
    advance_tranche_ladder,
    resolve_desired_state,
)

ASSETS = ("URTH", "SPY", "QQQ", "GLD", "SMH")
DAILY_YIELD = (1 + 0.02) ** (1 / 252) - 1


# ---------------------------------------------------------------------------
# Signal primitives
# ---------------------------------------------------------------------------

def test_desired_state_table() -> None:
    """resolve_desired_state truth table (sequential nested conditions)."""
    assert resolve_desired_state(False, 200.0, 100.0, 90.0, 80.0) == 0
    assert resolve_desired_state(True, 105.0, 100.0, 90.0, 80.0) == 3
    assert resolve_desired_state(True, 95.0, 100.0, 90.0, 80.0) == 0  # below e50
    assert resolve_desired_state(True, 92.0, 80.0, 90.0, 85.0) == 1  # above e50 only
    assert resolve_desired_state(True, 130.0, 80.0, 100.0, 120.0) == 1  # sick stack -> restart
    assert resolve_desired_state(True, 75.0, 100.0, 90.0, 80.0) == 0  # hard exit
    # exactly AT the hard-exit boundary is not below it -> ladder runs (T2 here)
    assert resolve_desired_state(True, 78.4, 70.0, 65.0, 80.0) == 2


def test_ladder_state_machine() -> None:
    """+1 step per day up, instant drop, reset & fresh accumulation."""
    assert advance_tranche_ladder([3, 3, 3]).tolist() == [1, 2, 3]
    assert advance_tranche_ladder([3, 3, 1, 0, 3]).tolist() == [1, 2, 1, 0, 1]
    assert advance_tranche_ladder([2]).tolist() == [1]
    assert advance_tranche_ladder([0]).tolist() == [0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _market_frame(n: int = 600, seed: int = 7) -> MarketData:
    """Synthetic aligned market: 5 risky assets + TIP with two regimes."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    trend = np.multiply.accumulate(1.0008 + rng.normal(0, 0.0007, n))
    flat = np.multiply.accumulate(1.0 + rng.normal(0, 0.0002, n))
    growth = np.multiply.accumulate(1.0009 + rng.normal(0, 0.0007, n))
    tip = np.multiply.accumulate(
        np.r_[np.full(n // 2, 1.0002), np.full(n - n // 2, 0.9997)] + rng.normal(0, 5e-5, n)
    )
    closes = pd.DataFrame(
        {
            "URTH": trend,
            "SPY": trend,
            "QQQ": growth,
            "GLD": flat,
            "SMH": trend,
            "TIP": tip,
        },
        index=idx,
    )
    # opens == closes keeps overnight/intraday decomposition exact for tests
    opens = closes[list(ASSETS)].copy()
    return MarketData(closes=closes, opens=opens)


def test_signal_engine_invariants() -> None:
    signals = SignalEngine(_market_frame(), SignalConfig()).compute()
    w = signals.weights
    assert (w.to_numpy() <= 0.20 + 1e-12).all()
    assert (w.to_numpy() >= 0).all()
    unsafe = ~signals.canary_safe
    if unsafe.any():
        assert (w[unsafe].to_numpy() == 0).all()
    states = signals.tranche_states.loc[signals.canary_safe]
    if len(states):
        assert (states.diff().dropna() <= 1.0).all().all()


# ---------------------------------------------------------------------------
# Portfolio ledger
# ---------------------------------------------------------------------------

def _portfolio_engine(market: MarketData) -> PortfolioEngine:
    return PortfolioEngine(
        market.opens, market.closes[list(ASSETS)], PortfolioConfig()
    )


def test_ledger_all_cash_only_yields() -> None:
    market = _market_frame()
    zero_w = pd.DataFrame(0.0, index=market.calendar, columns=list(ASSETS))
    ledger = _portfolio_engine(market).run(zero_w)
    nav_growth = ledger.returns.add(1).prod()
    expected = (1 + DAILY_YIELD) ** len(ledger.returns)
    assert abs(nav_growth / expected - 1.0) < 1e-10
    assert float(ledger.friction.abs().sum()) == 0.0
    assert float(ledger.turnover.abs().sum()) == 0.0
    np.testing.assert_allclose(ledger.returns.to_numpy(), DAILY_YIELD, atol=1e-12)


def test_ledger_constant_weight_reference() -> None:
    """Independent dollar-recursion reference for a constant 20%-SPY policy."""
    market = _market_frame(n=800, seed=11)
    idx = market.calendar
    w = pd.DataFrame(0.0, index=idx, columns=list(ASSETS))
    w["SPY"] = 0.20
    ledger = _portfolio_engine(market).run(w)
    cost = 5e-4
    r = market.closes["SPY"].pct_change()
    v, cash = 0.0, 1_000_000.0
    expected: list[float] = []
    for dt in idx[1:]:
        prev_nav = v + cash
        marked = v * (1.0 + r.loc[dt])  # overnight mark of the old SPY holding
        A = marked + cash  # open NAV (opens == closes -> intraday mult 1)
        post = A
        if w["SPY"].loc[dt] > 0.0:
            for _ in range(60):  # same fixed point, written independently
                traded = abs(0.2 * post - marked)
                post = A - cost * traded
        risky = 0.2 * post
        cash_new = post - risky
        keep = 1.0 - w["SPY"].loc[dt]
        net = (post + keep * post * DAILY_YIELD) / prev_nav - 1.0
        v, cash = risky, cash_new + keep * post * DAILY_YIELD
        expected.append(net)
    # Day 0 is all-cash (shift leaves no signal yet)
    all_returns = [DAILY_YIELD] + expected
    np.testing.assert_allclose(
        ledger.returns.to_numpy(), np.asarray(all_returns), atol=1e-9, rtol=1e-9
    )


def test_ledger_no_lookahead_prefix() -> None:
    market = _market_frame()
    idx = market.calendar
    rng = np.random.default_rng(3)
    w = pd.DataFrame(0.2, index=idx, columns=list(ASSETS))
    w.iloc[100:] = rng.random((len(idx) - 100, len(ASSETS))) * 0.2
    engine = _portfolio_engine(market)
    short = engine.run(w, start=idx[250])
    assert len(short.returns) == len(idx) - 250
    # extend the tape by one session; history must not change
    next_day = pd.bdate_range(idx[-1] + pd.Timedelta(days=1), periods=1)[0]
    closes = market.closes.copy()
    closes.loc[next_day] = closes.iloc[-1] * 0.99
    opens = market.opens.copy()
    opens.loc[next_day] = opens.iloc[-1] * 0.99
    extended = _portfolio_engine(MarketData(closes=closes, opens=opens))
    taller = extended.run(w, start=idx[250])
    np.testing.assert_array_equal(
        short.returns.to_numpy(), taller.returns.to_numpy()[: len(short.returns)]
    )
    assert short.returns.index.equals(taller.returns.index[: len(short.returns)])


def test_stress_lag_changes_weights() -> None:
    """T+2 must execute the signal from t-2, not t-1."""
    market = _market_frame()
    idx = market.calendar
    rng = np.random.default_rng(5)
    w = pd.DataFrame(
        rng.random((len(idx), len(ASSETS))) * 0.1, index=idx, columns=list(ASSETS)
    )
    engine = _portfolio_engine(market)
    base = engine.run(w, start=idx[300])
    lag2 = engine.run(w, start=idx[300], execution_lag=2)
    assert not np.allclose(base.returns.to_numpy(), lag2.returns.to_numpy())
    # executed weights at T+1 == signal at t-1 (strict mapping re-check)
    expected = w.shift(1).loc[base.weights.index].fillna(0.0)
    np.testing.assert_allclose(base.weights.to_numpy(), expected.to_numpy(), atol=1e-12)
    expected2 = w.shift(2).loc[lag2.weights.index].fillna(0.0)
    np.testing.assert_allclose(lag2.weights.to_numpy(), expected2.to_numpy(), atol=1e-12)


def test_cost_stress_reduces_nav() -> None:
    market = _market_frame()
    idx = market.calendar
    w = pd.DataFrame(0.2, index=idx, columns=list(ASSETS))
    engine = _portfolio_engine(market)
    base = engine.run(w, start=idx[300])
    doubled = engine.run(w, start=idx[300], cost_bps=10.0)
    assert doubled.nav.iloc[-1] < base.nav.iloc[-1]
    ratio = float(doubled.returns.mean() / base.returns.mean())
    assert ratio < 1.0


# ---------------------------------------------------------------------------
# Data engine
# ---------------------------------------------------------------------------

def test_data_engine_assembly() -> None:
    market = _market_frame(n=600)
    cfg = DataConfig(
        data_start="2020-01-01",
        backtest_start=str(market.calendar[300].date()),
        cache_dir="unused",
    )
    engine = DataEngine(cfg)
    snapshot = _snapshot(market)
    assembled = engine._assemble(snapshot)
    assert assembled.calendar.equals(market.closes.index)
    assert list(assembled.opens.columns) == list(ASSETS)
    assert "TIP" in assembled.closes.columns


def _snapshot(market: MarketData) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    for col in market.closes.columns:
        field = market.opens[col] if col in market.opens.columns else market.closes[col]
        frames[col] = pd.DataFrame({"Open": field, "Close": market.closes[col]})
    return pd.concat(frames, names=("Ticker", "Field")).sort_index()


# ---------------------------------------------------------------------------
# Audit metrics
# ---------------------------------------------------------------------------

def test_audit_metrics_and_hac() -> None:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2024-01-01", periods=500)
    bench = pd.Series(rng.normal(0.0008, 0.005, len(idx)), index=idx)
    strat = bench + 0.0002  # pure alpha over the benchmark
    rf = pd.Series(0.0, index=idx)

    from alphashield_v82.audit_suite import performance_metrics

    m = performance_metrics(strat, rf, bench)
    assert m["CAGR"] > 0
    assert 1.0 < m["Upside Capture"] < 1.35  # +2bp/day alpha lifts the up legs
    assert 0.8 < m["Downside Capture"] < 1.0  # ...and cushions the down legs

    reg = spanning_regression(strat, bench, rf)
    assert abs(reg["beta"] - 1.0) < 1e-6
    assert abs(reg["alpha_annual"] - 0.0002 * 252) < 1e-6
    assert reg["alpha_t_stat"] > 0

    same = spanning_regression(bench, bench, rf)
    assert abs(same["alpha_annual"]) < 1e-9
    assert abs(same["beta"] - 1.0) < 1e-9

    # drawdown machinery on a losing strategy
    down = pd.Series([-0.01, -0.02, 0.005, -0.001] * 20, index=pd.bdate_range("2020-01-01", periods=80))
    dd = max_drawdown(down)
    assert dd < -0.2  # long compounding decline
    assert nav_curve(down).iloc[-1] < 1.0


def main() -> None:
    tests = [
        test_desired_state_table,
        test_ladder_state_machine,
        test_signal_engine_invariants,
        test_ledger_all_cash_only_yields,
        test_ledger_constant_weight_reference,
        test_ledger_no_lookahead_prefix,
        test_stress_lag_changes_weights,
        test_cost_stress_reduces_nav,
        test_data_engine_assembly,
        test_audit_metrics_and_hac,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} invariants hold")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
