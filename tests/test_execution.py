# -*- coding: utf-8 -*-
"""Comprehensive tests for modular execution scaffold."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import pytest

from execution.base_broker import (
    AccountBalance,
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from execution.paper_broker import PaperBroker
from execution.risk_guard import RiskCheckResult, RiskGuard
from execution.run_execution import (
    construct_rebalance_orders,
    ingest_blend_signals,
    ingest_m7_signals,
    ingest_scaling_factor,
    run_execution_cycle,
)


def test_base_broker_abc() -> None:
    """BaseBroker is an ABC and cannot be directly instantiated."""
    with pytest.raises(TypeError):
        BaseBroker()  # type: ignore


def test_paper_broker_initial_state() -> None:
    """PaperBroker initializes with given cash, zero positions and balanced equity."""
    broker = PaperBroker(initial_cash=50_000.0)
    bal = broker.get_account_balance()
    assert bal.cash == 50_000.0
    assert bal.equity == 50_000.0
    assert bal.initial_balance == 50_000.0
    assert bal.realized_pnl == 0.0
    assert bal.unrealized_pnl == 0.0
    assert len(broker.get_positions()) == 0


def test_paper_broker_t_plus_1_matching_and_slippage(tmp_path: Path) -> None:
    """Orders submitted are pending until process_next_bar with exact 5 bps slippage."""
    json_path = tmp_path / "trade_log.json"
    broker = PaperBroker(initial_cash=100_000.0, slippage_bps=5.0, json_log_path=json_path)

    # Day t: submit BUY order for 10 shares of NVDA
    order = Order(
        order_id="ord-001",
        symbol="NVDA",
        side=OrderSide.BUY,
        quantity=10.0,
        created_at="2026-03-01",
    )
    submitted = broker.submit_order(order)
    assert submitted.status == OrderStatus.PENDING
    assert len(broker.get_orders()) == 1

    # Before t+1 bar, no positions held
    assert "NVDA" not in broker.get_positions()

    # Day t+1: next bar arrives with NVDA price = 100.0
    # 5 bps slippage on BUY: fill_price = 100.0 * (1 + 0.0005) = 100.05
    prices_t1 = {"NVDA": 100.0}
    filled = broker.process_next_bar(prices_t1, bar_date="2026-03-02")

    assert len(filled) == 1
    ord_filled = filled[0]
    assert ord_filled.status == OrderStatus.FILLED
    assert ord_filled.filled_price == 100.05
    assert ord_filled.filled_quantity == 10.0
    assert ord_filled.filled_at == "2026-03-02"

    # Verify cash deducted: 100_000 - (10 * 100.05) = 98_999.50
    bal = broker.get_account_balance()
    assert pytest.approx(bal.cash, rel=1e-5) == 98_999.50
    # Position market value at 100.0 = 1,000.0
    nvda_pos = broker.get_position("NVDA")
    assert nvda_pos.quantity == 10.0
    assert nvda_pos.avg_entry_price == 100.05
    assert nvda_pos.market_value == 1000.0
    # Unrealized PnL: (100.0 - 100.05) * 10 = -0.50
    assert pytest.approx(bal.unrealized_pnl, rel=1e-4) == -0.50


def test_paper_broker_sell_execution_and_realized_pnl() -> None:
    """Selling positions on next bar applies -5 bps slippage and records realized PnL."""
    broker = PaperBroker(initial_cash=100_000.0, slippage_bps=5.0)

    # Buy 10 AAPL at 200.0 -> fill price 200.10
    broker.submit_order(Order(order_id="b1", symbol="AAPL", side=OrderSide.BUY, quantity=10.0))
    broker.process_next_bar({"AAPL": 200.0}, bar_date="2026-03-01")

    # Next day sell all 10 AAPL at 210.0
    # 5 bps slippage on SELL: fill price = 210.0 * (1 - 0.0005) = 209.895
    broker.submit_order(Order(order_id="s1", symbol="AAPL", side=OrderSide.SELL, quantity=10.0))
    broker.process_next_bar({"AAPL": 210.0}, bar_date="2026-03-02")

    pos = broker.get_position("AAPL")
    assert pos.quantity == 0.0

    # Realized PnL: (209.895 - 200.10) * 10 = 97.95
    bal = broker.get_account_balance()
    assert pytest.approx(bal.realized_pnl, rel=1e-4) == 97.95
    assert pytest.approx(bal.cash, rel=1e-5) == 100_000.0 + 97.95


def test_paper_broker_cancel_order() -> None:
    """Cancelling a pending order updates its status and removes it from matching queue."""
    broker = PaperBroker()
    order = Order(order_id="cancel-me", symbol="MSFT", side=OrderSide.BUY, quantity=5.0)
    broker.submit_order(order)
    assert broker.cancel_order("cancel-me") is True
    assert broker.get_orders("MSFT")[0].status == OrderStatus.CANCELLED

    # When next bar runs, no order should fill
    filled = broker.process_next_bar({"MSFT": 400.0})
    assert len(filled) == 0


def test_paper_broker_persistence(tmp_path: Path) -> None:
    """Trade logging outputs valid JSON and SQLite tables."""
    json_path = tmp_path / "trade_log.json"
    db_path = tmp_path / "trades.db"

    broker = PaperBroker(
        initial_cash=50_000.0,
        json_log_path=json_path,
        sqlite_db_path=db_path,
    )
    broker.submit_order(Order(order_id="p1", symbol="SPY", side=OrderSide.BUY, quantity=10.0))
    broker.process_next_bar({"SPY": 500.0}, bar_date="2026-03-01")

    # Verify JSON
    assert json_path.exists()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "account" in data
    assert "orders" in data
    assert len(data["orders"]) == 1
    assert data["orders"][0]["order_id"] == "p1"
    assert data["orders"][0]["status"] == OrderStatus.FILLED

    # Verify SQLite
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT order_id, symbol, status FROM orders")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0] == ("p1", "SPY", OrderStatus.FILLED)


def test_risk_guard_whitelist() -> None:
    """Assets not present in whitelist are rejected."""
    guard = RiskGuard(whitelist={"NVDA", "AAPL", "SPY"})
    broker = PaperBroker(initial_cash=100_000.0)

    # Whitelisted asset
    order_good = Order(order_id="ok", symbol="NVDA", side=OrderSide.BUY, quantity=10.0)
    res_good = guard.validate_order(order_good, broker, estimated_price=100.0)
    assert res_good.passed is True

    # Non-whitelisted asset
    order_bad = Order(order_id="bad", symbol="GME", side=OrderSide.BUY, quantity=10.0)
    res_bad = guard.validate_order(order_bad, broker, estimated_price=25.0)
    assert res_bad.passed is False
    assert "not in authorized whitelist" in res_bad.reason
    assert res_bad.order is not None
    assert res_bad.order.status == OrderStatus.REJECTED


def test_risk_guard_20pct_position_cap() -> None:
    """BUY orders that would exceed 20% NAV are clamped or rejected."""
    guard_clamping = RiskGuard(max_position_pct=0.20, clamp_to_cap=True)
    guard_rejecting = RiskGuard(max_position_pct=0.20, clamp_to_cap=False)
    broker = PaperBroker(initial_cash=100_000.0)

    # 100k equity -> max 20k per asset.
    # Price = 100. 300 shares = 30k (30% of NAV).
    order = Order(order_id="cap1", symbol="NVDA", side=OrderSide.BUY, quantity=300.0)

    # Clamping behavior: clamps to 200 shares ($20k)
    res_clamped = guard_clamping.validate_order(order, broker, estimated_price=100.0)
    assert res_clamped.passed is True
    assert res_clamped.order is not None
    assert res_clamped.order.quantity == 200.0

    # Rejecting behavior: outright rejects order
    res_rejected = guard_rejecting.validate_order(order, broker, estimated_price=100.0)
    assert res_rejected.passed is False
    assert "pushes position to 30.0%" in res_rejected.reason

    # If position already holds 200 shares (20k), subsequent BUY is rejected even with clamp_to_cap
    broker.submit_order(res_clamped.order)
    broker.process_next_bar({"NVDA": 100.0})

    res_add = guard_clamping.validate_order(
        Order(order_id="cap2", symbol="NVDA", side=OrderSide.BUY, quantity=10.0),
        broker,
        estimated_price=100.0,
    )
    assert res_add.passed is False
    assert "already at or above 20% cap" in res_add.reason


def test_risk_guard_daily_loss_kill_switch() -> None:
    """When daily loss exceeds threshold, BUY orders are blocked while SELL orders pass."""
    guard = RiskGuard(max_daily_loss_pct=0.03)  # 3% daily loss limit
    broker = PaperBroker(initial_cash=100_000.0)

    # Day start anchor at 100k
    guard.reset_daily_anchor(100_000.0, "2026-03-01")

    # Simulate severe market drop: equity falls to 95k (-5% > 3%)
    broker.cash = 95_000.0

    # BUY order should be blocked
    buy_ord = Order(order_id="b1", symbol="MSFT", side=OrderSide.BUY, quantity=5.0)
    res_buy = guard.validate_order(buy_ord, broker, estimated_price=400.0)
    assert res_buy.passed is False
    assert "Kill-switch active" in res_buy.reason

    # SELL order should be allowed to de-risk
    sell_ord = Order(order_id="s1", symbol="MSFT", side=OrderSide.SELL, quantity=5.0)
    res_sell = guard.validate_order(sell_ord, broker, estimated_price=400.0)
    assert res_sell.passed is True

    # Next day anchor reset recovers system
    guard.reset_daily_anchor(95_000.0, "2026-03-02")
    res_buy_recovered = guard.validate_order(buy_ord, broker, estimated_price=400.0)
    assert res_buy_recovered.passed is True


def test_run_execution_cycle_integration(tmp_path: Path) -> None:
    """End-to-end integration test: signal ingestion -> risk clamp -> broker fill -> json trade log."""
    log_path = tmp_path / "trade_log.json"
    weights, prices, sig_date = ingest_m7_signals(offline_fallback=True)

    broker = PaperBroker(initial_cash=100_000.0, slippage_bps=5.0)
    risk_guard = RiskGuard(max_position_pct=0.20, clamp_to_cap=True)

    summary = run_execution_cycle(
        target_weights=weights,
        prices=prices,
        signal_date=sig_date,
        broker=broker,
        risk_guard=risk_guard,
        trade_log_path=log_path,
    )

    assert summary["submitted"] > 0
    assert summary["approved"] > 0
    assert summary["filled"] > 0
    assert log_path.exists()

    with open(log_path, encoding="utf-8") as f:
        log_data = json.load(f)
    assert "orders" in log_data
    assert len(log_data["orders"]) == summary["filled"]

    # Verify that each position obeys the 20% cap
    positions = broker.get_positions()
    equity = broker.get_account_balance().equity
    for sym, pos in positions.items():
        weight = pos.market_value / equity
        assert weight <= 0.20 + 0.01  # within tolerance including slippage


def test_ingest_scaling_factor_defaults(tmp_path: Path) -> None:
    """Missing or corrupted telemetry falls back safely to 1.0."""
    non_existent = tmp_path / "does_not_exist.json"
    assert ingest_scaling_factor(non_existent) == 1.0

    empty_json = tmp_path / "empty.json"
    empty_json.write_text("{}", encoding="utf-8")
    assert ingest_scaling_factor(empty_json) == 1.0

    valid_json = tmp_path / "telemetry.json"
    valid_json.write_text(json.dumps({"signal_health": {"k": 0.75}}), encoding="utf-8")
    assert ingest_scaling_factor(valid_json) == 0.75


def test_execution_scaling_factor_k_half_capital() -> None:
    """When k=0.5, total allocated capital is halved and cash absorbs the remainder."""
    initial_cash = 100_000.0
    broker_k1 = PaperBroker(initial_cash=initial_cash, slippage_bps=0.0)
    broker_k05 = PaperBroker(initial_cash=initial_cash, slippage_bps=0.0)

    guard = RiskGuard(max_position_pct=0.50, clamp_to_cap=False)
    targets = {"NVDA": 0.20, "AAPL": 0.20}
    prices = {"NVDA": 100.0, "AAPL": 200.0}
    sig_date = "2026-03-01"

    # Base allocation at k=1.0: 40% equity ($40,000), 60% cash ($60,000)
    summary_k1 = run_execution_cycle(
        target_weights=targets,
        prices=prices,
        signal_date=sig_date,
        broker=broker_k1,
        risk_guard=guard,
        k=1.0,
    )
    # De-risked allocation at k=0.5: effective weights 10% NVDA + 10% AAPL (total 20% equity = $20,000)
    summary_k05 = run_execution_cycle(
        target_weights=targets,
        prices=prices,
        signal_date=sig_date,
        broker=broker_k05,
        risk_guard=guard,
        k=0.5,
    )

    pos_val_k1 = sum(p.market_value for p in broker_k1.get_positions().values())
    pos_val_k05 = sum(p.market_value for p in broker_k05.get_positions().values())

    cash_k1 = broker_k1.get_account_balance().cash
    cash_k05 = broker_k05.get_account_balance().cash

    # Total allocated capital is halved
    assert pytest.approx(pos_val_k1, rel=1e-3) == 40_000.0
    assert pytest.approx(pos_val_k05, rel=1e-3) == 20_000.0
    assert pytest.approx(pos_val_k05, rel=1e-3) == pos_val_k1 * 0.5

    # Cash absorbs the remainder
    assert pytest.approx(cash_k1, rel=1e-3) == 60_000.0
    assert pytest.approx(cash_k05, rel=1e-3) == 80_000.0
    assert pytest.approx(cash_k05, rel=1e-3) == initial_cash - pos_val_k05
