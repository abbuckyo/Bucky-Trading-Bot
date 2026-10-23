# -*- coding: utf-8 -*-
"""Execution Runner: Ingest Signals, Route via Risk Guard, Execute on Paper Broker."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repository roots are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUCKY_ROOT = _REPO_ROOT / "Bucky-Trading-Bot" if (_REPO_ROOT / "Bucky-Trading-Bot").exists() else _REPO_ROOT
for p in (_BUCKY_ROOT, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from execution.base_broker import Order, OrderSide, OrderStatus, OrderType
from execution.paper_broker import PaperBroker
from execution.risk_guard import RiskGuard

DOCS_DIR = _BUCKY_ROOT / "docs" if (_BUCKY_ROOT / "docs").exists() else _REPO_ROOT / "docs"
DEFAULT_TRADE_LOG = DOCS_DIR / "trade_log.json"
DEFAULT_TELEMETRY = DOCS_DIR / "telemetry.json"
DEFAULT_SQLITE_DB = Path(__file__).resolve().parent / "trade_log.db"


def ingest_scaling_factor(telemetry_path: Path | str | None = None) -> float:
    """Ingest health scaling factor k from telemetry.json (defaults to 1.0 if missing)."""
    p = Path(telemetry_path) if telemetry_path else DEFAULT_TELEMETRY
    if not p.exists():
        return 1.0
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        k_val = data.get("signal_health", {}).get("k")
        if k_val is None:
            k_val = data.get("k", 1.0)
        return float(k_val)
    except Exception:
        return 1.0


def ingest_m7_signals(offline_fallback: bool = True) -> tuple[dict[str, float], dict[str, float], str]:
    """Ingest signals from the M7 concentrated momentum sleeve.

    Returns:
      (target_weights, current_prices, signal_date)
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        from research.single_stock_engine.run_m7 import M7_UNIVERSE, M7Config
        # Top-3 names with nominal 30% allocation each (to be capped by RiskGuard at 20%)
        m7_names = list(M7_UNIVERSE[:3])
        weights = {sym: 0.30 for sym in m7_names}
        prices = {
            "NVDA": 130.0,
            "MSFT": 420.0,
            "AAPL": 225.0,
            "AMZN": 185.0,
            "GOOGL": 165.0,
            "META": 580.0,
            "TSLA": 215.0,
            "SPY": 575.0,
        }
        return weights, prices, date_str
    except Exception as exc:  # noqa: BLE001
        if not offline_fallback:
            raise
        print(f"[Warning] M7 signal import warning ({exc}); using fallback signals.")
        weights = {"NVDA": 0.30, "MSFT": 0.30, "AAPL": 0.30}
        prices = {"NVDA": 130.0, "MSFT": 420.0, "AAPL": 225.0, "SPY": 575.0}
        return weights, prices, date_str


def ingest_blend_signals(offline_fallback: bool = True) -> tuple[dict[str, float], dict[str, float], str]:
    """Ingest blended sleeve allocation signals (Core ETF + Single Stock).

    Returns:
      (target_weights, current_prices, signal_date)
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        from research.alphashield_v82.config import CORE_UNIVERSE
        from research.single_stock_engine.run_m7 import M7_UNIVERSE
        # 50/50 blend: ETF sleeve (SPY) + Top M7 names
        weights = {
            "SPY": 0.20,
            "NVDA": 0.25,
            "MSFT": 0.25,
            "AAPL": 0.20,
        }
        prices = {
            "SPY": 575.0,
            "NVDA": 130.0,
            "MSFT": 420.0,
            "AAPL": 225.0,
        }
        return weights, prices, date_str
    except Exception as exc:  # noqa: BLE001
        if not offline_fallback:
            raise
        print(f"[Warning] Blend signal import warning ({exc}); using fallback signals.")
        weights = {"SPY": 0.20, "NVDA": 0.25, "MSFT": 0.25, "AAPL": 0.20}
        prices = {"SPY": 575.0, "NVDA": 130.0, "MSFT": 420.0, "AAPL": 225.0}
        return weights, prices, date_str


def construct_rebalance_orders(
    target_weights: dict[str, float],
    current_prices: dict[str, float],
    broker: PaperBroker,
    order_date: str,
) -> tuple[list[Order], list[Order]]:
    """Derive delta orders to transition portfolio to target weights.

    Separates into sell orders first (to free cash) and buy orders second.
    """
    equity = broker.get_account_balance().equity
    symbols = set(target_weights.keys()) | set(broker.get_positions().keys())

    sell_orders: list[Order] = []
    buy_orders: list[Order] = []

    for sym in sorted(symbols):
        price = current_prices.get(sym, 0.0)
        if price <= 0:
            continue

        target_w = target_weights.get(sym, 0.0)
        target_val = equity * target_w
        target_qty = target_val / price

        curr_pos = broker.get_position(sym)
        curr_qty = curr_pos.quantity

        delta_qty = target_qty - curr_qty

        if delta_qty < -1e-4:
            sell_qty = round(abs(delta_qty), 4)
            order = Order(
                order_id=str(uuid.uuid4())[:8],
                symbol=sym,
                side=OrderSide.SELL,
                quantity=sell_qty,
                order_type=OrderType.MARKET,
                created_at=order_date,
            )
            sell_orders.append(order)
        elif delta_qty > 1e-4:
            buy_qty = round(delta_qty, 4)
            order = Order(
                order_id=str(uuid.uuid4())[:8],
                symbol=sym,
                side=OrderSide.BUY,
                quantity=buy_qty,
                order_type=OrderType.MARKET,
                created_at=order_date,
            )
            buy_orders.append(order)

    return sell_orders, buy_orders


def run_execution_cycle(
    target_weights: dict[str, float],
    prices: dict[str, float],
    signal_date: str,
    broker: PaperBroker,
    risk_guard: RiskGuard,
    trade_log_path: Path | str | None = None,
    telemetry_path: Path | str | None = None,
    k: float | None = None,
) -> dict[str, Any]:
    """Execute end-to-end signal routing, risk check, next-bar execution & logging."""
    log_target = Path(trade_log_path) if trade_log_path else DEFAULT_TRADE_LOG
    log_target.parent.mkdir(parents=True, exist_ok=True)

    # Ingest health scaling factor k (defaults to 1.0 if missing) and scale target weights
    k_factor = float(k) if k is not None else ingest_scaling_factor(telemetry_path)
    effective_weights = {sym: w * k_factor for sym, w in target_weights.items()}

    # 1. Update prices in broker
    broker.update_market_prices(prices)

    # 2. Daily anchor initialization
    if risk_guard.day_start_equity is None:
        risk_guard.reset_daily_anchor(broker.get_account_balance().equity, signal_date)

    # 3. Formulate orders using effective weights (scaled by k)
    sells, buys = construct_rebalance_orders(effective_weights, prices, broker, signal_date)
    planned_orders = sells + buys

    approved_orders: list[Order] = []
    rejected_orders: list[Order] = []

    # 4. Route through Risk Guard
    for ord_req in planned_orders:
        price = prices.get(ord_req.symbol, 0.0)
        check = risk_guard.validate_order(ord_req, broker, price)
        if check.passed and check.order:
            approved_orders.append(check.order)
            broker.submit_order(check.order)
        else:
            if check.order:
                rejected_orders.append(check.order)
            else:
                ord_req.status = OrderStatus.REJECTED
                ord_req.rejection_reason = check.reason
                rejected_orders.append(ord_req)

    # 5. Process T+1 next bar matching (simulating next session matching)
    filled_orders = broker.process_next_bar(prices, bar_date=signal_date)

    # 6. Save trade log
    broker.log_to_json(log_target)
    if broker.sqlite_db_path:
        broker.log_to_sqlite()

    return {
        "signal_date": signal_date,
        "k": k_factor,
        "effective_weights": effective_weights,
        "submitted": len(planned_orders),
        "approved": len(approved_orders),
        "rejected": len(rejected_orders),
        "filled": len(filled_orders),
        "account": broker.get_account_balance().to_dict(),
        "trade_log_path": str(log_target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Modular Execution Scaffold Runner")
    parser.add_argument(
        "--source",
        choices=["m7", "blend"],
        default="m7",
        help="Signal source (m7 or blend)",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100_000.0,
        help="Initial broker cash",
    )
    parser.add_argument(
        "--trade-log",
        type=str,
        default=str(DEFAULT_TRADE_LOG),
        help="Path to output trade log JSON",
    )
    parser.add_argument(
        "--sqlite-db",
        type=str,
        default=str(DEFAULT_SQLITE_DB),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--max-cap",
        type=float,
        default=0.20,
        help="Max position size cap (default: 0.20 for 20%%)",
    )
    parser.add_argument(
        "--telemetry",
        type=str,
        default=str(DEFAULT_TELEMETRY),
        help="Path to telemetry JSON for scaling factor k",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=None,
        help="Explicit scaling factor override (default: ingest from telemetry.json)",
    )
    args = parser.parse_args()

    k_factor = args.k if args.k is not None else ingest_scaling_factor(args.telemetry)

    # Ingest signals
    if args.source == "m7":
        weights, prices, sig_date = ingest_m7_signals()
    else:
        weights, prices, sig_date = ingest_blend_signals()

    print(f"[{args.source.upper()}] Ingested target weights for {sig_date} (k={k_factor:.2f}):")
    for sym, w in weights.items():
        eff_w = w * k_factor
        print(f"  {sym:6s}: target={w:.1%} -> effective={eff_w:.1%}")

    broker = PaperBroker(
        initial_cash=args.cash,
        slippage_bps=5.0,
        json_log_path=args.trade_log,
        sqlite_db_path=args.sqlite_db,
    )
    risk_guard = RiskGuard(
        max_position_pct=args.max_cap,
        clamp_to_cap=True,
    )

    summary = run_execution_cycle(
        target_weights=weights,
        prices=prices,
        signal_date=sig_date,
        broker=broker,
        risk_guard=risk_guard,
        trade_log_path=args.trade_log,
        telemetry_path=args.telemetry,
        k=k_factor,
    )

    print("\nExecution Summary:")
    print(f"  Scaling Factor k: {summary['k']:.2f}")
    print(f"  Orders Submitted: {summary['submitted']}")
    print(f"  Orders Approved : {summary['approved']}")
    print(f"  Orders Rejected : {summary['rejected']}")
    print(f"  Orders Filled   : {summary['filled']}")
    print(f"  Account Equity  : ${summary['account']['equity']:,.2f}")
    print(f"  Available Cash  : ${summary['account']['cash']:,.2f}")
    print(f"  Trade Log Saved : {summary['trade_log_path']}")


if __name__ == "__main__":
    main()
