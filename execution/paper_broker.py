# -*- coding: utf-8 -*-
"""Mock Paper Broker with T+1 Next-Bar Matching, Slippage & Persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .base_broker import (
    AccountBalance,
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)


class PaperBroker(BaseBroker):
    """Paper execution broker implementing strict T+1 next-bar matching.

    Features:
      * Next-bar matching: orders queued at t match at t+1 open/price.
      * Configurable slippage: default 5 bps (0.0005).
      * Persistence: exports execution and trade logs to JSON and SQLite.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_bps: float = 5.0,
        allow_partial_fills: bool = False,
        json_log_path: Path | str | None = None,
        sqlite_db_path: Path | str | None = None,
    ) -> None:
        self.initial_cash: float = float(initial_cash)
        self.cash: float = float(initial_cash)
        self.slippage_bps: float = float(slippage_bps)
        self.allow_partial_fills: bool = allow_partial_fills
        self.json_log_path: Path | None = Path(json_log_path) if json_log_path else None
        self.sqlite_db_path: Path | None = Path(sqlite_db_path) if sqlite_db_path else None

        self._positions: dict[str, Position] = {}
        self._pending_orders: list[Order] = []
        self._all_orders: list[Order] = []
        self._realized_pnl: float = 0.0

        if self.sqlite_db_path:
            self._init_sqlite_db()

    def get_account_balance(self) -> AccountBalance:
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        portfolio_val = sum(p.market_value for p in self._positions.values())
        equity = self.cash + portfolio_val
        return AccountBalance(
            cash=round(self.cash, 4),
            equity=round(equity, 4),
            initial_balance=round(self.initial_cash, 4),
            realized_pnl=round(self._realized_pnl, 4),
            unrealized_pnl=round(unrealized, 4),
        )

    def get_positions(self) -> dict[str, Position]:
        return {s: p for s, p in self._positions.items() if p.quantity > 1e-9}

    def get_position(self, symbol: str) -> Position:
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        return self._positions[symbol]

    def submit_order(self, order: Order) -> Order:
        """Queue order for T+1 next-bar matching."""
        order.status = OrderStatus.PENDING
        order.slippage_bps = self.slippage_bps
        self._pending_orders.append(order)
        self._all_orders.append(order)
        self._sync_logs()
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for o in self._pending_orders:
            if o.order_id == order_id:
                o.status = OrderStatus.CANCELLED
                self._pending_orders.remove(o)
                self._sync_logs()
                return True
        return False

    def get_orders(self, symbol: str | None = None) -> list[Order]:
        if symbol is None:
            return list(self._all_orders)
        return [o for o in self._all_orders if o.symbol == symbol]

    def update_market_prices(self, prices: dict[str, float]) -> None:
        """Update market values and unrealized PnL of all holdings."""
        for sym, price in prices.items():
            if sym in self._positions:
                self._positions[sym].update_price(price)

    def process_next_bar(
        self,
        prices: dict[str, float],
        bar_date: str | None = None,
    ) -> list[Order]:
        """Match pending orders against next-bar price with 5 bps slippage."""
        filled_orders: list[Order] = []
        unmatched_orders: list[Order] = []

        # First update market prices for accurate portfolio valuation
        self.update_market_prices(prices)

        for order in self._pending_orders:
            symbol = order.symbol
            if symbol not in prices or prices[symbol] <= 0:
                unmatched_orders.append(order)
                continue

            base_price = float(prices[symbol])
            slip_multiplier = self.slippage_bps / 10_000.0

            side_val = getattr(order.side, "value", order.side)
            side = str(side_val).upper()

            if side in (OrderSide.BUY, "BUY"):
                fill_price = base_price * (1.0 + slip_multiplier)
                cost = fill_price * order.quantity

                if cost > self.cash:
                    if self.allow_partial_fills and self.cash > fill_price:
                        fill_qty = int(self.cash // fill_price)
                        cost = fill_price * fill_qty
                    else:
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "Insufficient cash for BUY execution"
                        filled_orders.append(order)
                        continue
                else:
                    fill_qty = order.quantity

                self.cash -= cost
                pos = self.get_position(symbol)
                new_qty = pos.quantity + fill_qty
                pos.avg_entry_price = (
                    (pos.quantity * pos.avg_entry_price + fill_qty * fill_price) / new_qty
                    if new_qty > 0
                    else fill_price
                )
                pos.quantity = new_qty
                pos.update_price(base_price)

                order.status = OrderStatus.FILLED
                order.filled_price = round(fill_price, 4)
                order.filled_quantity = round(fill_qty, 6)
                order.filled_at = bar_date
                filled_orders.append(order)

            elif side in (OrderSide.SELL, "SELL"):
                fill_price = base_price * (1.0 - slip_multiplier)
                pos = self.get_position(symbol)

                if pos.quantity <= 1e-9:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = "No position to sell"
                    filled_orders.append(order)
                    continue

                fill_qty = min(order.quantity, pos.quantity)
                revenue = fill_price * fill_qty
                self.cash += revenue

                realized = (fill_price - pos.avg_entry_price) * fill_qty
                self._realized_pnl += realized

                pos.quantity -= fill_qty
                if pos.quantity <= 1e-9:
                    pos.quantity = 0.0
                    pos.avg_entry_price = 0.0
                pos.update_price(base_price)

                order.status = OrderStatus.FILLED
                order.filled_price = round(fill_price, 4)
                order.filled_quantity = round(fill_qty, 6)
                order.filled_at = bar_date
                filled_orders.append(order)

        self._pending_orders = unmatched_orders
        self.update_market_prices(prices)
        self._sync_logs()
        return filled_orders

    def log_to_json(self, file_path: Path | str) -> None:
        """Write trade/order records to a JSON file."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "account": self.get_account_balance().to_dict(),
            "positions": {k: v.to_dict() for k, v in self.get_positions().items()},
            "orders": [o.to_dict() for o in self._all_orders],
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _init_sqlite_db(self) -> None:
        if not self.sqlite_db_path:
            return
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    order_type TEXT,
                    status TEXT,
                    created_at TEXT,
                    filled_at TEXT,
                    filled_price REAL,
                    filled_quantity REAL,
                    slippage_bps REAL,
                    fee REAL,
                    rejection_reason TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    cash REAL,
                    equity REAL,
                    realized_pnl REAL,
                    unrealized_pnl REAL
                )
                """
            )

    def log_to_sqlite(self, db_path: Path | str | None = None) -> None:
        """Persist order records to SQLite database."""
        path = Path(db_path) if db_path else self.sqlite_db_path
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            for o in self._all_orders:
                side_str = str(getattr(o.side, "value", o.side))
                type_str = str(getattr(o.order_type, "value", o.order_type))
                status_str = str(getattr(o.status, "value", o.status))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO orders (
                        order_id, symbol, side, quantity, order_type, status,
                        created_at, filled_at, filled_price, filled_quantity,
                        slippage_bps, fee, rejection_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        o.order_id,
                        o.symbol,
                        side_str,
                        o.quantity,
                        type_str,
                        status_str,
                        o.created_at,
                        o.filled_at,
                        o.filled_price,
                        o.filled_quantity,
                        o.slippage_bps,
                        o.fee,
                        o.rejection_reason,
                    ),
                )
            bal = self.get_account_balance()
            conn.execute(
                """
                INSERT INTO account_history (
                    timestamp, cash, equity, realized_pnl, unrealized_pnl
                ) VALUES (datetime('now'), ?, ?, ?, ?)
                """,
                (bal.cash, bal.equity, bal.realized_pnl, bal.unrealized_pnl),
            )

    def _sync_logs(self) -> None:
        if self.json_log_path:
            self.log_to_json(self.json_log_path)
        if self.sqlite_db_path:
            self.log_to_sqlite(self.sqlite_db_path)
