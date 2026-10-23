# -*- coding: utf-8 -*-
"""Abstract Base Broker Interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    def __str__(self) -> str:
        return self.value


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

    def __str__(self) -> str:
        return self.value


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

    def __str__(self) -> str:
        return self.value


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide | str
    quantity: float
    order_type: OrderType | str = OrderType.MARKET
    limit_price: float | None = None
    status: OrderStatus | str = OrderStatus.PENDING
    created_at: str = ""
    filled_at: str | None = None
    filled_price: float | None = None
    filled_quantity: float = 0.0
    slippage_bps: float = 0.0
    fee: float = 0.0
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["side"] = str(getattr(self.side, "value", self.side))
        d["order_type"] = str(getattr(self.order_type, "value", self.order_type))
        d["status"] = str(getattr(self.status, "value", self.status))
        return d


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0

    def update_price(self, price: float) -> None:
        self.current_price = price
        self.market_value = self.quantity * price
        self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AccountBalance:
    cash: float
    equity: float
    initial_balance: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseBroker(ABC):
    """Abstract interface for brokerage interactions."""

    @abstractmethod
    def get_account_balance(self) -> AccountBalance:
        """Return the current account balance and equity summary."""
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Return all active positions keyed by symbol."""
        raise NotImplementedError

    @abstractmethod
    def get_position(self, symbol: str) -> Position:
        """Return the position for a given symbol."""
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker."""
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by ID."""
        raise NotImplementedError

    @abstractmethod
    def get_orders(self, symbol: str | None = None) -> list[Order]:
        """Return list of orders, optionally filtered by symbol."""
        raise NotImplementedError
