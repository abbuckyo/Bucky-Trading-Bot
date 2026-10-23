# -*- coding: utf-8 -*-
"""Pre-Trade Risk Guard: Position Caps, Asset Whitelists & Kill-Switch."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .base_broker import BaseBroker, Order, OrderSide, OrderStatus

DEFAULT_WHITELIST: frozenset[str] = frozenset(
    {"NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "SPY"}
)


@dataclass
class RiskCheckResult:
    passed: bool
    reason: str
    order: Order | None = None


class RiskGuard:
    """Pre-trade risk gateway enforcing firm safety invariants.

    Invariants:
      1. Position Size Cap: No asset position may exceed 20% of total NAV.
      2. Whitelist: Orders for non-whitelisted assets are rejected.
      3. Daily Loss Kill-Switch: If account daily loss breaches max drawdown limit,
         all new BUY entries are blocked while SELL (de-risking) orders remain open.
    """

    def __init__(
        self,
        max_position_pct: float = 0.20,
        whitelist: set[str] | frozenset[str] | list[str] | None = None,
        max_daily_loss_pct: float = 0.03,
        clamp_to_cap: bool = True,
    ) -> None:
        self.max_position_pct: float = float(max_position_pct)
        self.whitelist: set[str] = set(whitelist) if whitelist is not None else set(DEFAULT_WHITELIST)
        self.max_daily_loss_pct: float = float(max_daily_loss_pct)
        self.clamp_to_cap: bool = clamp_to_cap

        self.day_start_equity: float | None = None
        self.kill_switch_active: bool = False
        self.last_check_date: str | None = None

    def reset_daily_anchor(self, current_equity: float, date_str: str | None = None) -> None:
        """Reset the daily reference equity benchmark at the start of a trading session."""
        self.day_start_equity = float(current_equity)
        self.kill_switch_active = False
        self.last_check_date = date_str

    def check_daily_loss(self, current_equity: float) -> bool:
        """Evaluate if the daily loss threshold has been breached."""
        if self.day_start_equity is None or self.day_start_equity <= 0:
            self.day_start_equity = current_equity
            return False

        daily_return = (current_equity - self.day_start_equity) / self.day_start_equity
        if daily_return < -self.max_daily_loss_pct:
            self.kill_switch_active = True
        return self.kill_switch_active

    def validate_order(
        self,
        order: Order,
        broker: BaseBroker,
        estimated_price: float,
    ) -> RiskCheckResult:
        """Audit order against risk invariants before routing to broker."""
        balance = broker.get_account_balance()
        equity = max(balance.equity, balance.cash, 1.0)
        side_val = getattr(order.side, "value", order.side)
        side = str(side_val).upper()

        # 1. Daily loss kill-switch check
        if self.check_daily_loss(balance.equity):
            if side in (OrderSide.BUY, "BUY"):
                order_copy = copy.copy(order)
                order_copy.status = OrderStatus.REJECTED
                order_copy.rejection_reason = (
                    f"Kill-switch active: Daily loss exceeded {self.max_daily_loss_pct:.1%}"
                )
                return RiskCheckResult(
                    passed=False,
                    reason=order_copy.rejection_reason,
                    order=order_copy,
                )

        # 2. Asset whitelist check
        if order.symbol not in self.whitelist:
            order_copy = copy.copy(order)
            order_copy.status = OrderStatus.REJECTED
            order_copy.rejection_reason = (
                f"Asset '{order.symbol}' not in authorized whitelist"
            )
            return RiskCheckResult(
                passed=False,
                reason=order_copy.rejection_reason,
                order=order_copy,
            )

        # 3. Position size cap (20% max NAV)
        if side in (OrderSide.BUY, "BUY"):
            if estimated_price <= 0:
                order_copy = copy.copy(order)
                order_copy.status = OrderStatus.REJECTED
                order_copy.rejection_reason = "Invalid asset price (<= 0)"
                return RiskCheckResult(passed=False, reason=order_copy.rejection_reason, order=order_copy)

            max_allowed_notional = equity * self.max_position_pct
            current_pos = broker.get_position(order.symbol)
            current_notional = current_pos.quantity * estimated_price
            order_notional = order.quantity * estimated_price
            total_projected_notional = current_notional + order_notional

            if total_projected_notional > max_allowed_notional + 1e-4:
                if self.clamp_to_cap:
                    remaining_capacity = max(0.0, max_allowed_notional - current_notional)
                    allowed_quantity = remaining_capacity / estimated_price

                    if allowed_quantity <= 1e-6:
                        order_copy = copy.copy(order)
                        order_copy.status = OrderStatus.REJECTED
                        order_copy.rejection_reason = (
                            f"Position for {order.symbol} already at or above "
                            f"{self.max_position_pct:.0%} cap ({current_notional/equity:.1%})"
                        )
                        return RiskCheckResult(
                            passed=False,
                            reason=order_copy.rejection_reason,
                            order=order_copy,
                        )

                    clamped_order = copy.copy(order)
                    clamped_order.quantity = round(allowed_quantity, 4)
                    return RiskCheckResult(
                        passed=True,
                        reason=(
                            f"Clamped order from {order.quantity} to {clamped_order.quantity} "
                            f"to respect {self.max_position_pct:.0%} cap"
                        ),
                        order=clamped_order,
                    )
                else:
                    order_copy = copy.copy(order)
                    order_copy.status = OrderStatus.REJECTED
                    order_copy.rejection_reason = (
                        f"Order notional (${order_notional:,.2f}) pushes position to "
                        f"{total_projected_notional/equity:.1%} (cap: {self.max_position_pct:.0%})"
                    )
                    return RiskCheckResult(
                        passed=False,
                        reason=order_copy.rejection_reason,
                        order=order_copy,
                    )

        return RiskCheckResult(passed=True, reason="Risk checks passed", order=order)
