# -*- coding: utf-8 -*-
"""Static configuration for the AlphaShield core system.

Empirical invariants are hard-coded here on purpose: they are audit findings,
not tunables.

1. Strict TIP macro canary: when the unweighted TIP 13612 momentum
   (r21 + r63 + r126 + r252, trading days) is <= 0, ALL risky assets are
   forced to 0% exposure. There is deliberately no override/soft-gate flag.
2. Fixed core universe: URTH, SPY, QQQ, GLD, SMH (5 ETFs).
3. Equal-cap: maximum allocation per asset is 20.0% of portfolio equity.
4. Strict T+1 execution: signals from Close(t) deploy at Open(t+1) or later.
   Execution lag 0 is rejected in configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# ---------------------------------------------------------------------------
# Frozen invariants
# ---------------------------------------------------------------------------

CORE_UNIVERSE: tuple[str, ...] = ("URTH", "SPY", "QQQ", "GLD", "SMH")
CANARY_TICKER: str = "TIP"
BENCHMARK_TICKER: str = "SPY"
EQUAL_CAP: float = 0.20
TRADING_DAYS_PER_YEAR: int = 252


@dataclass(frozen=True)
class DataConfig:
    """Data ingestion settings."""

    data_start: str = "2013-01-01"  # warm-up for 252d momentum + EMA200
    backtest_start: str = "2015-01-01"
    backtest_end: str | None = None  # None = today (exclusive)
    cache_dir: str = "cache"
    refresh: bool = False  # True = force re-download and overwrite cache
    max_retries: int = 3
    retry_delay_s: float = 2.0
    warmup_days_required: int = 252

    def __post_init__(self) -> None:
        date.fromisoformat(self.data_start)
        date.fromisoformat(self.backtest_start)
        if self.backtest_end is not None:
            date.fromisoformat(self.backtest_end)


@dataclass(frozen=True)
class SignalConfig:
    """Signal & canary engine settings (invariants enforced)."""

    universe: tuple[str, ...] = CORE_UNIVERSE
    canary_ticker: str = CANARY_TICKER
    ema_spans: tuple[int, int, int] = (50, 100, 200)
    momentum_windows: tuple[int, int, int, int] = (21, 63, 126, 252)
    hard_exit_buffer: float = 0.98  # Close < EMA200 * 0.98 -> force 0%
    max_weight_per_asset: float = EQUAL_CAP

    def __post_init__(self) -> None:
        if self.universe != CORE_UNIVERSE:
            raise ValueError("Core universe is a frozen invariant (URTH/SPY/QQQ/GLD/SMH).")
        if self.max_weight_per_asset != EQUAL_CAP:
            raise ValueError("Equal-cap 20% is a frozen invariant.")
        f, m, s = self.ema_spans
        if not (0 < f < m < s):
            raise ValueError("EMA spans must be strictly increasing (50, 100, 200).")
        if not 0.0 < self.hard_exit_buffer < 1.0:
            raise ValueError("hard_exit_buffer must be in (0, 1).")


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio construction / execution settings."""

    cost_bps: float = 5.0  # friction per unit of absolute (one-sided) turnover
    execution_lag: int = 1  # STRICT T+1; 0 is forbidden (look-ahead bias)
    cash_yield_annual: float = 0.02  # money-market park (SCBTMFPLUS-E / BIL proxy)
    initial_capital: float = 1_000_000.0

    def __post_init__(self) -> None:
        if self.execution_lag < 1:
            raise ValueError("Execution lag must be >= 1 (strict T+1; T+0 bias forbidden).")
        if self.cost_bps < 0 or self.cash_yield_annual < 0:
            raise ValueError("cost_bps and cash_yield_annual must be non-negative.")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
