# -*- coding: utf-8 -*-
"""Configuration and invariants for the Cross-Sectional Single-Stock Momentum Engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent
CACHE_DIR: Path = BASE_DIR / "cache"
OUTPUT_DIR: Path = BASE_DIR / "output"

BENCHMARK_TICKER: str = "SPY"

TARGET_UNIVERSE: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "UNH", "LLY", "V", "XOM", "MA", "JNJ", "HD",
    "PG", "COST", "ABBV", "MRK", "BAC", "CRM", "ADBE",
    "AMD", "NFLX", "WMT", "CVX", "KO", "PEP", "CSCO", "QCOM",
)

TRADING_DAYS_PER_YEAR: int = 252
MAX_SINGLE_STOCK_WEIGHT: float = 0.15
TARGET_RISK_PER_TRADE: float = 0.0075
DEFAULT_COST_BPS: float = 5.0
DEFAULT_CASH_YIELD: float = 0.02


@dataclass(frozen=True)
class EngineConfig:
    """Master configuration enforcing all single-stock quantitative invariants."""

    # Universe & benchmark
    universe: tuple[str, ...] = TARGET_UNIVERSE
    benchmark: str = BENCHMARK_TICKER

    # Data settings
    data_start: str = "2015-01-01"
    backtest_start: str = "2017-01-01"
    backtest_end: str | None = None
    cache_dir: Path = CACHE_DIR
    output_dir: Path = OUTPUT_DIR
    refresh: bool = False
    max_retries: int = 3
    retry_delay_s: float = 2.0
    warmup_days_required: int = 252

    # Portfolio & sizing constraints
    min_active_holdings: int = 5
    max_active_holdings: int = 10
    top_quantile: float = 0.20  # Top 20% of 30 stocks = 6 stocks (within [5, 10])
    max_weight_per_stock: float = MAX_SINGLE_STOCK_WEIGHT  # 15.0% hard cap
    target_risk_per_trade: float = TARGET_RISK_PER_TRADE  # 0.75% of NAV

    # Macro & individual trend filters
    spy_ema_window: int = 200
    stock_ema_window: int = 100

    # Cross-sectional momentum parameters
    mom_lookback_long: int = 252
    mom_lookback_skip: int = 21
    vol_lookback: int = 60
    vol_floor: float = 0.05

    # Risk guards & Chandelier exit parameters
    atr_window: int = 20
    chandelier_high_window: int = 22
    chandelier_atr_mult: float = 3.0
    earnings_blackout_days: int = 5
    earnings_tight_atr_mult: float = 1.5

    # Execution & ledger invariants
    cost_bps: float = DEFAULT_COST_BPS  # 5 bps per side
    execution_lag: int = 1  # Strict T+1 lag
    cash_yield_annual: float = DEFAULT_CASH_YIELD  # 2.0% money-market proxy
    initial_capital: float = 1_000_000.0

    def __post_init__(self) -> None:
        date.fromisoformat(self.data_start)
        date.fromisoformat(self.backtest_start)
        if self.backtest_end is not None:
            date.fromisoformat(self.backtest_end)
        if self.execution_lag < 1:
            raise ValueError("Execution lag must be >= 1 (strict T+1 invariant; T+0 forbidden).")
        if not (0.0 < self.max_weight_per_stock <= MAX_SINGLE_STOCK_WEIGHT + 1e-12):
            raise ValueError("max_weight_per_stock must be in (0, 0.15].")
        if self.target_risk_per_trade <= 0.0:
            raise ValueError("target_risk_per_trade must be positive.")
        if self.mom_lookback_long <= self.mom_lookback_skip:
            raise ValueError("mom_lookback_long must exceed mom_lookback_skip.")
        if self.cost_bps < 0.0 or self.cash_yield_annual < 0.0:
            raise ValueError("cost_bps and cash_yield_annual must be non-negative.")
