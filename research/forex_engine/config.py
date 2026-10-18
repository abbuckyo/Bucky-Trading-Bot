# -*- coding: utf-8 -*-
"""Configuration and invariants for the Regime-Switched G10 Forex Engine (Engine 3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent
CACHE_DIR: Path = BASE_DIR / "cache"
OUTPUT_DIR: Path = BASE_DIR / "output"

G10_UNIVERSE: tuple[str, ...] = (
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
)

DOLLAR_INDEX_TICKER: str = "DX-Y.NYB"
EQUITY_BENCHMARK_TICKER: str = "SPY"

TRADING_DAYS_PER_YEAR: int = 252
MAX_PAIR_WEIGHT: float = 0.20
MAX_GROSS_LEVERAGE: float = 1.00
TARGET_PAIR_VOL_ANNUAL: float = 0.025
DEFAULT_COST_BPS: float = 3.0
DEFAULT_CASH_YIELD: float = 0.02


@dataclass(frozen=True)
class ForexConfig:
    """Master configuration enforcing all G10 Forex quantitative invariants."""

    # Universe & macro benchmarks
    universe: tuple[str, ...] = G10_UNIVERSE
    dollar_index: str = DOLLAR_INDEX_TICKER
    equity_benchmark: str = EQUITY_BENCHMARK_TICKER

    # Data & cache settings
    data_start: str = "2015-01-01"
    backtest_start: str = "2017-01-01"
    backtest_end: str | None = None
    cache_dir: Path = CACHE_DIR
    output_dir: Path = OUTPUT_DIR
    refresh: bool = False
    max_retries: int = 3
    retry_delay_s: float = 2.0
    warmup_days_required: int = 200

    # Regime filter parameters (Wilder ADX_14 & Rolling Hurst H_100)
    adx_period: int = 14
    hurst_window: int = 100
    adx_trend_threshold: float = 22.0
    hurst_trend_threshold: float = 0.53
    adx_choppy_threshold: float = 18.0
    hurst_choppy_threshold: float = 0.48

    # Signal & carry confirmation parameters
    ema_fast: int = 20
    ema_slow: int = 50
    mom_fast_window: int = 63
    mom_slow_window: int = 126
    vol_fast_window: int = 20
    vol_slow_window: int = 60

    # Sizing & trailing stop parameters
    target_pair_vol: float = TARGET_PAIR_VOL_ANNUAL  # 2.5% annual vol per pair
    max_pair_weight: float = MAX_PAIR_WEIGHT        # 20.0% max notional per pair
    max_gross_leverage: float = MAX_GROSS_LEVERAGE  # 1.0x max gross exposure
    atr_window: int = 14
    trailing_stop_window: int = 20
    trailing_stop_atr_mult: float = 2.5

    # Execution & ledger invariants
    cost_bps: float = DEFAULT_COST_BPS              # 3.0 bps per rebalance
    execution_lag: int = 1                          # Strict T+1 execution lag
    cash_yield_annual: float = DEFAULT_CASH_YIELD   # 2.0% annual risk-free yield
    initial_capital: float = 1_000_000.0

    def __post_init__(self) -> None:
        date.fromisoformat(self.data_start)
        date.fromisoformat(self.backtest_start)
        if self.backtest_end is not None:
            date.fromisoformat(self.backtest_end)
        if self.execution_lag < 1:
            raise ValueError("execution_lag must be >= 1 (strict T+1 invariant; T+0 forbidden).")
        if not (0.0 < self.max_pair_weight <= MAX_PAIR_WEIGHT + 1e-12):
            raise ValueError(f"max_pair_weight must be in (0, {MAX_PAIR_WEIGHT}].")
        if not (0.0 < self.max_gross_leverage <= MAX_GROSS_LEVERAGE + 1e-12):
            raise ValueError(f"max_gross_leverage must be in (0, {MAX_GROSS_LEVERAGE}].")
        if self.target_pair_vol <= 0.0:
            raise ValueError("target_pair_vol must be positive.")
        if self.cost_bps < 0.0 or self.cash_yield_annual < 0.0:
            raise ValueError("cost_bps and cash_yield_annual must be non-negative.")
