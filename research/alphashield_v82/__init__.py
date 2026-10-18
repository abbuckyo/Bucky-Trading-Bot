# -*- coding: utf-8 -*-
"""AlphaShield — modular V8.1 backtest & institutional audit suite."""

from .config import (
    BENCHMARK_TICKER,
    CANARY_TICKER,
    CORE_UNIVERSE,
    EQUAL_CAP,
    DataConfig,
    PortfolioConfig,
    SignalConfig,
)
from .data_engine import DataEngine, MarketData
from .signal_engine import SignalEngine, SignalFrame
from .portfolio_engine import Ledger, PortfolioEngine

__version__ = "8.1.0"

__all__ = [
    "BENCHMARK_TICKER",
    "CANARY_TICKER",
    "CORE_UNIVERSE",
    "EQUAL_CAP",
    "DataConfig",
    "DataEngine",
    "Ledger",
    "MarketData",
    "PortfolioConfig",
    "PortfolioEngine",
    "SignalConfig",
    "SignalEngine",
    "SignalFrame",
]
