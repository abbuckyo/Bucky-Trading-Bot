# -*- coding: utf-8 -*-
"""Regime-Switched G10 Forex Quantitative Engine (Engine 3 — Research Module).

Trades 7 liquid G10 FX spot pairs symmetrically (Long +1 / Short -1) using:
  * Wilder's ADX(14) + 100-day Rolling Hurst Exponent (H_100) regime switching,
  * EMA(20/50) + DI(14) breakout trend signals confirmed by 63d/126d
    volatility-adjusted momentum,
  * Choppy regime guard (ADX_14 < 18 and H_100 < 0.48 -> flatten to 0.0),
  * 20-day Close Chandelier-style trailing stop (2.5x ATR_14),
  * Inverse-volatility position sizing (2.5% target annual vol per pair,
    20% max per pair, 1.0x max total gross exposure),
  * Strict T+1 execution ledger with 3 bps transaction costs and 2.0% cash yield.
"""
