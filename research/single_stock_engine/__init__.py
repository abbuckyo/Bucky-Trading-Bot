# -*- coding: utf-8 -*-
"""Cross-Sectional Single-Stock Momentum Engine (Research Module).

Trades a 30-stock US Large-Cap universe using Cross-Sectional Volatility-Adjusted
Momentum (12M-1M / 60D Realized Vol), governed by a broad market regime filter
(SPY > EMA200), individual trend guards (Close > EMA100), Earnings Blackouts
(5-day pre-earnings entry freeze + 1.5x ATR stop tightening), Chandelier Trailing
Exits (High_22 - 3.0x ATR_20), and Fixed Fractional Risk Parity sizing (0.75%
NAV risk budget, 15% max weight cap, strict T+1 execution, 5 bps friction,
2.0% cash park).
"""
