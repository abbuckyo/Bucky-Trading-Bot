import numpy as np
import pandas as pd
import pytest
from bot import rsi_wilder, CFG


def test_rsi_wilder_uptrend():
    """In a pure uptrend, RSI should be 100.0 after the initial warmup period."""
    length = 14
    prices = pd.Series([10.0 + i for i in range(30)])
    rsi = rsi_wilder(prices, length=length)

    # Initial warmup period (length items, accounting for diff NaN at index 0)
    assert rsi.iloc[:length].isna().all()
    # After warmup, pure uptrend results in RSI = 100.0
    assert (rsi.iloc[length:] == 100.0).all()


def test_rsi_wilder_downtrend():
    """In a pure downtrend, RSI should be 0.0 after the initial warmup period."""
    length = 14
    prices = pd.Series([100.0 - i for i in range(30)])
    rsi = rsi_wilder(prices, length=length)

    # Initial warmup period
    assert rsi.iloc[:length].isna().all()
    # After warmup, pure downtrend results in RSI = 0.0
    assert (rsi.iloc[length:] == 0.0).all()


def test_rsi_wilder_flat():
    """In a flat price series (no change), RSI output after warmup handled deterministically."""
    length = 14
    prices = pd.Series([50.0] * 30)
    rsi = rsi_wilder(prices, length=length)

    # Initial warmup period
    assert rsi.iloc[:length].isna().all()
    # After warmup, gain and loss are 0. avg_loss is 0, filled to 100.0
    assert (rsi.iloc[length:] == 100.0).all()


def test_rsi_wilder_short_series():
    """Series shorter than length parameter should yield all NaNs."""
    length = 14
    prices = pd.Series([10.0, 11.0, 12.0, 11.5, 13.0])
    rsi = rsi_wilder(prices, length=length)

    assert len(rsi) == 5
    assert rsi.isna().all()


def test_rsi_wilder_custom_length():
    """Custom length parameter should change the warmup period and calculation horizon."""
    length = 5
    prices = pd.Series([10.0 + i for i in range(15)])
    rsi = rsi_wilder(prices, length=length)

    # Warmup is length items
    assert rsi.iloc[:length].isna().all()
    assert rsi.iloc[length] == 100.0
    assert (rsi.iloc[length:] == 100.0).all()


def test_rsi_wilder_known_calculation():
    """Verify RSI calculation accuracy against standard Wilder's RSI calculation steps."""
    # Price sequence with alternating up/down movements
    close_prices = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
        46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41
    ]
    series = pd.Series(close_prices)
    length = 14
    rsi = rsi_wilder(series, length=length)

    # Index 0 to 13 are NaN (14 points total)
    assert rsi.iloc[:14].isna().all()
    # Index 14 (15th price): check that RSI is a valid float between 0 and 100
    rsi_14 = rsi.iloc[14]
    assert pd.notna(rsi_14)
    assert 0.0 <= rsi_14 <= 100.0

    # Verify manual calculation for 14-period Wilder RSI at index 14
    deltas = series.diff()
    gains = deltas.clip(lower=0.0)
    losses = (-deltas).clip(lower=0.0)

    # Wilder smooth using ewm(alpha=1/14, adjust=False, min_periods=14)
    avg_gain = gains.ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[14]
    avg_loss = losses.ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[14]
    rs = avg_gain / avg_loss
    expected_rsi = 100.0 - (100.0 / (1.0 + rs))

    assert rsi_14 == pytest.approx(expected_rsi, abs=1e-6)


def test_rsi_wilder_default_cfg_length():
    """Verify function defaults to CFG.RSI_LEN when length is omitted."""
    prices = pd.Series([10.0 + i for i in range(30)])
    rsi_default = rsi_wilder(prices)
    rsi_explicit = rsi_wilder(prices, length=CFG.RSI_LEN)

    pd.testing.assert_series_equal(rsi_default, rsi_explicit)


def test_rsi_wilder_handles_nan_input():
    """Verify that NaNs in close prices do not crash the function."""
    prices = pd.Series([10.0, 11.0, np.nan, 12.0, 13.0, 14.0] + [15.0 + i for i in range(20)])
    rsi = rsi_wilder(prices, length=14)

    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(prices)
