import numpy as np
import pandas as pd
import pytest
from bot import hist_volatility, CFG


def test_hist_volatility_constant_prices():
    """Constant prices should result in zero volatility after initial window periods."""
    dates = pd.date_range("2025-01-01", periods=30)
    close = pd.Series([100.0] * 30, index=dates, name="close")

    vol = hist_volatility(close, window=CFG.HV_WINDOW)

    # First 20 values (index 0 to 19) should be NaN due to shift and rolling window
    assert vol.iloc[:CFG.HV_WINDOW].isna().all()
    # Values from index 20 onwards should be exactly 0.0
    assert (vol.iloc[CFG.HV_WINDOW:] == 0.0).all()


def test_hist_volatility_constant_returns():
    """Exponential price growth (constant log returns) should yield 0.0 volatility."""
    dates = pd.date_range("2025-01-01", periods=30)
    prices = 100.0 * (1.02 ** np.arange(30))
    close = pd.Series(prices, index=dates, name="close")

    vol = hist_volatility(close, window=20)

    assert vol.iloc[:20].isna().all()
    # Rolling std of constant numbers is 0.0
    np.testing.assert_allclose(vol.iloc[20:].values, 0.0, atol=1e-12)


def test_hist_volatility_known_calculation():
    """Verify exact formula calculation against numpy / pandas manual steps."""
    prices = [100.0, 105.0, 102.0, 108.0, 104.0, 110.0]
    dates = pd.date_range("2025-01-01", periods=len(prices))
    close = pd.Series(prices, index=dates, name="close")

    window = 3
    vol = hist_volatility(close, window=window)

    # Manual step-by-step math
    log_ret = np.log(close / close.shift(1))
    expected_rolling_std = log_ret.rolling(window=window).std()
    expected_vol = expected_rolling_std * np.sqrt(CFG.TRADING_DAYS) * 100.0

    pd.testing.assert_series_equal(vol, expected_vol)

    # Check specific index manual calculation (index 3, window=3)
    # log_ret values at index 1, 2, 3:
    r1 = np.log(105.0 / 100.0)
    r2 = np.log(102.0 / 105.0)
    r3 = np.log(108.0 / 102.0)
    manual_std = np.std([r1, r2, r3], ddof=1)
    manual_vol = manual_std * np.sqrt(252) * 100.0

    assert vol.iloc[3] == pytest.approx(manual_vol, rel=1e-7)


def test_hist_volatility_custom_window():
    """Verify that a custom window parameter is respected."""
    dates = pd.date_range("2025-01-01", periods=15)
    close = pd.Series([100.0, 102.0, 101.0, 104.0, 103.0, 105.0, 106.0, 104.0, 107.0, 108.0, 106.0, 109.0, 110.0, 108.0, 111.0], index=dates)

    custom_window = 5
    vol = hist_volatility(close, window=custom_window)

    # Initial custom_window (5) entries should be NaN
    assert vol.iloc[:custom_window].isna().all()
    # Entry at index 5 onwards should be valid float numbers
    assert vol.iloc[custom_window:].notna().all()


def test_hist_volatility_insufficient_data():
    """If input series has fewer elements than window size, output should be all NaN."""
    dates = pd.date_range("2025-01-01", periods=10)
    close = pd.Series([100.0 + i for i in range(10)], index=dates)

    vol = hist_volatility(close, window=20)

    assert vol.isna().all()
    assert len(vol) == 10


def test_hist_volatility_series_properties():
    """Verify that output retains index and name properties of the input Series."""
    dates = pd.date_range("2025-01-01", periods=30)
    close = pd.Series([100.0 + (i % 5) for i in range(30)], index=dates, name="SCBNDQ(E)")

    vol = hist_volatility(close)

    assert isinstance(vol, pd.Series)
    assert vol.index.equals(close.index)
    assert vol.name == close.name
