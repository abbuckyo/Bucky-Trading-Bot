import numpy as np
import pandas as pd
import pytest
from bot import macd, ema, CFG


def test_macd_returns_tuple_of_series():
    """Verify macd returns a 3-tuple of Series with matching index and length."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    prices = pd.Series(np.linspace(100, 150, 50), index=dates)

    res = macd(prices)

    assert isinstance(res, tuple)
    assert len(res) == 3
    macd_line, signal_line, hist = res

    assert isinstance(macd_line, pd.Series)
    assert isinstance(signal_line, pd.Series)
    assert isinstance(hist, pd.Series)

    assert len(macd_line) == len(prices)
    assert len(signal_line) == len(prices)
    assert len(hist) == len(prices)

    pd.testing.assert_index_equal(macd_line.index, prices.index)
    pd.testing.assert_index_equal(signal_line.index, prices.index)
    pd.testing.assert_index_equal(hist.index, prices.index)


def test_macd_mathematical_identity():
    """Verify mathematical identities of MACD calculations."""
    prices = pd.Series([10.0 + i * 0.5 + (i % 3) for i in range(60)])

    macd_line, signal_line, hist = macd(prices)

    # Fast EMA (12) - Slow EMA (26)
    expected_fast = ema(prices, CFG.MACD_FAST)
    expected_slow = ema(prices, CFG.MACD_SLOW)
    expected_line = expected_fast - expected_slow

    pd.testing.assert_series_equal(macd_line, expected_line)

    # Hist = Line - Signal
    expected_hist = macd_line - signal_line
    pd.testing.assert_series_equal(hist, expected_hist)


def test_macd_constant_series():
    """Constant prices should result in zero MACD line, signal line, and histogram after min_periods."""
    prices = pd.Series([100.0] * 50)

    macd_line, signal_line, hist = macd(prices)

    # Fast=12, Slow=26, Signal=9. So after period 26 + 9 = 35 (index 34+), values should be 0.0
    valid_macd = macd_line.iloc[35:]
    valid_signal = signal_line.iloc[35:]
    valid_hist = hist.iloc[35:]

    np.testing.assert_allclose(valid_macd, 0.0, atol=1e-12)
    np.testing.assert_allclose(valid_signal, 0.0, atol=1e-12)
    np.testing.assert_allclose(valid_hist, 0.0, atol=1e-12)


def test_macd_uptrend_and_downtrend():
    """Test MACD behavior in strong uptrend vs strong downtrend."""
    # Strong uptrend: fast EMA exceeds slow EMA -> macd_line > 0
    uptrend_prices = pd.Series([100.0 + i * 2.0 for i in range(50)])
    macd_line_up, _, _ = macd(uptrend_prices)
    # Check valid values after slow EMA burn-in (26 periods)
    assert (macd_line_up.iloc[26:] > 0).all()

    # Strong downtrend: fast EMA falls below slow EMA -> macd_line < 0
    downtrend_prices = pd.Series([200.0 - i * 2.0 for i in range(50)])
    macd_line_down, _, _ = macd(downtrend_prices)
    assert (macd_line_down.iloc[26:] < 0).all()


def test_macd_short_series():
    """Series shorter than MACD_SLOW (26) should produce NaNs for macd, signal, and hist."""
    short_prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])

    macd_line, signal_line, hist = macd(short_prices)

    assert macd_line.isna().all()
    assert signal_line.isna().all()
    assert hist.isna().all()


def test_macd_custom_index():
    """Ensure non-default index (e.g. string labels) is preserved properly."""
    custom_index = [f"day_{i}" for i in range(40)]
    prices = pd.Series(range(100, 140), index=custom_index, dtype=float)

    macd_line, signal_line, hist = macd(prices)

    assert list(macd_line.index) == custom_index
    assert list(signal_line.index) == custom_index
    assert list(hist.index) == custom_index


def test_macd_nan_handling():
    """Ensure price series with NaNs does not crash and correctly propagates NaNs."""
    prices = pd.Series([100.0 + i for i in range(50)])
    prices.iloc[10] = np.nan

    macd_line, signal_line, hist = macd(prices)

    assert len(macd_line) == 50
    # NaNs should be present at/after index 10 until EMA recovers
    assert macd_line.iloc[10:].isna().any()
