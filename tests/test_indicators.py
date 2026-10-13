import numpy as np
import pandas as pd
import pytest
from bot import adx_wilder, CFG


def create_ohlc_df(dates, high, low, close, open_p=None):
    if open_p is None:
        open_p = (high + low) / 2.0
    return pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000,
        },
        index=dates,
    )


class TestADXWilder:
    def test_return_tuple_and_structure(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        np.random.seed(42)
        close = np.cumsum(np.random.randn(50)) + 100
        high = close + np.random.uniform(0.5, 2.0, size=50)
        low = close - np.random.uniform(0.5, 2.0, size=50)
        df = create_ohlc_df(dates, high, low, close)

        adx, plus_di, minus_di = adx_wilder(df, length=14)

        assert isinstance(adx, pd.Series)
        assert isinstance(plus_di, pd.Series)
        assert isinstance(minus_di, pd.Series)
        assert len(adx) == len(df)
        assert len(plus_di) == len(df)
        assert len(minus_di) == len(df)
        assert adx.index.equals(df.index)
        assert plus_di.index.equals(df.index)
        assert minus_di.index.equals(df.index)

    def test_strong_uptrend(self):
        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = np.linspace(100, 200, n)
        high = close + 1.0
        low = close - 1.0
        df = create_ohlc_df(dates, high, low, close)

        adx, plus_di, minus_di = adx_wilder(df, length=14)

        # Towards the tail of the uptrend:
        # plus_di should be dominant, minus_di should be near 0
        latest_plus = plus_di.iloc[-1]
        latest_minus = minus_di.iloc[-1]
        latest_adx = adx.iloc[-1]

        assert latest_plus > latest_minus
        assert latest_minus == pytest.approx(0.0, abs=1e-3)
        assert latest_plus > 50.0
        assert latest_adx > 50.0  # Strong trend

    def test_strong_downtrend(self):
        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = np.linspace(200, 100, n)
        high = close + 1.0
        low = close - 1.0
        df = create_ohlc_df(dates, high, low, close)

        adx, plus_di, minus_di = adx_wilder(df, length=14)

        latest_plus = plus_di.iloc[-1]
        latest_minus = minus_di.iloc[-1]
        latest_adx = adx.iloc[-1]

        assert latest_minus > latest_plus
        assert latest_plus == pytest.approx(0.0, abs=1e-3)
        assert latest_minus > 50.0
        assert latest_adx > 50.0  # Strong trend

    def test_flat_prices(self):
        n = 40
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        df = create_ohlc_df(dates, high, low, close)

        adx, plus_di, minus_di = adx_wilder(df, length=14)

        # Zero true range means plus_di and minus_di replaced 0 with NaN or 0
        # No division by zero exceptions should be raised
        assert isinstance(adx, pd.Series)
        assert isinstance(plus_di, pd.Series)
        assert isinstance(minus_di, pd.Series)

    def test_custom_length_parameter(self):
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = np.linspace(100, 150, n)
        high = close + 1.0
        low = close - 1.0
        df = create_ohlc_df(dates, high, low, close)

        adx7, plus7, minus7 = adx_wilder(df, length=7)
        adx14, plus14, minus14 = adx_wilder(df, length=14)

        # Shorter period length=7 should have fewer initial NaNs than length=14
        assert adx7.iloc[6:13].notna().any()
        assert adx14.iloc[6:13].isna().all()

    def test_short_dataframe(self):
        # DataFrame shorter than smoothing length
        n = 5
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = np.linspace(100, 105, n)
        high = close + 1.0
        low = close - 1.0
        df = create_ohlc_df(dates, high, low, close)

        adx, plus_di, minus_di = adx_wilder(df, length=14)

        assert len(adx) == 5
        assert adx.isna().all()
        assert plus_di.isna().all()
        assert minus_di.isna().all()
