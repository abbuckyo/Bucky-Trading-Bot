import numpy as np
import pandas as pd
import pytest

from bot import CFG, atr, true_range, wilder_smooth


def create_sample_ohlc(num_bars: int = 30) -> pd.DataFrame:
    """Helper to generate a predictable OHLC DataFrame."""
    np.random.seed(42)
    base_price = 100.0
    dates = pd.date_range(start="2024-01-01", periods=num_bars, freq="D")

    closes = base_price + np.cumsum(np.random.randn(num_bars))
    highs = closes + np.abs(np.random.randn(num_bars)) + 0.5
    lows = closes - np.abs(np.random.randn(num_bars)) - 0.5
    opens = (highs + lows) / 2.0

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        },
        index=dates,
    )


class TestATRIndicator:
    def test_atr_basic_calculation(self):
        df = create_sample_ohlc(30)
        length = 14
        result = atr(df, length=length)

        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

        # min_periods=length means first (length - 1) items should be NaN
        assert result.iloc[: length - 1].isna().all()
        assert not result.iloc[length - 1 :].isna().any()

        # Independently verify calculation against true_range + wilder_smooth
        tr = true_range(df)
        expected = wilder_smooth(tr, length)
        pd.testing.assert_series_equal(result, expected)

    def test_atr_default_length(self):
        df = create_sample_ohlc(30)
        result_default = atr(df)
        result_explicit = atr(df, length=CFG.ATR_LEN)

        pd.testing.assert_series_equal(result_default, result_explicit)

    def test_atr_custom_length(self):
        df = create_sample_ohlc(20)
        length = 5
        result = atr(df, length=length)

        # First 4 (length - 1) should be NaN
        assert result.iloc[:4].isna().all()
        assert not result.iloc[4:].isna().any()

    def test_atr_zero_volatility(self):
        """Flat price series should result in ATR of 0.0 once min_periods is reached."""
        num_bars = 20
        df = pd.DataFrame(
            {
                "open": [100.0] * num_bars,
                "high": [100.0] * num_bars,
                "low": [100.0] * num_bars,
                "close": [100.0] * num_bars,
            }
        )
        length = 10
        result = atr(df, length=length)

        # Values from index 9 onwards should be 0.0
        assert (result.iloc[length - 1 :] == 0.0).all()

    def test_atr_gap_up_down(self):
        """Test ATR response to large gaps between consecutive bars."""
        # Bar 0: H=105, L=95, C=100
        # Bar 1: Gap up to H=120, L=115, C=118 -> TR = max(120-115, |120-100|, |115-100|) = 20
        # Bar 2: Gap down to H=90, L=85, C=88 -> TR = max(90-85, |90-118|, |85-118|) = 33
        data = {
            "high": [105.0, 120.0, 90.0] + [100.0] * 10,
            "low": [95.0, 115.0, 85.0] + [90.0] * 10,
            "close": [100.0, 118.0, 88.0] + [95.0] * 10,
        }
        df = pd.DataFrame(data)
        length = 3
        result = atr(df, length=length)

        tr = true_range(df)
        # Bar 0 TR: H - L = 10
        assert tr.iloc[0] == 10.0
        # Bar 1 TR: max(5, 120-100=20, 115-100=15) = 20
        assert tr.iloc[1] == 20.0
        # Bar 2 TR: max(5, |90-118|=28, |85-118|=33) = 33
        assert tr.iloc[2] == 33.0

        # Check ATR at bar 2 (first non-NaN bar for length=3)
        assert not np.isnan(result.iloc[2])

    def test_atr_insufficient_data(self):
        """If data length < ATR length, all returned ATR values should be NaN."""
        df = create_sample_ohlc(10)
        length = 14
        result = atr(df, length=length)

        assert result.isna().all()
