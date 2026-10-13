import pandas as pd
import pandas.testing as pdt
import pytest
from bot import true_range


def test_true_range_calculation_scenarios():
    """
    Test true_range across standard scenarios:
    - Row 0: Initial row (prev_close is NaN). TR = High - Low.
    - Row 1: Gap Up where |High - Prev_Close| is maximum.
    - Row 2: Gap Down where |Low - Prev_Close| is maximum.
    - Row 3: High-Low range dominant where High - Low is maximum.
    """
    df = pd.DataFrame({
        "high":  [105.0, 115.0, 90.0, 100.0],
        "low":   [95.0,  110.0, 80.0, 80.0],
        "close": [100.0, 112.0, 85.0, 90.0],
    })

    expected = pd.Series([10.0, 15.0, 32.0, 20.0])
    result = true_range(df)

    pdt.assert_series_equal(result, expected, check_names=False)


def test_true_range_single_row():
    """Single row DataFrame has no previous close, so TR equals High - Low."""
    df = pd.DataFrame({
        "high": [105.0],
        "low": [95.0],
        "close": [100.0],
    })

    expected = pd.Series([10.0])
    result = true_range(df)

    pdt.assert_series_equal(result, expected, check_names=False)


def test_true_range_empty_dataframe():
    """Empty DataFrame should return empty Series."""
    df = pd.DataFrame(columns=["high", "low", "close"], dtype=float)

    result = true_range(df)

    assert result.empty
    assert isinstance(result, pd.Series)
