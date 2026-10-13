import numpy as np
import pandas as pd
import pytest

from bot import percentile_rank, CFG


def test_percentile_rank_short_series():
    """Series with fewer than 30 valid items should return 50.0."""
    series = pd.Series(range(20))
    assert percentile_rank(series) == 50.0


def test_percentile_rank_with_nans():
    """NaN values should be dropped prior to checking minimum length and calculating percentile."""
    # 40 total elements, but 15 are NaN -> 25 valid elements (< 30)
    data = [np.nan] * 15 + list(range(25))
    series = pd.Series(data)
    assert percentile_rank(series) == 50.0

    # 50 total elements, 10 are NaN -> 40 valid elements (>= 30)
    data = [np.nan] * 10 + list(range(1, 41))
    series = pd.Series(data)
    # latest element is 40, which is larger than 39 items out of 40
    assert percentile_rank(series) == (39 / 40) * 100.0


def test_percentile_rank_min_value():
    """Latest element is the minimum value in tail -> 0 items strictly lower -> 0.0."""
    data = list(range(100, 0, -1))  # 100 down to 1 (latest is 1)
    series = pd.Series(data)
    assert percentile_rank(series) == 0.0


def test_percentile_rank_max_value():
    """Latest element is the maximum value in 100 items -> 99 items strictly lower -> 99.0."""
    data = list(range(1, 101))  # 1 to 100 (latest is 100)
    series = pd.Series(data)
    assert percentile_rank(series) == 99.0


def test_percentile_rank_median_value():
    """Exact percentile return when half of the tail is strictly less than latest."""
    # 50 elements with value 10, 1 element with value 20, 49 elements with value 30
    # Total 100 elements. Latest is 20. Exactly 50 elements are < 20.
    data = [10.0] * 50 + [20.0]
    series = pd.Series(data)
    # 51 elements total, latest is 20. 50 elements < 20.
    assert percentile_rank(series) == (50 / 51) * 100.0


def test_percentile_rank_custom_lookback():
    """Lookback parameter limits the tail size evaluated."""
    # 100 elements: first 50 are 1..50, next 50 are 100..149, latest is 150
    # With lookback=30, only last 30 elements are considered.
    data = list(range(1, 151))  # 1 to 150
    series = pd.Series(data)

    # Tail of 30: 121..150. Latest is 150. 29 items < 150 out of 30 items.
    assert percentile_rank(series, lookback=30) == (29 / 30) * 100.0


def test_percentile_rank_all_equal():
    """When all items in series are equal, 0 items are strictly lower -> 0.0."""
    data = [5.0] * 50
    series = pd.Series(data)
    assert percentile_rank(series) == 0.0
