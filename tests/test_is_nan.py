import math
import numpy as np
import pandas as pd
import pytest

from bot import is_nan as bot_is_nan
from alphashield.data.provider import is_nan as provider_is_nan
from alphashield.strategy.data_integrity import is_nan as integrity_is_nan

IS_NAN_FUNCTIONS = [
    ("bot.is_nan", bot_is_nan),
    ("provider.is_nan", provider_is_nan),
    ("integrity.is_nan", integrity_is_nan),
]


@pytest.mark.parametrize("func_name, is_nan_func", IS_NAN_FUNCTIONS)
class TestIsNanHelper:
    def test_none_returns_true(self, func_name, is_nan_func):
        assert is_nan_func(None) is True

    def test_pandas_na_returns_true(self, func_name, is_nan_func):
        assert is_nan_func(pd.NA) is True

    def test_math_nan_returns_true(self, func_name, is_nan_func):
        assert is_nan_func(math.nan) is True
        assert is_nan_func(float("nan")) is True

    def test_numpy_nan_returns_true(self, func_name, is_nan_func):
        assert is_nan_func(np.nan) is True
        assert is_nan_func(np.float64("nan")) is True

    def test_infinities_return_true(self, func_name, is_nan_func):
        assert is_nan_func(float("inf")) is True
        assert is_nan_func(float("-inf")) is True
        assert is_nan_func(np.inf) is True
        assert is_nan_func(-np.inf) is True

    def test_valid_integers_and_floats_return_false(self, func_name, is_nan_func):
        assert is_nan_func(0) is False
        assert is_nan_func(100) is False
        assert is_nan_func(-42) is False
        assert is_nan_func(0.0) is False
        assert is_nan_func(100.5) is False
        assert is_nan_func(-10.25) is False
        assert is_nan_func(np.int64(50)) is False
        assert is_nan_func(np.float64(12.34)) is False

    def test_booleans_return_false(self, func_name, is_nan_func):
        assert is_nan_func(True) is False
        assert is_nan_func(False) is False

    def test_numeric_strings_return_false(self, func_name, is_nan_func):
        assert is_nan_func("123.45") is False
        assert is_nan_func("0") is False
        assert is_nan_func("-10.5") is False

    def test_invalid_strings_and_objects_return_true(self, func_name, is_nan_func):
        assert is_nan_func("corrupt") is True
        assert is_nan_func("NaN") is True
        assert is_nan_func("abc") is True
        assert is_nan_func([1, 2, 3]) is True
        assert is_nan_func({"a": 1}) is True
        assert is_nan_func(object()) is True
