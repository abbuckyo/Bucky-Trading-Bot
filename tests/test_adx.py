"""
Unit tests for Wilder's Directional Movement Index (ADX, +DI, -DI).
Validates mathematical accuracy against golden reference, deterministic extremes,
edge cases / data anomalies, and output integrity bounds.
"""
import numpy as np
import pandas as pd
import pytest
from bot import adx_wilder as bot_adx_wilder, CFG
from backtest_core import adx_wilder as core_adx_wilder


@pytest.fixture(params=[bot_adx_wilder, core_adx_wilder], ids=["bot_adx", "core_adx"])
def adx_func(request):
    """Parametrize over both bot.py and backtest_core.py implementations to ensure parity."""
    return request.param


# ==============================================================================
# 1. Mathematical Accuracy (Golden Reference Test)
# ==============================================================================

def test_adx_wilder_golden_reference(adx_func):
    """
    Test adx_wilder against an exact, deterministic step-by-step synthetic dataset
    with length=3 to trace Wilder's smoothing logic:
    i=0: TR=2, +DM=0, -DM=0 -> TR_s=NaN, +DI=NaN, -DI=NaN, DX=NaN, ADX=NaN
    i=1: TR=3, +DM=2, -DM=0 -> TR_s=NaN, +DI=NaN, -DI=NaN, DX=NaN, ADX=NaN
    i=2: TR=3, +DM=2, -DM=0 -> TR_s=2.5556, +DI=43.4783, -DI=0.0000, DX=100.0, ADX=NaN
    i=3: TR=3, +DM=0, -DM=1 -> TR_s=2.7037, +DI=27.3973, -DI=12.3288, DX=37.9310, ADX=NaN
    i=4: TR=4, +DM=2, -DM=0 -> TR_s=3.1358, +DI=37.0079, -DI=7.0866, DX=67.8571, ADX=75.4926
    i=5: TR=4, +DM=2, -DM=0 -> TR_s=3.4239, +DI=42.0673, -DI=4.3269, DX=81.3472, ADX=77.4441
    """
    df = pd.DataFrame({
        "high":  [10.0, 12.0, 14.0, 13.0, 15.0, 17.0],
        "low":   [ 8.0,  9.0, 11.0, 10.0, 12.0, 13.0],
        "close": [ 9.0, 11.0, 12.0, 11.0, 14.0, 16.0],
    })

    adx, pdi, mdi = adx_func(df, length=3)

    # First 2 rows: DI is NaN (min_periods = length = 3)
    assert pdi.iloc[:2].isna().all()
    assert mdi.iloc[:2].isna().all()

    # First 4 rows: ADX is NaN (min_periods for DX = 3, then min_periods for ADX smoothing = 3 -> row 4)
    assert adx.iloc[:4].isna().all()

    # Numerical assertions at row 2
    assert pytest.approx(pdi.iloc[2], rel=1e-4) == 43.4783
    assert pytest.approx(mdi.iloc[2], abs=1e-4) == 0.0

    # Numerical assertions at row 3
    assert pytest.approx(pdi.iloc[3], rel=1e-4) == 27.3973
    assert pytest.approx(mdi.iloc[3], rel=1e-4) == 12.3288

    # Numerical assertions at row 4
    assert pytest.approx(pdi.iloc[4], rel=1e-4) == 37.0079
    assert pytest.approx(mdi.iloc[4], rel=1e-4) == 7.0866
    assert pytest.approx(adx.iloc[4], rel=1e-4) == 75.4926

    # Numerical assertions at row 5
    assert pytest.approx(pdi.iloc[5], rel=1e-4) == 42.0673
    assert pytest.approx(mdi.iloc[5], rel=1e-4) == 4.3269
    assert pytest.approx(adx.iloc[5], rel=1e-4) == 77.4441


# ==============================================================================
# 2. Deterministic Extremes
# ==============================================================================

def test_adx_wilder_monotonic_uptrend(adx_func):
    """
    In a sustained monotonic uptrend:
    - +DI should dominate significantly (+DI >> -DI)
    - -DI should be zero or negligible
    - ADX should rise well above 50 (in strong trend regime, reaches up to 100)
    """
    periods = 60
    df_up = pd.DataFrame({
        "high":  [10.0 + i * 2.0 + 1.0 for i in range(periods)],
        "low":   [10.0 + i * 2.0 - 0.5 for i in range(periods)],
        "close": [10.0 + i * 2.0 + 0.8 for i in range(periods)],
    })

    adx, pdi, mdi = adx_func(df_up, length=14)

    # Tail of monotonic uptrend
    tail_pdi = pdi.dropna().iloc[-10:]
    tail_mdi = mdi.dropna().iloc[-10:]
    tail_adx = adx.dropna().iloc[-10:]

    assert (tail_pdi > 70.0).all()
    assert (tail_mdi == 0.0).all()
    assert (tail_pdi > tail_mdi * 10).all()
    assert (tail_adx > 50.0).all()


def test_adx_wilder_monotonic_downtrend(adx_func):
    """
    In a sustained monotonic downtrend:
    - -DI should dominate significantly (-DI >> +DI)
    - +DI should be zero or negligible
    - ADX should rise well above 50
    """
    periods = 60
    df_down = pd.DataFrame({
        "high":  [200.0 - i * 2.0 + 0.5 for i in range(periods)],
        "low":   [200.0 - i * 2.0 - 1.0 for i in range(periods)],
        "close": [200.0 - i * 2.0 - 0.8 for i in range(periods)],
    })

    adx, pdi, mdi = adx_func(df_down, length=14)

    tail_pdi = pdi.dropna().iloc[-10:]
    tail_mdi = mdi.dropna().iloc[-10:]
    tail_adx = adx.dropna().iloc[-10:]

    assert (tail_pdi == 0.0).all()
    assert (tail_mdi > 70.0).all()
    assert (tail_mdi > tail_pdi * 10).all()
    assert (tail_adx > 50.0).all()


def test_adx_wilder_flat_market_zero_volatility(adx_func):
    """
    In a completely flat market (identical OHLC across all bars):
    - True Range is zero.
    - +DM and -DM are zero.
    - Zero division must be handled cleanly without throwing ZeroDivisionError or unhandled exceptions.
    - Resulting values should safely be NaN.
    """
    df_flat = pd.DataFrame({
        "high":  [100.0] * 40,
        "low":   [100.0] * 40,
        "close": [100.0] * 40,
    })

    adx, pdi, mdi = adx_func(df_flat, length=14)

    assert isinstance(adx, pd.Series)
    assert isinstance(pdi, pd.Series)
    assert isinstance(mdi, pd.Series)
    assert len(adx) == 40
    # No crashes, all outputs gracefully become NaN
    assert adx.isna().all()
    assert pdi.isna().all()
    assert mdi.isna().all()


# ==============================================================================
# 3. Edge Cases & Data Anomalies
# ==============================================================================

def test_adx_wilder_insufficient_data_length(adx_func):
    """
    Dataframe with length < 2 * period (e.g., length=20 with period=14):
    - Must not crash with IndexError or out-of-bounds error.
    - Should return Series with the exact same length as input.
    - ADX requires 2 * period - 1 rows to produce first smoothed value,
      so for length < 2 * period, ADX should be all NaNs.
    """
    length = 14
    df_short = pd.DataFrame({
        "high":  [10.0 + i for i in range(20)],
        "low":   [ 9.0 + i for i in range(20)],
        "close": [ 9.5 + i for i in range(20)],
    })

    adx, pdi, mdi = adx_func(df_short, length=length)

    assert len(adx) == 20
    assert len(pdi) == 20
    assert len(mdi) == 20
    assert adx.isna().all()
    # +DI starts producing values after warmup (row index >= length - 1)
    assert pdi.iloc[length - 1:].notna().all()


def test_adx_wilder_very_short_dataframe(adx_func):
    """DataFrame shorter than period itself (e.g. 5 rows for period 14)."""
    length = 14
    df_short = pd.DataFrame({
        "high":  [10.0 + i for i in range(5)],
        "low":   [ 9.0 + i for i in range(5)],
        "close": [ 9.5 + i for i in range(5)],
    })

    adx, pdi, mdi = adx_func(df_short, length=length)

    assert len(adx) == 5
    assert adx.isna().all()
    assert pdi.isna().all()
    assert mdi.isna().all()


def test_adx_wilder_empty_dataframe(adx_func):
    """Empty DataFrame should return empty Series without crashing."""
    df_empty = pd.DataFrame(columns=["high", "low", "close"], dtype=float)

    adx, pdi, mdi = adx_func(df_empty, length=14)

    assert adx.empty
    assert pdi.empty
    assert mdi.empty
    assert isinstance(adx, pd.Series)
    assert isinstance(pdi, pd.Series)
    assert isinstance(mdi, pd.Series)


def test_adx_wilder_high_intraday_gap(adx_func):
    """
    High gap bar tests that True Range accounts for |High - C_prev| and |Low - C_prev|:
    - Bar 0: High=105, Low=95, Close=100
    - Bar 1: Huge gap up: High=160, Low=150, Close=155.
      TR = max(160-150, |160-100|, |150-100|) = 60.0.
      +DM = 160 - 105 = 55.0.
    """
    df = pd.DataFrame({
        "high":  [105.0, 160.0] + [160.0 + i for i in range(1, 15)],
        "low":   [ 95.0, 150.0] + [150.0 + i for i in range(1, 15)],
        "close": [100.0, 155.0] + [155.0 + i for i in range(1, 15)],
    })

    adx, pdi, mdi = adx_func(df, length=5)

    assert len(adx) == len(df)
    # The gap should propagate into strong +DI
    assert pdi.dropna().iloc[0] > 0.0
    assert (pdi.dropna() > mdi.dropna()).all()


# ==============================================================================
# 4. Output Integrity & Range Bounds
# ==============================================================================

def test_adx_wilder_output_integrity_and_bounds(adx_func):
    """
    Verify on stochastic random walk market data that:
    - Returns 3 pandas Series (adx, plus_di, minus_di)
    - Output index matches input DataFrame index exactly
    - All non-NaN values of ADX, +DI, and -DI are strictly bounded within [0.0, 100.0]
    """
    np.random.seed(1337)
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    base = 100.0 + np.cumsum(np.random.normal(0, 1.5, n))
    high = base + np.random.uniform(0.5, 3.0, n)
    low = base - np.random.uniform(0.5, 3.0, n)
    close = base + np.random.uniform(-0.4, 0.4, n)

    df = pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)

    adx, pdi, mdi = adx_func(df, length=CFG.ADX_LEN)

    assert isinstance(adx, pd.Series)
    assert isinstance(pdi, pd.Series)
    assert isinstance(mdi, pd.Series)

    # Index preservation
    assert (adx.index == df.index).all()
    assert (pdi.index == df.index).all()
    assert (mdi.index == df.index).all()

    # Bounded within [0, 100]
    valid_adx = adx.dropna()
    valid_pdi = pdi.dropna()
    valid_mdi = mdi.dropna()

    assert len(valid_adx) > 0
    assert (valid_adx >= 0.0).all() and (valid_adx <= 100.0).all()
    assert (valid_pdi >= 0.0).all() and (valid_pdi <= 100.0).all()
    assert (valid_mdi >= 0.0).all() and (valid_mdi <= 100.0).all()


def test_adx_wilder_bot_and_core_parity():
    """Verify exact parity between bot.py and backtest_core.py adx_wilder."""
    np.random.seed(42)
    n = 80
    close = 50.0 + np.cumsum(np.random.randn(n))
    high = close + np.random.uniform(0.2, 1.0, n)
    low = close - np.random.uniform(0.2, 1.0, n)
    df = pd.DataFrame({"high": high, "low": low, "close": close})

    adx_bot, pdi_bot, mdi_bot = bot_adx_wilder(df, length=14)
    adx_core, pdi_core, mdi_core = core_adx_wilder(df, length=14)

    pd.testing.assert_series_equal(adx_bot, adx_core)
    pd.testing.assert_series_equal(pdi_bot, pdi_core)
    pd.testing.assert_series_equal(mdi_bot, mdi_core)
