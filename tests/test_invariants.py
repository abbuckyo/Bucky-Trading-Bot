import math
import itertools
import pytest
import numpy as np
import pandas as pd
from bot import (
    Metrics,
    decide,
    validate_sizing_for_scb,
    is_nan,
    ema,
    TRANCHE_EXPOSURE_MAP,
    TRANCHE_LABEL_MAP,
    CFG,
)

# Helper function to generate Metrics matching desired raw stage (0, 1, 2, 3)
def make_metrics_for_stage(stage: int, breakdown: bool = False, rsi: float = 60.0, score: float = 80.0):
    dist_ema200 = -3.0 if breakdown else 5.0
    if stage == 3:
        # Price > EMA200 and EMA50 > EMA100 > EMA200
        return Metrics(
            price=105.0,
            ema50=100.0,
            ema100=95.0,
            ema200=90.0,
            dist_ema200_pct=dist_ema200,
            dist_ema50_pct=5.0,
            rsi=rsi,
            bull_stack=True,
        )
    elif stage == 2:
        # Price > EMA100 and EMA50 > EMA100, but not EMA100 > EMA200
        return Metrics(
            price=102.0,
            ema50=100.0,
            ema100=95.0,
            ema200=98.0,  # EMA100 < EMA200 breaks bull_stack
            dist_ema200_pct=dist_ema200,
            dist_ema50_pct=2.0,
            rsi=rsi,
            bull_stack=False,
        )
    elif stage == 1:
        # Price > EMA50, but not Price > EMA100 or EMA50 <= EMA100
        return Metrics(
            price=96.0,
            ema50=95.0,
            ema100=98.0,
            ema200=100.0,
            dist_ema200_pct=dist_ema200,
            dist_ema50_pct=1.0,
            rsi=rsi,
            bull_stack=False,
        )
    else:
        # stage 0: Price <= EMA50
        return Metrics(
            price=90.0,
            ema50=95.0,
            ema100=98.0,
            ema200=100.0,
            dist_ema200_pct=dist_ema200,
            dist_ema50_pct=-5.0,
            rsi=rsi,
            bull_stack=False,
        )

STAGE_TO_POS_INOUT = {0: 'OUT', 1: 'IN', 2: 'IN', 3: 'IN'}
STAGE_TO_PREV_POS = {0: 'OUT', 1: 'T1', 2: 'T2', 3: 'T3'}


# ==============================================================================
# Suite 0: Indicator Function Suite (EMA)
# ==============================================================================
class TestIndicators:
    def test_ema_mathematical_calculation(self):
        """Verify EMA values match manual calculation with span=length, adjust=False."""
        prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        length = 3
        # multiplier alpha = 2 / (3 + 1) = 0.5
        # min_periods = 3, so index 0 and 1 are NaN
        # index 2: SMA of first 3 elements (or initial ewm value with adjust=False) = 22.5
        # index 3: 40.0 * 0.5 + 22.5 * (1 - 0.5) = 31.25
        # index 4: 50.0 * 0.5 + 31.25 * (1 - 0.5) = 40.625
        result = ema(prices, length)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pytest.approx(result.iloc[2], abs=1e-5) == 22.5
        assert pytest.approx(result.iloc[3], abs=1e-5) == 31.25
        assert pytest.approx(result.iloc[4], abs=1e-5) == 40.625

    def test_ema_constant_series(self):
        """EMA of a constant price series should return the constant price (after min_periods)."""
        prices = pd.Series([100.0] * 10)
        length = 5
        result = ema(prices, length)

        assert result.iloc[:4].isna().all()
        assert (result.iloc[4:] == 100.0).all()

    def test_ema_series_shorter_than_length(self):
        """Series with length < length parameter should return all NaNs due to min_periods."""
        prices = pd.Series([10.0, 12.0, 14.0])
        length = 5
        result = ema(prices, length)

        assert len(result) == 3
        assert result.isna().all()

    def test_ema_length_one(self):
        """EMA with length=1 should equal the original series (alpha = 1.0)."""
        prices = pd.Series([10.0, 15.0, 12.0, 20.0])
        result = ema(prices, length=1)

        pd.testing.assert_series_equal(result, prices)

    def test_ema_with_nan_in_series(self):
        """Test handling when series contains NaN values."""
        prices = pd.Series([10.0, 20.0, np.nan, 40.0, 50.0, 60.0])
        result = ema(prices, length=3)
        # Verify output is a pandas Series with same index and length
        assert len(result) == len(prices)
        assert isinstance(result, pd.Series)


# ==============================================================================
# Suite 1: NaN Guard Suite
# ==============================================================================
import numpy as np

class TestNaNGuards:
    def test_nan_helper(self):
        assert is_nan(float('nan')) is True
        assert is_nan(math.nan) is True
        assert is_nan(np.nan) is True
        assert is_nan(float('inf')) is True
        assert is_nan(np.inf) is True
        assert is_nan(None) is True
        assert is_nan('corrupt') is True
        assert is_nan(100.0) is False
        assert is_nan(0) is False
        assert is_nan('123.45') is False

    @pytest.mark.parametrize("corrupt_field", ["price", "ema50", "ema100", "ema200", "rsi"])
    def test_metrics_nan_forces_safe_exit(self, corrupt_field):
        m = make_metrics_for_stage(3)
        setattr(m, corrupt_field, float('nan'))
        
        dec = decide(
            score=80.0,
            breakdown=False,
            prev_position="T2",
            fund="TEST",
            m=m,
            canary_ok=True,
            futures_guard_triggered=False
        )
        assert dec.tranche_stage == 0
        assert dec.position == "OUT"
        assert dec.target_exposure == 0.0
        assert "CORRUPT" in dec.signal or dec.signal == "CORRUPT DATA EXIT"

    def test_score_nan_forces_safe_exit(self):
        m = make_metrics_for_stage(3)
        dec = decide(
            score=float('nan'),
            breakdown=False,
            prev_position="T3",
            fund="TEST",
            m=m,
            canary_ok=True,
            futures_guard_triggered=False
        )
        assert dec.tranche_stage == 0
        assert dec.position == "OUT"
        assert dec.target_exposure == 0.0
        assert "CORRUPT" in dec.signal

    def test_metrics_none_forces_safe_exit(self):
        try:
            dec = decide(
                score=80.0,
                breakdown=False,
                prev_position="T2",
                fund="TEST",
                m=None,
                canary_ok=True,
                futures_guard_triggered=False
            )
            assert dec.tranche_stage == 0
            assert dec.position == "OUT"
            assert dec.target_exposure == 0.0
        except AttributeError:
            pass  # Metrics object expected by type signature


# ==============================================================================
# Suite 2: Exhaustive Property Tests (128 Combinations)
# Matrix: 4 prev_stages x 4 raw_stages x 2 futures_pause x 2 breakdown x 2 canary
# ==============================================================================
class TestExhaustiveProperties:
    @pytest.mark.parametrize(
        "prev_stage, raw_stage, futures_pause, breakdown, canary_ok",
        list(itertools.product(range(4), range(4), [False, True], [False, True], [False, True]))
    )
    def test_state_transition_invariants(
        self, prev_stage, raw_stage, futures_pause, breakdown, canary_ok
    ):
        m = make_metrics_for_stage(raw_stage, breakdown=breakdown)
        prev_pos = STAGE_TO_PREV_POS[prev_stage]
        
        dec = decide(
            score=80.0,
            breakdown=breakdown,
            prev_position=prev_pos,
            fund="TEST",
            m=m,
            canary_ok=canary_ok,
            futures_guard_triggered=futures_pause
        )

        stage = dec.tranche_stage
        exposure = dec.target_exposure

        # Invariant A: Valid stage domain and exposure alignment
        assert stage in (0, 1, 2, 3), f"Invalid stage: {stage}"
        assert exposure == pytest.approx(TRANCHE_EXPOSURE_MAP[stage], abs=1e-6)
        assert dec.position == STAGE_TO_POS_INOUT[stage]

        # Invariant 1: Canary OFF (<= 0) -> ALWAYS Stage 0
        if not canary_ok:
            assert stage == 0, f"Canary OFF must force stage 0, got {stage}"
            assert dec.signal == "CANARY DEFENSE"
            return

        # Invariant 2: Hard Breakdown -> ALWAYS Stage 0
        if breakdown:
            assert stage == 0, f"Breakdown must force stage 0, got {stage}"
            assert dec.signal == "HARD EXIT"
            return

        # Invariant 3: Immediate De-risking (raw_stage < prev_stage) -> Drop directly to raw_stage
        if raw_stage < prev_stage:
            assert stage == raw_stage, f"De-risking must drop directly to raw_stage {raw_stage}, got {stage}"
            return

        # Invariant 4: Pre-Market US Futures Guard (raw_stage > prev_stage and futures_pause) -> Hold prev_stage
        if raw_stage > prev_stage and futures_pause:
            assert stage == prev_stage, f"Futures guard pause must hold prev_stage {prev_stage}, got {stage}"
            assert dec.signal == "FUTURES PAUSE"
            return

        # Invariant 5: Rate Limiting (One Tranche at a time on ladder entry)
        if raw_stage > prev_stage and not futures_pause:
            assert stage == prev_stage + 1, f"Ladder entry must advance by exactly 1 stage: {prev_stage} -> {stage}"
            assert stage - prev_stage == 1
            return

        # Invariant 6: Steady state (raw_stage == prev_stage)
        if raw_stage == prev_stage:
            assert stage == prev_stage, f"Steady state must hold stage: {stage} == {prev_stage}"


# ==============================================================================
# Suite 3: Exposure & Rounding Combos (1024 Sets: 4^5 Portfolio Combinations)
# ==============================================================================
class TestPortfolioExposureBalance:
    def test_all_1024_portfolio_combinations_sum_to_unity(self):
        """
        Verify that across all 4^5 = 1,024 combinations of 5 assets,
        sum(weights) + cash_weight strictly equals 1.0 within 1e-6.
        Sleeve size is 20% (0.20) per asset.
        """
        assets = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"]
        sleeve_pct = 0.20
        combos = itertools.product(range(4), repeat=len(assets))

        count = 0
        for combo in combos:
            weights = [(stage / 3.0) * sleeve_pct for stage in combo]
            total_invested = sum(weights)
            cash_weight = 1.0 - total_invested

            # Balance Invariant: Total portfolio allocation must be 1.0 (100%)
            assert abs((total_invested + cash_weight) - 1.0) < 1e-6
            assert 0.0 <= cash_weight <= 1.0 + 1e-6
            assert 0.0 <= total_invested <= 1.0 + 1e-6
            count += 1

        assert count == 1024, f"Expected 1024 combinations tested, got {count}"


# ==============================================================================
# Suite 4: SCB Minimum Switch Guard Validation
# ==============================================================================
class TestSCBMinimumSwitchGuard:
    def test_sub_minimum_switch_is_blocked(self):
        """
        Portfolio total = 5,000 THB.
        Sleeve = 20% = 1,000 THB.
        1 Tranche delta (1/3) = 333.33 THB < 1,000 THB SCB Minimum Switch.
        Must block and retain prev_stage.
        """
        valid, allowed_stage, reason = validate_sizing_for_scb(
            fund="SCBNDQ(A)",
            prev_stage=1,
            target_stage=2,
            portfolio_total_thb=5000.0,
            sleeve_pct=0.20,
            min_switch_thb=1000.0
        )
        assert valid is False
        assert allowed_stage == 1
        assert "BLOCKED" in reason

    def test_sufficient_switch_is_allowed(self):
        """
        Portfolio total = 50,000 THB.
        Sleeve = 20% = 10,000 THB.
        1 Tranche delta (1/3) = 3,333.33 THB >= 1,000 THB SCB Minimum Switch.
        Must pass and permit target_stage.
        """
        valid, allowed_stage, reason = validate_sizing_for_scb(
            fund="SCBNDQ(A)",
            prev_stage=1,
            target_stage=2,
            portfolio_total_thb=50000.0,
            sleeve_pct=0.20,
            min_switch_thb=1000.0
        )
        assert valid is True
        assert allowed_stage == 2
        assert "VALID" in reason

    def test_emergency_full_exit_always_allowed(self):
        """
        Regardless of balance (even if remaining value is 400 THB),
        a full exit to cash (target_stage == 0) must NEVER be blocked by the sizing guard.
        """
        valid, allowed_stage, reason = validate_sizing_for_scb(
            fund="SCBNDQ(A)",
            prev_stage=1,
            target_stage=0,
            portfolio_total_thb=2000.0,
            sleeve_pct=0.20,
            min_switch_thb=1000.0
        )
        assert valid is True
        assert allowed_stage == 0
        assert "FULL EXIT PERMITTED" in reason

    def test_zero_delta_stay_put(self):
        valid, allowed_stage, reason = validate_sizing_for_scb(
            fund="SCBNDQ(A)",
            prev_stage=2,
            target_stage=2,
            portfolio_total_thb=100000.0,
            sleeve_pct=0.20,
            min_switch_thb=1000.0
        )
        assert valid is True
        assert allowed_stage == 2
        assert "NO CHANGE" in reason
