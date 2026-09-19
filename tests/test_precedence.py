import pytest
from bot import Metrics, decide, TRANCHE_EXPOSURE_MAP, TRANCHE_LABEL_MAP

def make_metrics(price=100.0, ema50=95.0, ema100=90.0, ema200=85.0, dist_ema200_pct=5.0):
    return Metrics(
        price=price,
        ema50=ema50,
        ema100=ema100,
        ema200=ema200,
        dist_ema200_pct=dist_ema200_pct,
        dist_ema50_pct=((price / ema50 - 1.0) * 100.0) if ema50 else 0.0,
        rsi=60.0,
        bull_stack=(ema50 > ema100 > ema200)
    )

# ------------------------------------------------------------------------------
# 1. Canary OFF (<= 0) Precedence Tests (Overrides everything -> Atomic Reset to 0)
# ------------------------------------------------------------------------------
def test_case_01_canary_off_from_stage_3():
    m = make_metrics(price=100, ema50=95, ema100=90, ema200=85)
    dec = decide(score=85, breakdown=False, prev_position='T3', fund='TEST', m=m, canary_ok=False)
    assert dec.tranche_stage == 0
    assert dec.position == 'OUT'
    assert dec.signal == 'CANARY DEFENSE'
    assert dec.target_exposure == 0.0

def test_case_02_canary_off_from_stage_1():
    m = make_metrics(price=96, ema50=95, ema100=98, ema200=100)
    dec = decide(score=70, breakdown=False, prev_position='T1', fund='TEST', m=m, canary_ok=False)
    assert dec.tranche_stage == 0
    assert dec.position == 'OUT'
    assert dec.signal == 'CANARY DEFENSE'

def test_case_03_canary_off_with_futures_triggered():
    m = make_metrics(price=100, ema50=95, ema100=90, ema200=85)
    dec = decide(score=85, breakdown=False, prev_position='T2', fund='TEST', m=m, canary_ok=False, futures_guard_triggered=True)
    assert dec.tranche_stage == 0
    assert dec.position == 'OUT'
    assert dec.signal == 'CANARY DEFENSE'

def test_case_04_canary_off_cold_start():
    m = make_metrics(price=100, ema50=95, ema100=90, ema200=85)
    dec = decide(score=85, breakdown=False, prev_position='OUT', fund='TEST', m=m, canary_ok=False)
    assert dec.tranche_stage == 0
    assert dec.position == 'OUT'
    assert dec.signal == 'CANARY DEFENSE'

# ------------------------------------------------------------------------------
# 2. Hard Breakdown (-2% EMA 200) Precedence Tests
# ------------------------------------------------------------------------------
def test_case_05_hard_breakdown_dist_pct():
    m = make_metrics(price=97, ema50=99, ema100=99, ema200=100, dist_ema200_pct=-3.0)
    dec = decide(score=40, breakdown=False, prev_position='T3', fund='TEST', m=m, canary_ok=True)
    assert dec.tranche_stage == 0
    assert dec.position == 'OUT'
    assert dec.signal == 'HARD EXIT'
    assert dec.target_exposure == 0.0

def test_case_06_hard_breakdown_flag():
    m = make_metrics(price=98, ema50=99, ema100=99, ema200=100, dist_ema200_pct=-2.5)
    dec = decide(score=30, breakdown=True, prev_position='T2', fund='TEST', m=m, canary_ok=True)
    assert dec.tranche_stage == 0
    assert dec.position == 'OUT'
    assert dec.signal == 'HARD EXIT'

def test_case_07_hard_breakdown_even_with_futures_pause():
    m = make_metrics(price=95, ema50=98, ema100=99, ema200=100, dist_ema200_pct=-5.0)
    dec = decide(score=20, breakdown=True, prev_position='T1', fund='TEST', m=m, canary_ok=True, futures_guard_triggered=True)
    assert dec.tranche_stage == 0
    assert dec.position == 'OUT'
    assert dec.signal == 'HARD EXIT'

# ------------------------------------------------------------------------------
# 3. Asymmetric De-risking (Normal Trim / Exit) Precedence Tests
# ------------------------------------------------------------------------------
def test_case_08_asymmetric_drop_stage_3_to_0():
    # Price falls below EMA50, EMA100, EMA200 (but not -2% below EMA200 yet)
    m = make_metrics(price=99, ema50=102, ema100=101, ema200=100, dist_ema200_pct=-1.0)
    dec = decide(score=45, breakdown=False, prev_position='T3', fund='TEST', m=m, canary_ok=True)
    assert dec.tranche_stage == 0
    assert dec.position == 'OUT'
    assert dec.signal == 'CASH PARK'
    assert dec.target_exposure == 0.0

def test_case_09_asymmetric_drop_stage_3_to_1():
    # Price > EMA50 (stage 1 condition), but broke EMA100 & EMA stack
    m = make_metrics(price=101, ema50=100, ema100=102, ema200=98)
    dec = decide(score=60, breakdown=False, prev_position='T3', fund='TEST', m=m, canary_ok=True)
    assert dec.tranche_stage == 1
    assert dec.position == 'IN'
    assert dec.signal == 'TRIM RISK'
    assert abs(dec.target_exposure - 1.0/3.0) < 1e-6

def test_case_10_asymmetric_drop_stage_2_to_0():
    # Previous T2, now below EMA50, EMA100, EMA200
    m = make_metrics(price=89, ema50=95, ema100=93, ema200=90, dist_ema200_pct=-1.1)
    dec = decide(score=50, breakdown=False, prev_position='T2', fund='TEST', m=m, canary_ok=True)
    assert dec.tranche_stage == 0
    assert dec.position == 'OUT'
    assert dec.signal == 'CASH PARK'
    assert dec.target_exposure == 0.0

# ------------------------------------------------------------------------------
# 4. Futures Guard PAUSE (Blocks Entry/Increase, allows Exit/Hold)
# ------------------------------------------------------------------------------
def test_case_11_futures_pause_blocks_0_to_1():
    m = make_metrics(price=96, ema50=95, ema100=98, ema200=90)
    dec = decide(score=70, breakdown=False, prev_position='OUT', fund='TEST', m=m, canary_ok=True, futures_guard_triggered=True)
    assert dec.tranche_stage == 0
    assert dec.position == 'OUT'
    assert dec.signal == 'FUTURES PAUSE'
    assert dec.target_exposure == 0.0

def test_case_12_futures_pause_blocks_1_to_2():
    m = make_metrics(price=101, ema50=100, ema100=98, ema200=95)
    dec = decide(score=80, breakdown=False, prev_position='T1', fund='TEST', m=m, canary_ok=True, futures_guard_triggered=True)
    assert dec.tranche_stage == 1
    assert dec.position == 'IN'
    assert dec.signal == 'FUTURES PAUSE'
    assert abs(dec.target_exposure - 1.0/3.0) < 1e-6

def test_case_13_futures_pause_allows_trim_down():
    # Previous T3, now condition drops to T2 (price broke EMA200 or EMA stack), futures triggered -> must allow trim to T2
    m = make_metrics(price=99, ema50=98, ema100=97, ema200=100, dist_ema200_pct=-1.0)
    dec = decide(score=65, breakdown=False, prev_position='T3', fund='TEST', m=m, canary_ok=True, futures_guard_triggered=True)
    assert dec.tranche_stage == 2
    assert dec.position == 'IN'
    assert dec.signal == 'TRIM RISK'
    assert abs(dec.target_exposure - 2.0/3.0) < 1e-6

# ------------------------------------------------------------------------------
# 5. Tranche Entry Ladder (Max +1 stage per day)
# ------------------------------------------------------------------------------
def test_case_14_entry_ladder_0_to_1():
    m = make_metrics(price=105, ema50=100, ema100=95, ema200=90)
    dec = decide(score=85, breakdown=False, prev_position='OUT', fund='TEST', m=m, canary_ok=True, futures_guard_triggered=False)
    assert dec.tranche_stage == 1
    assert dec.signal == 'STARTER'
    assert abs(dec.target_exposure - 1.0/3.0) < 1e-6

def test_case_15_entry_ladder_1_to_2():
    m = make_metrics(price=105, ema50=100, ema100=95, ema200=90)
    dec = decide(score=85, breakdown=False, prev_position='T1', fund='TEST', m=m, canary_ok=True, futures_guard_triggered=False)
    assert dec.tranche_stage == 2
    assert dec.signal == 'SCALE IN'
    assert abs(dec.target_exposure - 2.0/3.0) < 1e-6

def test_case_16_entry_ladder_2_to_3():
    m = make_metrics(price=105, ema50=100, ema100=95, ema200=90)
    dec = decide(score=85, breakdown=False, prev_position='T2', fund='TEST', m=m, canary_ok=True, futures_guard_triggered=False)
    assert dec.tranche_stage == 3
    assert dec.signal == 'FULL ALLOCATION'
    assert abs(dec.target_exposure - 1.0) < 1e-6
