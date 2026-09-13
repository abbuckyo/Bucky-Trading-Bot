"""
Unit tests for Divergence Monitor (tools/divergence_log.py)
===========================================================
Verifies dual-engine decision comparison between:
- Engine A (Legacy V7.2): Binary 0/100% Quant Score
- Engine B (AlphaShield V8.1): Staged Tranche + TIP Canary + Futures Guard + Data Integrity
"""
import os
import csv
import pytest
from tools.divergence_log import (
    decide_v7_legacy,
    classify_divergence,
    record_divergence_log,
    log_daily_divergence,
    CSV_HEADER,
)


def test_v7_legacy_hysteresis():
    # Buy threshold >= 75.0
    assert decide_v7_legacy(score=80.0, breakdown=False, prev_pos="OUT") == "IN"
    assert decide_v7_legacy(score=75.0, breakdown=False, prev_pos="OUT") == "IN"

    # Sell threshold < 55.0
    assert decide_v7_legacy(score=54.9, breakdown=False, prev_pos="IN") == "OUT"
    assert decide_v7_legacy(score=40.0, breakdown=False, prev_pos="IN") == "OUT"

    # Breakdown override
    assert decide_v7_legacy(score=85.0, breakdown=True, prev_pos="IN") == "OUT"

    # Hysteresis middle band (55 <= score < 75)
    assert decide_v7_legacy(score=65.0, breakdown=False, prev_pos="OUT") == "OUT"
    assert decide_v7_legacy(score=65.0, breakdown=False, prev_pos="IN") == "IN"
    assert decide_v7_legacy(score=65.0, breakdown=False, prev_pos="T2") == "IN"


def test_classify_aligned():
    # Both OUT
    div, reason = classify_divergence(
        v7_pos="OUT", v7_score=45.0, v8_stage=0, v8_exposure=0.0,
        v8_signal="CASH PARK", canary_ok=True, futures_guard=False
    )
    assert div is False
    assert reason == "ALIGNED"

    # Both 100%
    div, reason = classify_divergence(
        v7_pos="IN", v7_score=80.0, v8_stage=3, v8_exposure=1.0,
        v8_signal="SWITCH IN", canary_ok=True, futures_guard=False
    )
    assert div is False
    assert reason == "ALIGNED"


def test_classify_macro_canary_divergence():
    # V7.2 sees great score (85) -> IN (100%), but V8.1 enforces Macro Canary OFF -> Cash Park (0.0)
    div, reason = classify_divergence(
        v7_pos="IN", v7_score=85.0, v8_stage=0, v8_exposure=0.0,
        v8_signal="CANARY DEFENSE", canary_ok=False, futures_guard=False
    )
    assert div is True
    assert reason == "MACRO_CANARY_OFF"


def test_classify_ladder_steps():
    # V7.2 all-in (IN 100%) vs V8.1 ladder stage 1 (Starter 33%)
    div, reason = classify_divergence(
        v7_pos="IN", v7_score=80.0, v8_stage=1, v8_exposure=1.0/3.0,
        v8_signal="SWITCH IN", canary_ok=True, futures_guard=False
    )
    assert div is True
    assert reason == "LADDER_STEP_1"

    # V7.2 all-in (IN 100%) vs V8.1 ladder stage 2 (Mid 66%)
    div, reason = classify_divergence(
        v7_pos="IN", v7_score=80.0, v8_stage=2, v8_exposure=2.0/3.0,
        v8_signal="SWITCH IN", canary_ok=True, futures_guard=False
    )
    assert div is True
    assert reason == "LADDER_STEP_2"


def test_classify_futures_pause():
    # V8.1 circuit breaker paused entry/scaling
    div, reason = classify_divergence(
        v7_pos="IN", v7_score=78.0, v8_stage=1, v8_exposure=1.0/3.0,
        v8_signal="FUTURES PAUSE", canary_ok=True, futures_guard=True
    )
    assert div is True
    assert reason == "FUTURES_PAUSE"


def test_classify_de_risking():
    # V7.2 is still IN (score 65 in hysteresis), but V8.1 broke below EMA50 -> Trim Risk or Cash Park
    div, reason = classify_divergence(
        v7_pos="IN", v7_score=65.0, v8_stage=0, v8_exposure=0.0,
        v8_signal="CASH PARK", canary_ok=True, futures_guard=False
    )
    assert div is True
    assert reason == "DE_RISKING"


def test_classify_corrupt_and_frozen():
    # V8.1 2-strike corrupt data exit
    div, reason = classify_divergence(
        v7_pos="OUT", v7_score=0.0, v8_stage=0, v8_exposure=0.0,
        v8_signal="CORRUPT DATA EXIT", canary_ok=True, futures_guard=False,
        is_corrupt=True
    )
    assert div is True
    assert reason == "CORRUPT_DATA_EXIT"

    # V8.1 Day 1 frozen
    div, reason = classify_divergence(
        v7_pos="OUT", v7_score=0.0, v8_stage=2, v8_exposure=2.0/3.0,
        v8_signal="FROZEN", canary_ok=True, futures_guard=False
    )
    assert div is True
    assert reason == "DATA_INTEGRITY_FREEZE"


def test_record_divergence_log_and_batch(tmp_path):
    test_csv = str(tmp_path / "test_divergence.csv")

    assets_mock = {
        "SCBNDQ(E)": {
            "tranche_stage": 1,
            "target_exposure": 1.0 / 3.0,
            "signal": "SWITCH IN",
            "score": 80.0,
            "breakdown": False,
            "position": "IN",
        },
        "SCBGOLDE": {
            "tranche_stage": 0,
            "target_exposure": 0.0,
            "signal": "CANARY DEFENSE",
            "score": 76.0,
            "breakdown": False,
            "position": "OUT",
        }
    }

    records = log_daily_divergence(
        run_date="2026-09-14",
        assets_state=assets_mock,
        tip_mom=-0.5,
        canary_ok=False,
        futures_guard=False,
        log_path=test_csv,
    )

    assert len(records) == 2
    assert os.path.exists(test_csv)

    with open(test_csv, "r", encoding="utf-8") as fh:
        reader = list(csv.DictReader(fh))
        assert len(reader) == 2
        assert list(reader[0].keys()) == CSV_HEADER
        # Row 1 check
        assert reader[0]["fund"] == "SCBNDQ(E)"
        assert reader[0]["v7_pos"] == "IN"
        assert reader[0]["v8_stage"] == "1"
        assert reader[0]["divergence_detected"] == "True"
        assert reader[0]["reason"] == "MACRO_CANARY_OFF"  # canary_ok is False
