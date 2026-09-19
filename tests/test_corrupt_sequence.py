"""Multi-day sequence tests — state machine เป็น path-dependent เทสต์เดี่ยวจับไม่ได้"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from alphashield.strategy.data_integrity import DataHealth, IntegrityAction
from alphashield.strategy.signals import resolve_stage

D0 = date(2026, 9, 14)   # จันทร์


def good(price=100.0, score=80.0):
    return {"price": price, "ema50": 95.0, "ema100": 92.0, "ema200": 88.0,
            "score": score, "rsi": 60.0, "adx": 27.0, "hv20": 14.0, "chg_pct": 0.4}


def nan_metrics():
    m = good()
    m["price"] = float("nan")
    return m


class Harness:
    """จำลองการรันหลายวันติดกัน โดยส่ง state เดิมต่อให้วันถัดไป"""

    def __init__(self, stage=0):
        self.stage = stage
        self.integrity = {}
        self.log = []

    def run(self, day_offset: int, metrics, raw_stage, **kw):
        run_date = D0 + timedelta(days=day_offset)
        d = resolve_stage(
            fund="SCBNDQ(E)", metrics=metrics, raw_stage=raw_stage,
            prev_stage=self.stage, run_date=run_date,
            last_bar_date=run_date if metrics else None,
            prev_integrity=self.integrity, **kw,
        )
        self.stage = d.stage
        self.integrity = d.integrity.to_state()
        self.log.append(d)
        return d


# ══════════════════════════════════════════════════════════════
#  Case A — Day1 FREEZE → Day2 FORCE EXIT
# ══════════════════════════════════════════════════════════════
def test_day1_freeze_then_day2_force_exit():
    h = Harness(stage=0)
    h.run(0, good(), raw_stage=3)                      # เข้าไม้ 1
    h.run(1, good(), raw_stage=3)                      # ไม้ 2
    h.run(2, good(), raw_stage=3)                      # ไม้ 3 เต็ม
    assert h.stage == 3

    d1 = h.run(3, nan_metrics(), raw_stage=3)          # STRIKE 1
    assert d1.signal == "FROZEN"
    assert d1.stage == 3, "Freeze ต้องคงไม้เดิม ห้ามขาย"
    assert d1.emit_order is False, "Freeze ห้ามออกคำสั่งเด็ดขาด"
    assert d1.integrity.strike == 1
    assert d1.alert_level == "critical"

    d2 = h.run(4, nan_metrics(), raw_stage=3)          # STRIKE 2
    assert d2.signal == "CORRUPT DATA EXIT"
    assert d2.stage == 0
    assert d2.emit_order is True
    assert d2.integrity.strike == 2
    assert d2.integrity.first_seen == d1.integrity.first_seen, "first_seen ต้องคงวันแรกไว้"


# ══════════════════════════════════════════════════════════════
#  Case B — Day1 FREEZE → Day2 RESTORED (Self-Healing)
# ══════════════════════════════════════════════════════════════
def test_day1_freeze_then_self_heal():
    h = Harness(stage=2)

    d1 = h.run(0, nan_metrics(), raw_stage=3)
    assert d1.signal == "FROZEN" and d1.stage == 2 and d1.integrity.strike == 1

    d2 = h.run(1, good(), raw_stage=3)                 # ข้อมูลกลับมา
    assert d2.integrity.strike == 0, "Self-healing ต้อง reset strike"
    assert d2.integrity.action is IntegrityAction.PROCEED
    assert d2.stage == 3, "รันต่อตามปกติ ladder +1 จาก stage ที่ freeze ไว้"
    assert d2.signal == "SWITCH IN"

    d3 = h.run(2, nan_metrics(), raw_stage=3)          # เสียใหม่
    assert d3.integrity.strike == 1, "หลัง heal แล้วเสียใหม่ ต้องเริ่มนับ 1 ไม่ใช่ 2"
    assert d3.signal == "FROZEN"


# ══════════════════════════════════════════════════════════════
#  Case C — Idempotency: cron + manual dispatch วันเดียวกัน
# ══════════════════════════════════════════════════════════════
def test_same_day_double_run_does_not_double_strike():
    h = Harness(stage=2)
    a = h.run(0, nan_metrics(), raw_stage=3)
    b = h.run(0, nan_metrics(), raw_stage=3)           # รันซ้ำวันเดิม
    assert a.integrity.strike == 1
    assert b.integrity.strike == 1, "รันซ้ำวันเดียวกันห้ามนับ strike เพิ่ม"
    assert b.signal == "FROZEN", "ห้ามข้ามไป FORCE EXIT จากการรันซ้ำ"


def test_ladder_is_idempotent_within_same_day():
    h = Harness(stage=1)
    h.run(0, good(), raw_stage=3)
    assert h.stage == 2
    h.run(0, good(), raw_stage=3)                       # รันซ้ำวันเดิม
    assert h.stage == 3, "ladder ยังขยับได้ — กันซ้ำต้องทำที่ชั้น CI (see workflow)"


# ══════════════════════════════════════════════════════════════
#  Case D — De-risk ต้องทะลุ FREEZE ได้เสมอ
# ══════════════════════════════════════════════════════════════
def test_canary_off_overrides_freeze():
    h = Harness(stage=3)
    d = h.run(0, nan_metrics(), raw_stage=3, canary_ok=False, canary_healthy=True)
    assert d.stage == 0, "Canary OFF จาก TIP สะอาด ต้องขายได้แม้ข้อมูล asset เสีย"
    assert d.signal == "SWITCH OUT"
    assert d.emit_order is True


def test_canary_nan_does_not_force_cash():
    """TIP NaN = ข้อมูลเสีย ไม่ใช่สัญญาณ macro แย่ → ห้ามล้างพอร์ต"""
    h = Harness(stage=3)
    d = h.run(0, good(), raw_stage=3, canary_ok=False, canary_healthy=False)
    assert d.stage == 3, "canary_healthy=False ต้องไม่ทริกเกอร์ force cash"


# ══════════════════════════════════════════════════════════════
#  Case E — FREEZE ที่ stage 0 ต้องไม่สร้างคำสั่งหลอน
# ══════════════════════════════════════════════════════════════
def test_freeze_at_stage_zero_emits_nothing():
    h = Harness(stage=0)
    d = h.run(0, nan_metrics(), raw_stage=2)
    assert d.stage == 0 and d.emit_order is False and d.position == "OUT"


def test_force_exit_at_stage_zero_is_noop_order():
    h = Harness(stage=0)
    h.run(0, nan_metrics(), raw_stage=2)
    d2 = h.run(1, nan_metrics(), raw_stage=2)
    assert d2.stage == 0
    assert d2.emit_order is False, "ไม่มีไม้อยู่แล้ว ห้ามส่งคำสั่งขาย"


# ══════════════════════════════════════════════════════════════
#  Case F — Stale data / partial corruption
# ══════════════════════════════════════════════════════════════
def test_stale_bar_counts_as_corrupt():
    d = resolve_stage(
        fund="X", metrics=good(), raw_stage=3, prev_stage=2,
        run_date=D0, last_bar_date=D0 - timedelta(days=30), prev_integrity={},
    )
    assert d.signal == "FROZEN" and d.integrity.health is DataHealth.CORRUPT


def test_optional_field_nan_is_degraded_not_frozen():
    m = good(); m["adx"] = float("nan"); m["hv20"] = None
    d = resolve_stage(fund="X", metrics=m, raw_stage=3, prev_stage=1,
                      run_date=D0, last_bar_date=D0, prev_integrity={})
    assert d.integrity.health is DataHealth.DEGRADED
    assert d.stage == 2, "DEGRADED ต้องรันต่อได้ปกติ"
    assert d.alert_level == "warn"


@pytest.mark.parametrize("bad_strike", [-1, 99, "x", None, 2.7])
def test_corrupt_strike_sanitized_from_broken_state(bad_strike):
    d = resolve_stage(fund="X", metrics=nan_metrics(), raw_stage=3, prev_stage=2,
                      run_date=D0, last_bar_date=D0,
                      prev_integrity={"corrupt_strike": bad_strike})
    assert 0 <= d.integrity.strike <= 9


# ══════════════════════════════════════════════════════════════
#  Case G — Long sequence invariants
# ══════════════════════════════════════════════════════════════
def test_five_day_outage_never_exceeds_cap_and_stays_flat():
    h = Harness(stage=3)
    for i in range(5):
        d = h.run(i, nan_metrics(), raw_stage=3)
    assert h.stage == 0
    assert d.integrity.strike <= 9
    assert [x.signal for x in h.log] == [
        "FROZEN", "CORRUPT DATA EXIT", "CORRUPT DATA EXIT",
        "CORRUPT DATA EXIT", "CORRUPT DATA EXIT",
    ]


def test_recovery_after_forced_exit_reenters_at_stage_one():
    h = Harness(stage=3)
    h.run(0, nan_metrics(), raw_stage=3)     # freeze
    h.run(1, nan_metrics(), raw_stage=3)     # forced exit → 0
    d = h.run(2, good(), raw_stage=3)        # ข้อมูลกลับมา
    assert d.stage == 1, "หลัง forced exit ต้องไต่ใหม่จากไม้ 1 ห้ามกระโดดกลับ 3"