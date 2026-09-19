"""
AlphaShield V8.1 — Precedence Resolver
======================================
Strict deterministic precedence order for AlphaShield V8.1:
Step 0a: Canary Clean (Macro De-risk: if canary_ok=False and canary_healthy=True -> stage 0)
Step 0b: Corrupt Data Force Exit (strike >= 2 -> stage 0)
Step 0c: Corrupt Data Freeze (strike 1 -> hold prev_stage, emit_order=False)
Step 1:  Hard Breakdown (-2% below EMA200 -> stage 0)
Step 2:  De-risk / Normal Exit (raw_stage < prev_stage -> drop immediately to raw_stage)
Step 3:  Futures Pause (raw_stage > prev_stage and futures_guard_triggered -> hold prev_stage)
Step 4:  Entry Ladder (raw_stage > prev_stage -> climb +1 stage)
Step 5:  Steady State (raw_stage == prev_stage -> hold)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Optional

from alphashield.strategy.data_integrity import (
    DataHealth,
    IntegrityAction,
    IntegrityVerdict,
    assess_data_health,
    evaluate_integrity,
)

TRANCHE_EXPOSURE_MAP: dict[int, float] = {
    0: 0.0,
    1: 1.0 / 3.0,
    2: 2.0 / 3.0,
    3: 1.0,
}

TRANCHE_LABEL_MAP: dict[int, str] = {
    0: "0% (Cash Park)",
    1: "33% (Starter)",
    2: "66% (Scale In)",
    3: "100% (Full Allocation)",
}

STAGE_TO_POS: dict[int, str] = {
    0: "OUT",
    1: "IN",
    2: "IN",
    3: "IN",
}


@dataclass(frozen=True)
class TrancheDecision:
    fund: str
    stage: int
    exposure: float
    position: str             # "IN" | "OUT"
    signal: str               # "SWITCH IN" | "SWITCH OUT" | "FROZEN" | "CORRUPT DATA EXIT" | "CASH PARK" | "TRIM RISK" | "FUTURES PAUSE" | "HOLD"
    action: str
    emit_order: bool          # False when FROZEN, or when target == prev
    alert_level: str          # "none" | "warn" | "critical"
    integrity: IntegrityVerdict
    detail: str
    changed: bool


def resolve_stage(
    fund: str,
    metrics: Optional[Mapping[str, Any]],
    raw_stage: int,
    prev_stage: int,
    run_date: date,
    last_bar_date: Any = None,
    prev_integrity: Optional[Mapping[str, Any]] = None,
    canary_ok: bool = True,
    canary_healthy: bool = True,
    breakdown: bool = False,
    futures_guard_triggered: bool = False,
    now_iso: Optional[str] = None,
) -> TrancheDecision:
    """
    Core Precedence Resolver for AlphaShield V8.1.
    Evaluates integrity verdict and executes deterministic precedence ladder.
    """
    prev_integ = prev_integrity or {}
    prev_strike = prev_integ.get("corrupt_strike", 0)
    prev_last_date = prev_integ.get("corrupt_last_date")
    prev_first_seen = prev_integ.get("corrupt_first_seen_ict")

    # Assess data health
    health_report = assess_data_health(metrics, last_bar_date=last_bar_date, run_date=run_date)
    verdict = evaluate_integrity(
        health_report,
        run_date=run_date,
        prev_strike=prev_strike,
        prev_last_date=prev_last_date,
        prev_first_seen=prev_first_seen,
        now_iso=now_iso,
    )

    prev_stage = max(0, min(int(prev_stage), 3))

    # --------------------------------------------------------------------------
    # Step 0a: Canary Macro De-risk (Canary clean & negative momentum)
    # De-risk overrides corrupt data freeze!
    # If canary_healthy is False, Canary NaN is bad data, NOT a macro signal.
    # --------------------------------------------------------------------------
    if canary_healthy and not canary_ok:
        stage = 0
        changed = (prev_stage != 0)
        return TrancheDecision(
            fund=fund,
            stage=stage,
            exposure=TRANCHE_EXPOSURE_MAP[stage],
            position="OUT",
            signal="SWITCH OUT" if changed else "CASH PARK",
            action="[CANARY CASH]",
            emit_order=changed,
            alert_level="warn" if changed else "none",
            integrity=verdict,
            detail=f"HAA TIP Canary สั่งล็อกหลบภัย 100% (Clean Macro Signal)",
            changed=changed,
        )

    # --------------------------------------------------------------------------
    # Step 0b: Corrupt Data Force Exit (Strike >= 2)
    # --------------------------------------------------------------------------
    if verdict.action is IntegrityAction.FORCE_EXIT:
        stage = 0
        changed = (prev_stage != 0)
        # emit_order is True only if we actually held a position that needs liquidation
        emit_order = (prev_stage > 0)
        return TrancheDecision(
            fund=fund,
            stage=stage,
            exposure=TRANCHE_EXPOSURE_MAP[stage],
            position="OUT",
            signal="CORRUPT DATA EXIT",
            action="[FORCE EXIT CASH]",
            emit_order=emit_order,
            alert_level=verdict.alert_level,
            integrity=verdict,
            detail=f"ข้อมูลเสียหายต่อเนื่อง 2 วัน (Strike {verdict.strike}) ล้างพอร์ตเข้า Cash Park เพื่อความปลอดภัย",
            changed=changed,
        )

    # --------------------------------------------------------------------------
    # Step 0c: Corrupt Data Freeze (Strike 1)
    # --------------------------------------------------------------------------
    if verdict.action is IntegrityAction.FREEZE:
        stage = prev_stage
        return TrancheDecision(
            fund=fund,
            stage=stage,
            exposure=TRANCHE_EXPOSURE_MAP[stage],
            position=STAGE_TO_POS[stage],
            signal="FROZEN",
            action=f"[FREEZE: T{stage}]",
            emit_order=False,
            alert_level=verdict.alert_level,
            integrity=verdict,
            detail=f"ข้อมูลผิดปกติวันแรก (Strike 1) ตรึงสถานะเดิม ไม่ส่งคำสั่งซื้อขาย",
            changed=False,
        )

    # --------------------------------------------------------------------------
    # Step 1: Hard Breakdown (-2% EMA 200)
    # --------------------------------------------------------------------------
    if breakdown:
        stage = 0
        changed = (prev_stage != 0)
        emit_order = (prev_stage > 0)
        return TrancheDecision(
            fund=fund,
            stage=stage,
            exposure=TRANCHE_EXPOSURE_MAP[stage],
            position="OUT",
            signal="SWITCH OUT" if changed else "CASH PARK",
            action="[HARD EXIT]",
            emit_order=emit_order,
            alert_level="critical" if changed else "none",
            integrity=verdict,
            detail="หลุดแนวรับวิกฤต EMA200 เกิน -2% ตัดขาย 100% เข้า Cash Park",
            changed=changed,
        )

    # --------------------------------------------------------------------------
    # Step 2: De-risk / Normal Exit (raw_stage < prev_stage)
    # Asymmetric De-risking: Drops immediately to raw_stage
    # --------------------------------------------------------------------------
    if raw_stage < prev_stage:
        stage = raw_stage
        changed = True
        return TrancheDecision(
            fund=fund,
            stage=stage,
            exposure=TRANCHE_EXPOSURE_MAP[stage],
            position=STAGE_TO_POS[stage],
            signal="SWITCH OUT" if stage == 0 else "TRIM RISK",
            action=f"[TRIM: T{stage}]" if stage > 0 else "[CASH PARK]",
            emit_order=True,
            alert_level="warn",
            integrity=verdict,
            detail=f"ลดสัดส่วนลงมาที่ไม้ {stage} ({TRANCHE_LABEL_MAP[stage]})",
            changed=changed,
        )

    # --------------------------------------------------------------------------
    # Step 3: Futures Pause (raw_stage > prev_stage and futures_guard_triggered)
    # Blocks ONLY entry or scale up
    # --------------------------------------------------------------------------
    if raw_stage > prev_stage and futures_guard_triggered:
        stage = prev_stage
        return TrancheDecision(
            fund=fund,
            stage=stage,
            exposure=TRANCHE_EXPOSURE_MAP[stage],
            position=STAGE_TO_POS[stage],
            signal="FUTURES PAUSE",
            action=f"[PAUSE: T{stage}]",
            emit_order=False,
            alert_level="warn",
            integrity=verdict,
            detail="US Futures ติดลบหนัก ชะลอการเปิด/เพิ่มไม้ คงสถานะเดิม",
            changed=False,
        )

    # --------------------------------------------------------------------------
    # Step 4: Entry Ladder (raw_stage > prev_stage)
    # Rate Limiting: Max +1 stage per day
    # --------------------------------------------------------------------------
    if raw_stage > prev_stage:
        stage = prev_stage + 1
        return TrancheDecision(
            fund=fund,
            stage=stage,
            exposure=TRANCHE_EXPOSURE_MAP[stage],
            position="IN",
            signal="SWITCH IN",
            action=f"[TRANCHE {stage}: {TRANCHE_LABEL_MAP[stage]}]",
            emit_order=True,
            alert_level="none" if verdict.alert_level == "none" else verdict.alert_level,
            integrity=verdict,
            detail=f"ไต่ระดับเข้าไม้ {stage} ({TRANCHE_LABEL_MAP[stage]})",
            changed=True,
        )

    # --------------------------------------------------------------------------
    # Step 5: Steady State (raw_stage == prev_stage)
    # --------------------------------------------------------------------------
    stage = prev_stage
    return TrancheDecision(
        fund=fund,
        stage=stage,
        exposure=TRANCHE_EXPOSURE_MAP[stage],
        position=STAGE_TO_POS[stage],
        signal="INVESTED" if stage > 0 else "CASH PARK",
        action=f"[HOLD: T{stage}]",
        emit_order=False,
        alert_level="none" if verdict.alert_level == "none" else verdict.alert_level,
        integrity=verdict,
        detail=f"คงสถานะเดิมที่ไม้ {stage} ({TRANCHE_LABEL_MAP[stage]})",
        changed=False,
    )
