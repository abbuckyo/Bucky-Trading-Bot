"""
AlphaShield V8.1 — Divergence Monitor (Shadow Mode)
===================================================
Compares parallel decision outputs:
- Engine A (Legacy V7.2): Deterministic Quant Score (75/55) with Binary Position (0/100%)
- Engine B (AlphaShield V8.1): Staged Tranche Scaling (0%, 33%, 66%, 100%) + TIP Regime Canary
                              + Pre-Market US Futures Guard + 2-Strike Corrupt Data Integrity

Appends daily comparison rows to logs/divergence_log.csv:
[date, fund, v7_pos, v7_score, v8_stage, v8_exposure, v8_signal, tip_mom, canary_ok, futures_guard, divergence_detected, reason]
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Tuple

LOG = logging.getLogger("quantbot.divergence")

CSV_HEADER = [
    "date",
    "fund",
    "v7_pos",
    "v7_score",
    "v8_stage",
    "v8_exposure",
    "v8_signal",
    "tip_mom",
    "canary_ok",
    "futures_guard",
    "divergence_detected",
    "reason",
]


def decide_v7_legacy(score: float, breakdown: bool, prev_pos: str = "OUT") -> str:
    """
    Engine A (Legacy V7.2):
    - Hard breakdown or score < 55.0 -> OUT (Cash Park)
    - score >= 75.0 -> IN (100%)
    - 55.0 <= score < 75.0 -> Hysteresis hold prev_pos ('IN' or 'OUT')
    """
    p_norm = str(prev_pos).upper()
    effective_prev = "IN" if p_norm in ("IN", "T1", "T2", "T3", "FULL") else "OUT"

    if breakdown or score < 55.0:
        return "OUT"
    elif score >= 75.0:
        return "IN"
    else:
        return effective_prev


def classify_divergence(
    v7_pos: str,
    v7_score: float,
    v8_stage: int,
    v8_exposure: float,
    v8_signal: str,
    canary_ok: bool,
    futures_guard: bool,
    breakdown: bool = False,
    is_corrupt: bool = False,
) -> Tuple[bool, str]:
    """
    Identifies whether V7.2 and V8.1 diverge, and categorizes the underlying rationale.
    """
    # Check if exposures or states diverge
    # V7.2 is binary: IN (1.0) or OUT (0.0)
    v7_exposure = 1.0 if v7_pos == "IN" else 0.0
    divergence_detected = (v7_exposure != v8_exposure) or (v8_signal in ("FUTURES PAUSE", "FROZEN", "CORRUPT DATA EXIT"))

    if not divergence_detected:
        return False, "ALIGNED"

    # Determine reason
    if is_corrupt or v8_signal == "CORRUPT DATA EXIT":
        return True, "CORRUPT_DATA_EXIT"
    if v8_signal == "FROZEN":
        return True, "DATA_INTEGRITY_FREEZE"
    if not canary_ok or v8_signal == "CANARY DEFENSE":
        return True, "MACRO_CANARY_OFF"
    if futures_guard or v8_signal == "FUTURES PAUSE":
        return True, "FUTURES_PAUSE"
    if breakdown or v8_signal == "HARD EXIT":
        return True, "HARD_BREAKDOWN"
    if v8_signal in ("TRIM RISK", "CASH PARK", "SWITCH OUT") and v7_pos == "IN":
        return True, "DE_RISKING"
    if v8_stage == 1:
        return True, "LADDER_STEP_1"
    if v8_stage == 2:
        return True, "LADDER_STEP_2"
    if v8_stage == 3 and v7_pos == "OUT":
        return True, "QUANT_VS_EMA_DIVERGENCE"

    return True, f"DIVERGENCE_{v8_signal}"


def record_divergence_log(
    run_date: str,
    fund: str,
    v7_pos: str,
    v7_score: float,
    v8_stage: int,
    v8_exposure: float,
    v8_signal: str,
    tip_mom: float,
    canary_ok: bool,
    futures_guard: bool,
    divergence_detected: bool,
    reason: str,
    log_path: str = "logs/divergence_log.csv",
) -> Dict[str, Any]:
    """
    Appends a single observation row to CSV log.
    """
    row = {
        "date": run_date,
        "fund": fund,
        "v7_pos": v7_pos,
        "v7_score": f"{v7_score:.2f}",
        "v8_stage": v8_stage,
        "v8_exposure": f"{v8_exposure:.4f}",
        "v8_signal": v8_signal,
        "tip_mom": f"{tip_mom:.2f}",
        "canary_ok": canary_ok,
        "futures_guard": futures_guard,
        "divergence_detected": divergence_detected,
        "reason": reason,
    }

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0

    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return row


def log_daily_divergence(
    run_date: str,
    assets_state: Mapping[str, Any],
    tip_mom: float,
    canary_ok: bool,
    futures_guard: bool,
    log_path: str = "logs/divergence_log.csv",
) -> List[Dict[str, Any]]:
    """
    Iterates over processed assets from bot.py and writes divergence log rows.
    """
    records = []
    for fund, entry in assets_state.items():
        v8_stage = entry.get("tranche_stage", 0)
        v8_exposure = entry.get("target_exposure", 0.0)
        v8_signal = entry.get("signal", "NO DATA")
        score = entry.get("score", 0.0)
        breakdown = entry.get("breakdown", False)
        prev_pos = entry.get("position", "OUT")

        # Compute Legacy V7.2 Decision
        v7_pos = decide_v7_legacy(score, breakdown=breakdown, prev_pos=prev_pos)

        # Check for corruption in integrity verdict
        integrity_state = entry.get("integrity", {})
        strike = integrity_state.get("corrupt_strike", 0)
        is_corrupt = (strike >= 2) or (v8_signal == "CORRUPT DATA EXIT")

        div_detected, reason = classify_divergence(
            v7_pos=v7_pos,
            v7_score=score,
            v8_stage=v8_stage,
            v8_exposure=v8_exposure,
            v8_signal=v8_signal,
            canary_ok=canary_ok,
            futures_guard=futures_guard,
            breakdown=breakdown,
            is_corrupt=is_corrupt,
        )

        rec = record_divergence_log(
            run_date=run_date,
            fund=fund,
            v7_pos=v7_pos,
            v7_score=score,
            v8_stage=v8_stage,
            v8_exposure=v8_exposure,
            v8_signal=v8_signal,
            tip_mom=tip_mom,
            canary_ok=canary_ok,
            futures_guard=futures_guard,
            divergence_detected=div_detected,
            reason=reason,
            log_path=log_path,
        )
        records.append(rec)

    return records
