"""
alphashield.engine.decision
===========================
Deep Module: Decision & Allocation Engine
Encapsulates strict deterministic precedence hierarchy, data integrity safeguards,
asymmetric de-risking, futures pause, tranche rate-limiting, and two-phase commit
approval gating behind a unified, cohesive interface.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

from alphashield.strategy.data_integrity import (
    DataHealth,
    IntegrityAction,
    IntegrityVerdict,
    assess_data_health,
    evaluate_integrity,
    is_nan,
)
from alphashield.strategy.signals import (
    TrancheDecision,
    resolve_stage,
    TRANCHE_EXPOSURE_MAP,
    TRANCHE_LABEL_MAP,
    STAGE_TO_POS,
)
from alphashield.execution.approval import (
    GateOutcome,
    GateVerdict,
    OrderSizing,
    ApprovalTicket,
    evaluate_gate,
    compute_order_sizing,
    ICT,
    SLEEVE_WEIGHT,
    SCB_MIN_SWITCH_THB,
)

LOG = logging.getLogger("quantbot.engine")


@dataclass(frozen=True)
class MarketSnapshot:
    """
    Encapsulates all market input data, indicators, macro regimes, and safety flags.
    """
    fund: str
    price: float = 0.0
    ema50: float = 0.0
    ema100: float = 0.0
    ema200: float = 0.0
    score: float = 0.0
    rsi: float = 50.0
    adx: float = 0.0
    hv20: float = 0.0
    chg_pct: float = 0.0
    dist_ema50_pct: float = 0.0
    dist_ema200_pct: float = 0.0
    bull_stack: bool = False
    breakdown: bool = False
    breakdown_reason: str = ""
    canary_ok: bool = True
    canary_healthy: bool = True
    futures_guard_triggered: bool = False
    last_bar_date: Optional[date] = None
    run_date: Optional[date] = None
    now_iso: Optional[str] = None
    is_data_available: bool = True


@dataclass(frozen=True)
class StrategyOutcome:
    """
    Encapsulates the complete strategy outcome, committed stage, target weight,
    execution order sizing, and approval gate interaction.
    """
    fund: str
    stage: int                       # Committed stage (0, 1, 2, 3)
    proposed_stage: int              # Stage proposed before approval gate
    exposure: float                  # Tranche exposure: 0.0, 1/3, 2/3, 1.0
    portfolio_weight: float          # exposure * sleeve_weight (e.g. 0.0 - 0.20)
    position: str                    # "IN" | "OUT"
    signal: str                      # "SWITCH IN", "TRIM RISK", "CASH PARK", "PENDING APPROVAL", etc.
    action: str                      # Short action tag e.g. [TRANCHE 1: 33%]
    action_detail: str               # Human readable explanation
    icon: str                        # 🟢, 🔵, 🟡, ⚪, 🔐, ⚠️, 🚨, 🧨
    emit_order: bool                 # True if actionable trade order is emitted
    changed: bool                    # True if stage changed from previous
    tranche_label: str               # Tranche display label
    gate_verdict: Optional[GateVerdict] = None
    approval_ticket: Optional[ApprovalTicket] = None
    integrity: Optional[IntegrityVerdict] = None
    order_sizing: Optional[OrderSizing] = None
    reasons: Tuple[str, ...] = field(default_factory=tuple)

    def to_summary_tuple(self) -> Tuple[str, float, str, str, str, str, int, str]:
        """Format tuple expected by report tables: (fund, score, signal, pos, icon, detail, stage, label)."""
        return (
            self.fund,
            0.0,  # score placeholder, caller can substitute score
            self.signal,
            self.position,
            self.icon,
            self.action_detail,
            self.stage,
            self.tranche_label,
        )


class DecisionEngine:
    """
    Deep Module: Decision & Allocation Engine.

    Encapsulates strict deterministic precedence:
      1. Assess Data Health & Integrity Engine (Corrupt Strike Freeze / Force Exit).
      2. HAA TIP Canary Check (if RED -> Force 100% Cash Park / Stage 0).
      3. Hard Breakdown (-2% below EMA 200 -> Force Exit / Stage 0).
      4. Asymmetric De-risking (raw_stage < prev_stage -> drop immediately).
      5. Pre-Market Futures Circuit Breaker (if TRIGGERED -> Pause opening/scaling).
      6. Tranche Entry Ladder (climb at most +1 stage/day).
      7. Two-Phase Commit Approval Gate (evaluate token and sizing).
    """

    def __init__(
        self,
        sleeve_weight: float = SLEEVE_WEIGHT,
        min_order_thb: float = SCB_MIN_SWITCH_THB,
        default_capital_thb: float = 100_000.0,
    ):
        self.sleeve_weight = sleeve_weight
        self.min_order_thb = min_order_thb
        self.default_capital_thb = default_capital_thb

    def compute_raw_stage(self, price: float, ema50: float, ema100: float, ema200: float) -> int:
        """Calculate technical raw stage (0, 1, 2, 3) from moving averages."""
        if price > ema200 and (ema50 > ema100 > ema200):
            return 3
        elif price > ema100 and (ema50 > ema100):
            return 2
        elif price > ema50:
            return 1
        return 0

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        current_state: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> StrategyOutcome:
        """
        Primary entry point for evaluating an asset strategy decision.
        """
        cfg = config or {}
        state = current_state or {}
        capital_thb = float(cfg.get("capital_thb", self.default_capital_thb))
        require_manual_confirm = bool(cfg.get("require_manual_confirm", True))
        submitted_token = str(cfg.get("submitted_token", "")).strip()
        now_dt = cfg.get("now_dt") or datetime.now(ICT)

        prev_entry = state.get(snapshot.fund, {}) if isinstance(state, Mapping) else {}
        prev_stage = int(prev_entry.get("tranche_stage", 0))
        prev_integ = prev_entry.get("integrity", {
            "corrupt_strike": prev_entry.get("corrupt_strike", 0),
            "corrupt_last_date": prev_entry.get("corrupt_last_date"),
            "corrupt_first_seen_ict": prev_entry.get("corrupt_first_seen_ict"),
        })

        run_date = snapshot.run_date or now_dt.date()
        now_iso = snapshot.now_iso or now_dt.isoformat(timespec="seconds")

        # ----------------------------------------------------------------------
        # 1. Evaluate Technical Raw Stage & Metrics Dictionary
        # ----------------------------------------------------------------------
        if not snapshot.is_data_available:
            metrics_dict = None
            raw_stage = 0
        else:
            raw_stage = self.compute_raw_stage(
                snapshot.price, snapshot.ema50, snapshot.ema100, snapshot.ema200
            )
            metrics_dict = {
                "price": snapshot.price,
                "ema50": snapshot.ema50,
                "ema100": snapshot.ema100,
                "ema200": snapshot.ema200,
                "score": snapshot.score,
                "rsi": snapshot.rsi,
                "adx": snapshot.adx,
                "hv20": snapshot.hv20,
                "chg_pct": snapshot.chg_pct,
            }

        # ----------------------------------------------------------------------
        # 2. Strategy Precedence Ladder (alphashield.strategy.signals.resolve_stage)
        # ----------------------------------------------------------------------
        td: TrancheDecision = resolve_stage(
            fund=snapshot.fund,
            metrics=metrics_dict,
            raw_stage=raw_stage,
            prev_stage=prev_stage,
            run_date=run_date,
            last_bar_date=snapshot.last_bar_date,
            prev_integrity=prev_integ,
            canary_ok=snapshot.canary_ok,
            canary_healthy=snapshot.canary_healthy,
            breakdown=snapshot.breakdown,
            futures_guard_triggered=snapshot.futures_guard_triggered,
            now_iso=now_iso,
        )

        proposed_stage = td.stage
        committed_stage = proposed_stage
        gate_verdict: Optional[GateVerdict] = None
        ticket: Optional[ApprovalTicket] = None
        final_signal = td.signal
        final_action = td.action
        final_detail = td.detail
        final_emit_order = td.emit_order
        final_pos = td.position

        # Determine default icon based on signal/stage
        dec_icon = "⚪"
        if td.signal == "CORRUPT DATA EXIT":
            dec_icon = "🚨"
        elif td.signal == "FROZEN":
            dec_icon = "⚠️"
        elif td.signal == "HARD EXIT":
            dec_icon = "🧨"
        elif td.signal == "SWITCH OUT":
            dec_icon = "⚪"
        elif td.signal == "TRIM RISK":
            dec_icon = "🔵" if td.stage == 2 else "🟡"
        elif td.signal == "FUTURES PAUSE":
            dec_icon = "⚠️"
        elif td.stage == 3:
            dec_icon = "🟢"
        elif td.stage == 2:
            dec_icon = "🔵"
        elif td.stage == 1:
            dec_icon = "🟡"

        # ----------------------------------------------------------------------
        # 3. Two-Phase Commit Approval Gate
        # ----------------------------------------------------------------------
        pending_approvals = cfg.get("pending_approvals", {})
        pending_dict = pending_approvals.get(snapshot.fund) if isinstance(pending_approvals, Mapping) else None

        is_first_switch_in = (prev_stage == 0 and proposed_stage == 1 and td.signal == "SWITCH IN")

        if is_first_switch_in and require_manual_confirm:
            gate_verdict = evaluate_gate(
                fund=snapshot.fund,
                prev_stage=prev_stage,
                proposed_stage=proposed_stage,
                signal=td.signal,
                capital_thb=capital_thb,
                require_manual_confirm=require_manual_confirm,
                pending=pending_dict,
                submitted_token=submitted_token,
                now=now_dt,
            )

            committed_stage = gate_verdict.committed_stage
            ticket = gate_verdict.ticket
            final_emit_order = gate_verdict.emit_order

            if gate_verdict.outcome == GateOutcome.APPROVED:
                LOG.info("Execution Gate: APPROVED for %s with token (committed_stage=%d)", snapshot.fund, committed_stage)
                final_action = f"[TRANCHE {committed_stage}: APPROVED]"
            elif gate_verdict.outcome in (GateOutcome.PENDING, GateOutcome.EXPIRED, GateOutcome.SUPERSEDED):
                token_str = ticket.token if ticket else "N/A"
                LOG.warning("Execution Gate: %s for %s (Token: %s)", gate_verdict.outcome.value, snapshot.fund, token_str)
                final_signal = "PENDING APPROVAL"
                final_pos = "OUT"
                dec_icon = "🔐"
                final_action = "[PENDING APPROVAL]"
                final_detail = f"รอการยืนยันคำสั่งจากมนุษย์ (Token: {token_str})"
            elif gate_verdict.outcome == GateOutcome.REJECTED_SIZE:
                reason = gate_verdict.reasons[0] if gate_verdict.reasons else "Order size too small"
                LOG.warning("Execution Gate: REJECTED_SIZE for %s (%s)", snapshot.fund, reason)
                final_pos = "OUT"
                final_detail = reason
                final_action = "[REJECTED SIZE]"

        # Calculate final exposure and portfolio weight
        final_exposure = TRANCHE_EXPOSURE_MAP.get(committed_stage, 0.0)
        portfolio_weight = final_exposure * self.sleeve_weight

        # Calculate Order Sizing
        order_sizing = compute_order_sizing(
            capital_thb=capital_thb,
            prev_stage=prev_stage,
            next_stage=committed_stage,
            sleeve_weight=self.sleeve_weight,
            minimum_thb=self.min_order_thb,
        )

        tranche_label = TRANCHE_LABEL_MAP.get(committed_stage, "0% (Cash Park)")
        if final_signal == "PENDING APPROVAL":
            tranche_label = "0% (Pending Confirmation)"

        changed = (committed_stage != prev_stage)

        return StrategyOutcome(
            fund=snapshot.fund,
            stage=committed_stage,
            proposed_stage=proposed_stage,
            exposure=final_exposure,
            portfolio_weight=portfolio_weight,
            position=final_pos,
            signal=final_signal,
            action=final_action,
            action_detail=final_detail,
            icon=dec_icon,
            emit_order=final_emit_order,
            changed=changed,
            tranche_label=tranche_label,
            gate_verdict=gate_verdict,
            approval_ticket=ticket,
            integrity=td.integrity,
            order_sizing=order_sizing,
            reasons=gate_verdict.reasons if gate_verdict else (),
        )
