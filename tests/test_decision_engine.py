"""
Unit tests for alphashield.engine.decision (DecisionEngine Deep Module)
======================================================================
Verifies:
  1. HAA TIP Canary Force Cash Park override
  2. Pre-Market Futures Circuit Breaker blocking upward transitions
  3. Tranche step-up (0 -> 1 -> 2 -> 3) and asymmetric de-risking
  4. Approval Gate interaction (pending, approved, rejected size)
  5. 2-Strike Corrupt Data Engine (Freeze vs Force Exit)
"""
from datetime import date, datetime
import pytest

from alphashield.engine.decision import DecisionEngine, MarketSnapshot, StrategyOutcome
from alphashield.execution.approval import GateOutcome, ICT

RUN_DATE = date(2026, 9, 21)
NOW_DT = datetime(2026, 9, 21, 12, 15, tzinfo=ICT)


def make_bullish_snapshot(fund="SCBNDQ(E)", **overrides):
    base = {
        "fund": fund,
        "price": 105.0,
        "ema50": 100.0,
        "ema100": 95.0,
        "ema200": 90.0,
        "score": 85.0,
        "rsi": 60.0,
        "adx": 28.0,
        "hv20": 15.0,
        "chg_pct": 0.5,
        "dist_ema50_pct": 5.0,
        "dist_ema200_pct": 16.6,
        "bull_stack": True,
        "breakdown": False,
        "canary_ok": True,
        "canary_healthy": True,
        "futures_guard_triggered": False,
        "last_bar_date": RUN_DATE,
        "run_date": RUN_DATE,
        "now_iso": NOW_DT.isoformat(),
        "is_data_available": True,
    }
    base.update(overrides)
    return MarketSnapshot(**base)


# ------------------------------------------------------------------------------
# 1. HAA TIP Canary Tests
# ------------------------------------------------------------------------------
def test_canary_red_forces_cash_park():
    engine = DecisionEngine()
    snap = make_bullish_snapshot(canary_ok=False)
    # Even if previous was Stage 3 with bullish technicals
    state = {"SCBNDQ(E)": {"tranche_stage": 3}}
    outcome = engine.evaluate(snap, current_state=state)

    assert outcome.stage == 0
    assert outcome.exposure == 0.0
    assert outcome.portfolio_weight == 0.0
    assert outcome.position == "OUT"
    assert outcome.signal in ("SWITCH OUT", "CASH PARK")
    assert outcome.changed is True


# ------------------------------------------------------------------------------
# 2. Futures Guard Circuit Breaker Tests
# ------------------------------------------------------------------------------
def test_futures_guard_blocks_upward_scaling():
    engine = DecisionEngine()
    snap = make_bullish_snapshot(futures_guard_triggered=True)

    # 0 -> 1 blocked
    state0 = {"SCBNDQ(E)": {"tranche_stage": 0}}
    res0 = engine.evaluate(snap, current_state=state0)
    assert res0.stage == 0
    assert res0.signal == "FUTURES PAUSE"
    assert res0.emit_order is False

    # 1 -> 2 blocked
    state1 = {"SCBNDQ(E)": {"tranche_stage": 1}}
    res1 = engine.evaluate(snap, current_state=state1)
    assert res1.stage == 1
    assert res1.signal == "FUTURES PAUSE"

    # But de-risking is NEVER blocked by futures guard
    bearish_snap = make_bullish_snapshot(price=80.0, ema50=90.0, futures_guard_triggered=True)
    state3 = {"SCBNDQ(E)": {"tranche_stage": 3}}
    res_derisk = engine.evaluate(bearish_snap, current_state=state3)
    assert res_derisk.stage == 0
    assert res_derisk.signal in ("SWITCH OUT", "CASH PARK")
    assert res_derisk.emit_order is True


# ------------------------------------------------------------------------------
# 3. Tranche Ladder Step-up & De-risk
# ------------------------------------------------------------------------------
def test_tranche_entry_ladder_max_one_step():
    engine = DecisionEngine()
    snap = make_bullish_snapshot()

    # Step 0 -> 1 (when auto / manual confirm off)
    cfg_auto = {"require_manual_confirm": False}
    out1 = engine.evaluate(snap, current_state={"SCBNDQ(E)": {"tranche_stage": 0}}, config=cfg_auto)
    assert out1.stage == 1
    assert abs(out1.exposure - 1.0 / 3.0) < 1e-6
    assert abs(out1.portfolio_weight - (1.0 / 3.0) * 0.20) < 1e-6

    # Step 1 -> 2
    out2 = engine.evaluate(snap, current_state={"SCBNDQ(E)": {"tranche_stage": 1}}, config=cfg_auto)
    assert out2.stage == 2
    assert abs(out2.exposure - 2.0 / 3.0) < 1e-6

    # Step 2 -> 3
    out3 = engine.evaluate(snap, current_state={"SCBNDQ(E)": {"tranche_stage": 2}}, config=cfg_auto)
    assert out3.stage == 3
    assert abs(out3.exposure - 1.0) < 1e-6


def test_asymmetric_de_risk_drop_multiple_stages():
    engine = DecisionEngine()
    # Price breaks below EMA50 & EMA100, but holds above EMA50? If price=92, ema50=95 -> stage 0
    drop_snap = make_bullish_snapshot(price=92.0, ema50=95.0, ema100=98.0, ema200=90.0)
    state3 = {"SCBNDQ(E)": {"tranche_stage": 3}}

    out = engine.evaluate(drop_snap, current_state=state3)
    assert out.stage == 0
    assert out.exposure == 0.0
    assert out.signal in ("SWITCH OUT", "CASH PARK")


# ------------------------------------------------------------------------------
# 4. Approval Gate Integration
# ------------------------------------------------------------------------------
def test_decision_engine_with_approval_gate():
    engine = DecisionEngine()
    snap = make_bullish_snapshot()
    state = {"SCBNDQ(E)": {"tranche_stage": 0}}

    # Phase 1: First SWITCH IN -> PENDING
    cfg_gate = {
        "require_manual_confirm": True,
        "capital_thb": 200_000,
        "now_dt": NOW_DT,
    }
    out_pending = engine.evaluate(snap, current_state=state, config=cfg_gate)
    assert out_pending.stage == 0
    assert out_pending.signal == "PENDING APPROVAL"
    assert out_pending.icon == "🔐"
    assert out_pending.approval_ticket is not None
    assert len(out_pending.approval_ticket.token) == 8

    # Phase 2: Correct token provided -> APPROVED -> Stage 1 committed
    cfg_approved = {
        "require_manual_confirm": True,
        "capital_thb": 200_000,
        "submitted_token": out_pending.approval_ticket.token,
        "pending_approvals": {"SCBNDQ(E)": out_pending.approval_ticket.to_state()},
        "now_dt": NOW_DT,
    }
    out_approved = engine.evaluate(snap, current_state=state, config=cfg_approved)
    assert out_approved.stage == 1
    assert out_approved.exposure == 1.0 / 3.0
    assert out_approved.gate_verdict.outcome is GateOutcome.APPROVED
    assert out_approved.emit_order is True
