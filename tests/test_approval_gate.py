from datetime import datetime, timedelta
from alphashield.execution.approval import (
    ICT, GateOutcome, evaluate_gate, compute_order_sizing,
)

NOW = datetime(2026, 9, 21, 12, 15, tzinfo=ICT)
BASE = dict(fund="SCBNDQ(E)", prev_stage=0, proposed_stage=1,
            signal="SWITCH IN", capital_thb=200_000, require_manual_confirm=True)


def test_first_switch_in_is_pending_and_writes_nothing():
    v = evaluate_gate(**BASE, now=NOW)
    assert v.outcome is GateOutcome.PENDING
    assert v.committed_stage == 0, "PENDING ห้ามขยับ stage"
    assert v.emit_order is False
    assert len(v.ticket.token) == 8


def test_correct_token_approves_and_unlocks_auto():
    p = evaluate_gate(**BASE, now=NOW)
    v = evaluate_gate(**BASE, pending=p.ticket.to_state(),
                      submitted_token=p.ticket.token.lower(), now=NOW)
    assert v.outcome is GateOutcome.APPROVED
    assert v.committed_stage == 1 and v.emit_order and v.unlock_auto


def test_wrong_token_stays_pending():
    p = evaluate_gate(**BASE, now=NOW)
    v = evaluate_gate(**BASE, pending=p.ticket.to_state(),
                      submitted_token="DEADBEEF", now=NOW)
    assert v.outcome is GateOutcome.PENDING and v.committed_stage == 0


def test_derisk_never_blocked_by_gate():
    v = evaluate_gate(fund="X", prev_stage=3, proposed_stage=0, signal="SWITCH OUT",
                      capital_thb=5_000, require_manual_confirm=True, now=NOW)
    assert v.outcome is GateOutcome.NOT_REQUIRED and v.emit_order


def test_tiny_capital_rejected_not_advanced():
    v = evaluate_gate(**{**BASE, "capital_thb": 10_000}, now=NOW)   # ไม้ = 667 บาท
    assert v.outcome is GateOutcome.REJECTED_SIZE
    assert v.committed_stage == 0


def test_ticket_superseded_when_signal_changes():
    p = evaluate_gate(**BASE, now=NOW)
    v = evaluate_gate(**{**BASE, "prev_stage": 1, "proposed_stage": 2},
                      pending=p.ticket.to_state(),
                      submitted_token=p.ticket.token, now=NOW)
    assert v.outcome is GateOutcome.SUPERSEDED, "ห้าม approve ข้ามสัญญาณ"


def test_expired_ticket_reissues():
    p = evaluate_gate(**BASE, now=NOW)
    v = evaluate_gate(**BASE, pending=p.ticket.to_state(),
                      submitted_token=p.ticket.token, now=NOW + timedelta(hours=49))
    assert v.outcome is GateOutcome.EXPIRED
    assert v.ticket.token != p.ticket.token


def test_sizing_math():
    s = compute_order_sizing(200_000, 0, 1)
    assert abs(s.sleeve_thb - 40_000) < 1e-6
    assert abs(s.order_thb - 40_000 / 3) < 1e-6 and s.passes_minimum
