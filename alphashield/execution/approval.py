"""
AlphaShield V8.1 — Manual Confirmation Gate
===========================================
Serverless two-phase commit สำหรับคำสั่ง SWITCH IN ครั้งแรก
Phase 1: สร้าง ticket + alert  (ไม่เขียน stage)
Phase 2: token ตรง + signal ยังเหมือนเดิม → commit + auto-unlock
"""
from __future__ import annotations

import hashlib
import hmac
import math
import secrets
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Optional

ICT = timezone(timedelta(hours=7))

SCB_MIN_SWITCH_THB = 1_000.0
SLEEVE_WEIGHT = 0.20
TRANCHE_EXPOSURE_MAP = {0: 0.0, 1: 1 / 3, 2: 2 / 3, 3: 1.0}
TICKET_TTL_HOURS = 48          # หมดอายุ = สัญญาณเก่าเกินไป ต้องออกใบใหม่
TOKEN_BYTES = 4                # 8 hex chars — พิมพ์จากมือถือได้


class GateOutcome(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"     # auto mode หรือไม่ใช่ SWITCH IN
    PENDING = "PENDING"               # รอมนุษย์อนุมัติ
    APPROVED = "APPROVED"             # token ตรง → เขียน stage ได้
    REJECTED_SIZE = "REJECTED_SIZE"   # ไม้ต่ำกว่าขั้นต่ำ SCB
    EXPIRED = "EXPIRED"               # ticket หมดอายุ
    SUPERSEDED = "SUPERSEDED"         # สัญญาณเปลี่ยนไปแล้ว ticket เก่าใช้ไม่ได้


# ─────────────────────────────────────────────────────────────
# SIZING
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class OrderSizing:
    capital_thb: float
    sleeve_thb: float
    delta_exposure: float
    order_thb: float
    target_holding_thb: float
    passes_minimum: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_order_sizing(
    capital_thb: float,
    prev_stage: int,
    next_stage: int,
    *,
    sleeve_weight: float = SLEEVE_WEIGHT,
    minimum_thb: float = SCB_MIN_SWITCH_THB,
) -> OrderSizing:
    """
    คำนวณขนาดคำสั่งจริงเป็นบาท + เช็กขั้นต่ำ SCB
    de-risk (next < prev) ผ่านเสมอ — การขายห้ามถูกบล็อกด้วยเกณฑ์ขั้นต่ำ
    """
    sleeve = float(capital_thb) * sleeve_weight
    delta = TRANCHE_EXPOSURE_MAP[next_stage] - TRANCHE_EXPOSURE_MAP[prev_stage]
    order = abs(delta) * sleeve
    target = TRANCHE_EXPOSURE_MAP[next_stage] * sleeve

    if delta <= 0:
        return OrderSizing(capital_thb, sleeve, delta, order, target, True,
                           "de-risk / liquidation — ยกเว้นเกณฑ์ขั้นต่ำเสมอ")
    if order < minimum_thb:
        return OrderSizing(
            capital_thb, sleeve, delta, order, target, False,
            f"ไม้ {order:,.0f} บาท < ขั้นต่ำ {minimum_thb:,.0f} บาท "
            f"— ไม่ขยับ stage เพื่อกัน state desync",
        )
    return OrderSizing(capital_thb, sleeve, delta, order, target, True,
                       f"ผ่านเกณฑ์ขั้นต่ำ ({order:,.0f} ≥ {minimum_thb:,.0f} บาท)")


# ─────────────────────────────────────────────────────────────
# TICKET
# ─────────────────────────────────────────────────────────────
def _signal_fingerprint(fund: str, prev_stage: int, next_stage: int, signal: str) -> str:
    """ลายนิ้วมือสัญญาณ — ถ้าเปลี่ยน ticket เก่าใช้ไม่ได้ (กัน approve ย้อนหลังผิดตัว)"""
    raw = f"{fund}|{prev_stage}->{next_stage}|{signal}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class ApprovalTicket:
    token: str
    fund: str
    prev_stage: int
    next_stage: int
    signal: str
    fingerprint: str
    issued_ict: str
    expires_ict: str
    sizing: dict

    def to_state(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_state(d: Optional[Mapping[str, Any]]) -> Optional["ApprovalTicket"]:
        if not d or not d.get("token"):
            return None
        try:
            return ApprovalTicket(**{k: d[k] for k in ApprovalTicket.__annotations__})
        except (KeyError, TypeError):
            return None

    def is_expired(self, now: datetime) -> bool:
        try:
            return now > datetime.fromisoformat(self.expires_ict)
        except ValueError:
            return True


def issue_ticket(fund, prev_stage, next_stage, signal, sizing, *, now=None) -> ApprovalTicket:
    now = now or datetime.now(ICT)
    return ApprovalTicket(
        token=secrets.token_hex(TOKEN_BYTES).upper(),
        fund=fund, prev_stage=prev_stage, next_stage=next_stage, signal=signal,
        fingerprint=_signal_fingerprint(fund, prev_stage, next_stage, signal),
        issued_ict=now.isoformat(timespec="seconds"),
        expires_ict=(now + timedelta(hours=TICKET_TTL_HOURS)).isoformat(timespec="seconds"),
        sizing=sizing.to_dict(),
    )


# ─────────────────────────────────────────────────────────────
# GATE
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GateVerdict:
    outcome: GateOutcome
    committed_stage: int          # stage ที่อนุญาตให้เขียนลง state จริง
    ticket: Optional[ApprovalTicket]
    sizing: Optional[OrderSizing]
    unlock_auto: bool             # True = ปลด require_manual_confirm หลังรอบนี้
    alert_level: str
    reasons: tuple

    @property
    def emit_order(self) -> bool:
        return self.outcome in (GateOutcome.APPROVED, GateOutcome.NOT_REQUIRED)


def evaluate_gate(
    *,
    fund: str,
    prev_stage: int,
    proposed_stage: int,
    signal: str,
    capital_thb: float,
    require_manual_confirm: bool,
    pending: Optional[Mapping[str, Any]] = None,
    submitted_token: str = "",
    now: Optional[datetime] = None,
) -> GateVerdict:
    now = now or datetime.now(ICT)
    sizing = compute_order_sizing(capital_thb, prev_stage, proposed_stage)
    existing = ApprovalTicket.from_state(pending)

    def V(outcome, stage, *reasons, ticket=None, unlock=False, alert="warn"):
        return GateVerdict(outcome, stage, ticket, sizing, unlock, alert, tuple(reasons))

    # ── de-risk / no-change: ไม่ต้องอนุมัติ ห้ามบล็อกเด็ดขาด ───────
    if proposed_stage <= prev_stage:
        return V(GateOutcome.NOT_REQUIRED, proposed_stage,
                 "ลดไม้/คงสถานะ — ไม่ต้องขออนุมัติ", alert="none")

    # ── เกณฑ์ขั้นต่ำ SCB: ไม่ผ่าน = ไม่ขยับ stage (กัน state desync) ─
    if not sizing.passes_minimum:
        return V(GateOutcome.REJECTED_SIZE, prev_stage, sizing.reason)

    # ── auto mode ────────────────────────────────────────────────
    if not require_manual_confirm:
        return V(GateOutcome.NOT_REQUIRED, proposed_stage,
                 "โหมดอัตโนมัติ (ปลดล็อกแล้ว)", alert="none")

    fp = _signal_fingerprint(fund, prev_stage, proposed_stage, signal)

    # ── Phase 2: ตรวจ ticket เดิม ────────────────────────────────
    if existing is not None:
        if existing.is_expired(now):
            new_t = issue_ticket(fund, prev_stage, proposed_stage, signal, sizing, now=now)
            return V(GateOutcome.EXPIRED, prev_stage,
                     f"ticket {existing.token} หมดอายุ — ออกใบใหม่ {new_t.token}",
                     ticket=new_t, alert="critical")

        if existing.fingerprint != fp:
            new_t = issue_ticket(fund, prev_stage, proposed_stage, signal, sizing, now=now)
            return V(GateOutcome.SUPERSEDED, prev_stage,
                     f"สัญญาณเปลี่ยน ({existing.prev_stage}→{existing.next_stage} "
                     f"เป็น {prev_stage}→{proposed_stage}) — ออกใบใหม่ {new_t.token}",
                     ticket=new_t, alert="critical")

        if submitted_token and hmac.compare_digest(
            submitted_token.strip().upper(), existing.token
        ):
            return V(GateOutcome.APPROVED, proposed_stage,
                     f"✅ อนุมัติด้วย token {existing.token} — ส่งคำสั่ง "
                     f"{sizing.order_thb:,.0f} บาท",
                     ticket=None, unlock=True, alert="none")

        return V(GateOutcome.PENDING, prev_stage,
                 f"⏳ รออนุมัติ token {existing.token} "
                 f"(หมดอายุ {existing.expires_ict})",
                 ticket=existing, alert="critical")

    # ── Phase 1: ออก ticket ใหม่ ─────────────────────────────────
    t = issue_ticket(fund, prev_stage, proposed_stage, signal, sizing, now=now)
    return V(GateOutcome.PENDING, prev_stage,
             f"🔔 คำสั่ง SWITCH IN ครั้งแรกของ V8 — รอการยืนยันจากมนุษย์",
             ticket=t, alert="critical")


# ─────────────────────────────────────────────────────────────
# ALERT RENDERER
# ─────────────────────────────────────────────────────────────
def render_approval_alert(v: GateVerdict, *, repo_url: str = "") -> str:
    t, s = v.ticket, v.sizing
    if t is None or s is None:
        return ""
    return "\n".join([
        "🔐 **ALPHASHIELD — ต้องการการยืนยันคำสั่ง**",
        "━━━━━━━━━━━━━━━━━━━━",
        f"📌 กองทุน : `{t.fund}`",
        f"🪜 บันได  : Tranche {t.prev_stage} → **{t.next_stage}** ({t.signal})",
        f"💰 เงินต้น : {s.capital_thb:,.0f} บาท",
        f"🧮 Sleeve  : {s.sleeve_thb:,.0f} บาท (20% ต่อสินทรัพย์)",
        f"➡️ **ไม้นี้สั่งจริง : {s.order_thb:,.0f} บาท**",
        f"🎯 ถือรวมหลังคำสั่ง : {s.target_holding_thb:,.0f} บาท",
        f"✔️ เกณฑ์ขั้นต่ำ SCB : {'ผ่าน' if s.passes_minimum else '❌ ไม่ผ่าน'} "
        f"({s.reason})",
        "━━━━━━━━━━━━━━━━━━━━",
        f"🎟️ **TOKEN : `{t.token}`**",
        f"⏳ หมดอายุ : {t.expires_ict}",
        "",
        "**วิธีอนุมัติ:**",
        f"1. เปิด {repo_url or 'GitHub'} → Actions → AlphaShield Live Runner",
        "2. กด **Run workflow**",
        f"3. ใส่ `approve_token` = `{t.token}` แล้วกด Run",
        "",
        "⚠️ ยังไม่มีการเขียน state และยังไม่มีคำสั่งใด ๆ ถูกส่ง",
        "⚠️ เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน",
    ])
