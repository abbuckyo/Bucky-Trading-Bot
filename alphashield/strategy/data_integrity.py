"""
AlphaShield V8.1 — 2-Strike Corrupt Data Engine
===============================================
Day 1 (strike 1) : FREEZE     — คงสถานะเดิม 100%, ไม่ออกคำสั่ง, ยิง Alert แดง
Day 2 (strike 2) : FORCE EXIT — ล้างเข้า Cash Park เพื่อปกป้องเงินต้น
Data restored    : SELF-HEAL  — reset strike = 0 อัตโนมัติ

Pure logic: ไม่มี I/O, deterministic, testable 100%
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CRITICAL_FIELDS: tuple = ("price", "ema50", "ema100", "ema200", "score")
OPTIONAL_FIELDS: tuple = ("rsi", "adx", "macd_hist", "hv20", "chg_pct")

STRIKE_LIMIT: int = 2      # strike >= 2 → forced exit
STRIKE_CAP: int = 9        # กัน counter ล้นจาก state เสีย
MAX_STALE_DAYS: int = 10   # แท่งล่าสุดเก่ากว่านี้ = CORRUPT


class DataHealth(str, Enum):
    CLEAN = "CLEAN"
    DEGRADED = "DEGRADED"   # เสียเฉพาะ optional → รันต่อได้
    CORRUPT = "CORRUPT"     # critical เสีย / ข้อมูลค้าง / ไม่มีข้อมูล


class IntegrityAction(str, Enum):
    PROCEED = "PROCEED"
    FREEZE = "FREEZE"
    FORCE_EXIT = "FORCE_EXIT"


# ─────────────────────────────────────────────────────────────
# NaN GUARD
# ─────────────────────────────────────────────────────────────
def is_nan(value: Any) -> bool:
    """True ถ้าเป็น None / NaN / Inf / ตัวเลขไม่ได้ — ใช้ร่วมกันทั้งระบบ"""
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return True
    return math.isnan(f) or math.isinf(f)


def _as_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value[:10]).date()
        except ValueError:
            return None
    return None


# ─────────────────────────────────────────────────────────────
# HEALTH ASSESSMENT
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HealthReport:
    health: DataHealth
    bad_critical: tuple
    bad_optional: tuple
    stale_days: Optional[int]
    reasons: tuple

    @property
    def is_corrupt(self) -> bool:
        return self.health is DataHealth.CORRUPT


def assess_data_health(
    metrics: Optional[Mapping[str, Any]],
    *,
    last_bar_date: Any = None,
    run_date: Optional[date] = None,
    critical: Sequence[str] = CRITICAL_FIELDS,
    optional: Sequence[str] = OPTIONAL_FIELDS,
    max_stale_days: int = MAX_STALE_DAYS,
) -> HealthReport:
    reasons: list = []

    if not metrics:
        return HealthReport(DataHealth.CORRUPT, tuple(critical), (), None,
                            ("ไม่มีข้อมูล metrics เลย (source cascade ล้มครบทุกชั้น)",))

    bad_critical = tuple(f for f in critical if is_nan(metrics.get(f)))
    bad_optional = tuple(f for f in optional if is_nan(metrics.get(f)))

    if bad_critical:
        reasons.append(f"critical field เสีย: {', '.join(bad_critical)}")

    # ราคาติดลบ/ศูนย์ = ข้อมูลเพี้ยน
    price = metrics.get("price")
    if not is_nan(price) and float(price) <= 0:
        bad_critical = bad_critical + ("price<=0",)
        reasons.append(f"ราคาผิดปกติ: {price}")

    stale_days: Optional[int] = None
    bar_d, run_d = _as_date(last_bar_date), (run_date or date.today())
    if bar_d is not None:
        stale_days = (run_d - bar_d).days
        if stale_days > max_stale_days:
            bad_critical = bad_critical + ("stale_bar",)
            reasons.append(f"ข้อมูลค้าง {stale_days} วัน (เกิน {max_stale_days})")
        elif stale_days < 0:
            bad_critical = bad_critical + ("future_bar",)
            reasons.append(f"แท่งล่าสุดเป็นอนาคต ({bar_d}) — timezone/feed ผิด")

    if bad_critical:
        health = DataHealth.CORRUPT
    elif bad_optional:
        health = DataHealth.DEGRADED
        reasons.append(f"optional field เสีย: {', '.join(bad_optional)}")
    else:
        health = DataHealth.CLEAN

    return HealthReport(health, bad_critical, bad_optional, stale_days, tuple(reasons))


# ─────────────────────────────────────────────────────────────
# STRIKE LEDGER
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class IntegrityVerdict:
    health: DataHealth
    action: IntegrityAction
    strike: int
    last_date: Optional[str]
    first_seen: Optional[str]
    alert_level: str          # "none" | "warn" | "critical"
    reasons: tuple

    @property
    def blocks_entry(self) -> bool:
        return self.action is not IntegrityAction.PROCEED

    def to_state(self) -> dict:
        return {
            "corrupt_strike": self.strike,
            "corrupt_last_date": self.last_date,
            "corrupt_first_seen_ict": self.first_seen,
        }


def _sanitize_strike(raw: Any) -> int:
    """กัน state เสีย: -1, 99, "2", None → clamp เข้า [0, CAP]"""
    try:
        s = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, min(s, STRIKE_CAP))


def evaluate_integrity(
    report: HealthReport,
    *,
    run_date: date,
    prev_strike: Any = 0,
    prev_last_date: Any = None,
    prev_first_seen: Any = None,
    now_iso: Optional[str] = None,
    strike_limit: int = STRIKE_LIMIT,
) -> IntegrityVerdict:
    """
    Idempotent: รันซ้ำวันเดียวกัน strike ไม่เพิ่ม
    Self-healing: ข้อมูลกลับมาดี → strike = 0 ทันที
    """
    run_iso = run_date.isoformat()
    strike = _sanitize_strike(prev_strike)
    prev_d = _as_date(prev_last_date)

    # ── SELF-HEALING ──────────────────────────────────────────
    if not report.is_corrupt:
        healed = strike > 0
        return IntegrityVerdict(
            health=report.health,
            action=IntegrityAction.PROCEED,
            strike=0,
            last_date=None,
            first_seen=None,
            alert_level="warn" if (healed or report.health is DataHealth.DEGRADED) else "none",
            reasons=(("ข้อมูลกลับมาสมบูรณ์ — reset strike",) if healed else ()) + report.reasons,
        )

    # ── STRIKE ACCUMULATION ───────────────────────────────────
    if prev_d == run_date:
        pass                                  # รันซ้ำวันเดิม → ไม่เพิ่ม
    elif prev_d is not None and (run_date - prev_d).days < 0:
        strike = min(strike + 1, STRIKE_CAP)  # clock skew → นับแบบ conservative
    else:
        strike = min(strike + 1, STRIKE_CAP)

    first_seen = (prev_first_seen if (strike > 1 and prev_first_seen) else (now_iso or run_iso))
    action = IntegrityAction.FORCE_EXIT if strike >= strike_limit else IntegrityAction.FREEZE

    return IntegrityVerdict(
        health=DataHealth.CORRUPT,
        action=action,
        strike=strike,
        last_date=run_iso,
        first_seen=first_seen,
        alert_level="critical",
        reasons=report.reasons,
    )