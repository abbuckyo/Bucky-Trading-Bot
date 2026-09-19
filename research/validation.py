"""
AlphaShield — Walk-Forward OOS & Sensitivity Sweep
==================================================
เป้าหมายเดียว: พิสูจน์ว่า CAGR 14.97% ไม่ใช่ curve-fit บน 2022

รัน:
  python -m research.validation --mode oos
  python -m research.validation --mode sweep
  python -m research.validation --mode all --out docs/validation_report.md
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from research.simulator import simulate           # event-driven, ใช้ตรรกะ SSOT เดียวกับ live
from alphashield.config import StrategyConfig, load_config

# ── WINDOWS ───────────────────────────────────────────────────
TRAIN = ("2015-01-01", "2021-12-31")
TEST  = ("2022-01-01", "2026-12-31")
FULL  = ("2015-01-01", "2026-12-31")

TRADING_DAYS = 252
RF = 0.0175          # cash park yield = risk-free proxy


# ═══════════════════════════════════════════════════════════════
#  COST MODEL — per-order ไม่ใช่ boolean
# ═══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class CostModel:
    slippage_per_order: float = 0.0005      # 0.05% ต่อ "คำสั่งจริง" 1 ใบ
    stages: int = 3

    def orders_for(self, prev_stage: int, next_stage: int) -> int:
        """0→3 = 3 คำสั่ง (ไต่วันละขั้น) | 3→0 = 1 คำสั่งล้างทีเดียว"""
        d = next_stage - prev_stage
        return abs(d) if d > 0 else (1 if d < 0 else 0)

    def drag(self, prev_stage: int, next_stage: int, sleeve_w: float) -> float:
        notional = abs(next_stage - prev_stage) / self.stages * sleeve_w
        return notional * self.slippage_per_order


# ═══════════════════════════════════════════════════════════════
#  KPI
# ═══════════════════════════════════════════════════════════════
@dataclass
class KPI:
    label: str
    start: str
    end: str
    years: float
    cagr: float
    mdd: float
    sharpe: float
    sortino: float
    calmar: float
    vol: float
    hit_rate: float
    total_orders: int
    orders_per_year: float
    cost_drag_pct: float
    final_equity: float

    def row(self) -> dict:
        return asdict(self)


def compute_kpi(equity: pd.Series, orders: int, cost_drag: float, label: str) -> KPI:
    equity = equity.dropna()
    if len(equity) < 2:
        raise ValueError(f"{label}: equity curve สั้นเกินไป")

    ret = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0]

    cagr = total ** (1 / years) - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    dd = equity / equity.cummax() - 1.0
    mdd = float(dd.min())

    excess = ret - (1 + RF) ** (1 / TRADING_DAYS) + 1
    sharpe = (excess.mean() / ret.std() * np.sqrt(TRADING_DAYS)) if ret.std() > 0 else 0.0
    downside = ret[ret < 0].std()
    sortino = (excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0.0

    return KPI(
        label=label,
        start=str(equity.index[0].date()), end=str(equity.index[-1].date()),
        years=round(years, 2),
        cagr=round(cagr, 4), mdd=round(mdd, 4),
        sharpe=round(float(sharpe), 3), sortino=round(float(sortino), 3),
        calmar=round(cagr / abs(mdd), 3) if mdd else 0.0,
        vol=round(float(vol), 4),
        hit_rate=round(float((ret > 0).mean()), 4),
        total_orders=int(orders),
        orders_per_year=round(orders / years, 1),
        cost_drag_pct=round(cost_drag * 100, 3),
        final_equity=round(float(total), 4),
    )


# ═══════════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════════
def run_window(prices: dict, cfg: StrategyConfig, window, label: str,
               costs: CostModel = CostModel()) -> KPI:
    start, end = window
    sliced = {k: df.loc[start:end] for k, df in prices.items()}
    res = simulate(sliced, cfg, cost_model=costs)   # ต้องคืน .equity / .order_count / .cost_drag
    return compute_kpi(res.equity, res.order_count, res.cost_drag, label)


def run_oos(prices: dict, cfg: StrategyConfig) -> pd.DataFrame:
    rows = [
        run_window(prices, cfg, TRAIN, "IN-SAMPLE 2015–2021").row(),
        run_window(prices, cfg, TEST,  "OUT-OF-SAMPLE 2022–2026").row(),
        run_window(prices, cfg, FULL,  "FULL 2015–2026").row(),
    ]
    df = pd.DataFrame(rows)

    is_, oos = df.iloc[0], df.iloc[1]
    df.attrs["degradation"] = {
        "cagr_delta_pp":   round((oos.cagr - is_.cagr) * 100, 2),
        "sharpe_ratio_oos_over_is": round(oos.sharpe / is_.sharpe, 3) if is_.sharpe else None,
        "mdd_worse_by_pp": round((abs(oos.mdd) - abs(is_.mdd)) * 100, 2),
        # เกณฑ์ตัดสิน: OOS Sharpe ต้อง >= 60% ของ IS จึงถือว่าไม่ overfit
        "verdict": "PASS" if (is_.sharpe and oos.sharpe / is_.sharpe >= 0.60) else "FAIL — สงสัย overfit",
    }
    return df


# ═══════════════════════════════════════════════════════════════
#  SENSITIVITY SWEEP ±30%
# ═══════════════════════════════════════════════════════════════
def _sweep_values(base: float, pct: float = 0.30, n: int = 5) -> list:
    return [round(base * (1 + s), 4) for s in np.linspace(-pct, pct, n)]


TIP_WEIGHT_VARIANTS = {
    "13612_equal":   (1 / 4, 1 / 4, 1 / 4, 1 / 4),
    "13612_weighted": (12 / 22, 4 / 22, 3 / 22, 3 / 22),   # Keller ดั้งเดิม
    "front_heavy":   (0.40, 0.30, 0.20, 0.10),
    "back_heavy":    (0.10, 0.20, 0.30, 0.40),
    "12mo_only":     (0.0, 0.0, 0.0, 1.0),
}


def run_sweep(prices: dict, cfg: StrategyConfig, window=TEST) -> pd.DataFrame:
    """สวีปเฉพาะบน OOS window — สวีปบน train = หลอกตัวเอง"""
    rows = []

    # 1) Futures guard thresholds (grid 5×5 = 25)
    for es, nq in itertools.product(
        _sweep_values(cfg.futures_es_limit),     # -1.2%
        _sweep_values(cfg.futures_nq_limit),     # -1.5%
    ):
        c = replace(cfg, futures_es_limit=es, futures_nq_limit=nq)
        k = run_window(prices, c, window, f"ES{es:+.3f}/NQ{nq:+.3f}")
        rows.append({**k.row(), "param_group": "futures_guard",
                     "es_limit": es, "nq_limit": nq})

    # 2) TIP momentum weighting (5 variants)
    for name, w in TIP_WEIGHT_VARIANTS.items():
        c = replace(cfg, tip_momentum_weights=w)
        k = run_window(prices, c, window, f"TIP:{name}")
        rows.append({**k.row(), "param_group": "tip_weights", "variant": name})

    # 3) Hysteresis band ±30%
    for buy in _sweep_values(cfg.buy_threshold):
        for sell in _sweep_values(cfg.sell_threshold):
            if sell >= buy:
                continue
            c = replace(cfg, buy_threshold=buy, sell_threshold=sell)
            k = run_window(prices, c, window, f"BUY{buy:.0f}/SELL{sell:.0f}")
            rows.append({**k.row(), "param_group": "hysteresis",
                         "buy": buy, "sell": sell})

    # 4) Slippage stress ×1 / ×2 / ×3
    for mult in (1, 2, 3):
        cm = CostModel(slippage_per_order=0.0005 * mult)
        k = run_window(prices, cfg, window, f"slippage×{mult}", costs=cm)
        rows.append({**k.row(), "param_group": "slippage", "multiplier": mult})

    df = pd.DataFrame(rows)
    df.attrs["robustness"] = {
        g: {
            "sharpe_median": round(float(sub.sharpe.median()), 3),
            "sharpe_min": round(float(sub.sharpe.min()), 3),
            "sharpe_p05": round(float(sub.sharpe.quantile(0.05)), 3),
            "cagr_spread_pp": round(float((sub.cagr.max() - sub.cagr.min()) * 100), 2),
            # เปราะ = ค่าต่ำสุดหลุดไปต่ำกว่าครึ่งหนึ่งของ median
            "verdict": "ROBUST" if sub.sharpe.min() >= sub.sharpe.median() * 0.5 else "FRAGILE",
        }
        for g, sub in df.groupby("param_group")
    }
    return df


# ═══════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════
def render_report(oos: pd.DataFrame, sweep: Optional[pd.DataFrame]) -> str:
    cols = ["label", "cagr", "mdd", "sharpe", "sortino", "calmar",
            "orders_per_year", "cost_drag_pct"]
    out = ["# AlphaShield V8.1 — Validation Report", "",
           "## 1. Walk-Forward OOS", "",
           oos[cols].to_markdown(index=False), "",
           "```json", json.dumps(oos.attrs["degradation"], indent=2, ensure_ascii=False), "```", ""]
    if sweep is not None:
        out += ["## 2. Sensitivity Sweep (OOS window only)", "",
                "```json", json.dumps(sweep.attrs["robustness"], indent=2, ensure_ascii=False), "```", "",
                "### Worst 10 configurations", "",
                sweep.nsmallest(10, "sharpe")[["label", "param_group", "cagr", "mdd", "sharpe"]]
                    .to_markdown(index=False), ""]
    out += ["## เกณฑ์ตัดสิน", "",
            "- **OOS PASS**: `Sharpe_OOS / Sharpe_IS >= 0.60`",
            "- **ROBUST**: `min(Sharpe) >= median(Sharpe) × 0.5` ในทุก param group",
            "- ถ้า FAIL หรือ FRAGILE → ลด tranche เหลือ 2 ขั้น และ/หรือถอด futures guard ออก"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oos", "sweep", "all"], default="all")
    ap.add_argument("--cache", default="historical_data_10y.pkl")
    ap.add_argument("--config", default="config/strategy.yml")
    ap.add_argument("--out", default="docs/validation_report.md")
    a = ap.parse_args()

    prices = pd.read_pickle(a.cache)
    cfg = load_config(a.config)

    oos = run_oos(prices, cfg)
    print(oos[["label", "cagr", "mdd", "sharpe", "orders_per_year"]].to_string(index=False))
    print(json.dumps(oos.attrs["degradation"], indent=2, ensure_ascii=False))

    sweep = None
    if a.mode in ("sweep", "all"):
        sweep = run_sweep(prices, cfg)
        sweep.to_csv("docs/sensitivity_sweep.csv", index=False)
        print(json.dumps(sweep.attrs["robustness"], indent=2, ensure_ascii=False))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(render_report(oos, sweep), encoding="utf-8")
    print(f"\n→ เขียนรายงานที่ {a.out}")


if __name__ == "__main__":
    main()