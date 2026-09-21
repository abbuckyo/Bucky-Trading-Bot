"""
AlphaShield Quant Research — Tranche Structure Comparison
==========================================================
Compare:
- Model A (V8.1 Baseline): Equal 3 Tranches (33.3% / 33.3% / 33.3%)
- Model B (Pyramid 4 Tranches): 10% / 20% / 30% / 40%
- Model C (Back-heavy / High Conviction): 20% / 30% / 50%
  * Tranche 1 (Price > EMA 50): 20%
  * Tranche 2 (Price > EMA 100 + Bull Stack): 50% (20% + 30%)
  * Tranche 3 (Price > EMA 200 + Full Bull Stack): 100% (50% + 50%)
- Model D (Front-balanced): 30% / 30% / 40%
  * Tranche 1 (Price > EMA 50): 30%
  * Tranche 2 (Price > EMA 100 + Bull Stack): 60% (30% + 30%)
  * Tranche 3 (Price > EMA 200 + Full Bull Stack): 100% (60% + 40%)

Period: 11 Years (2015-01-01 to 2026-02-28)
Slippage & Friction: 0.05% per order (switching_friction = 0.0005)
Settlement Lag: 1 Day
Canary: HAA TIP Canary (13612 Momentum)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from run_paper_experiments import load_or_fetch_all_data, calc_momentum_13612, compute_kpis
from backtest_core import ema

UNIVERSE_TICKERS = ['URTH', 'CSPX.L', 'QQQ', 'GLD', 'SMH']

def get_target_model_a(row: pd.Series) -> float:
    """
    Model A (V8.1 Standard Equal 3 Tranches: 33.3% / 33.3% / 33.3%):
    - Tranche 3 (100%): Price > EMA 200 and EMA 50 > EMA 100 > EMA 200 (Full Bull Stack)
    - Tranche 2 (66.6%): Price > EMA 100 and EMA 50 > EMA 100 (Bull Stack)
    - Tranche 1 (33.3%): Price > EMA 50
    - Hard breakdown: Price < EMA 200 by -2% -> 0.0
    """
    p = row['close']
    e50, e100, e200 = row['ema50'], row['ema100'], row['ema200']

    if pd.isna(e50) or pd.isna(e100) or pd.isna(e200):
        return 0.0

    dist_ema200 = (p / e200 - 1.0) * 100.0 if e200 > 0 else 0.0
    if dist_ema200 < -2.0:
        return 0.0

    if p > e200 and (e50 > e100 > e200):
        return 1.0
    if p > e100 and (e50 > e100):
        return 2.0 / 3.0
    if p > e50:
        return 1.0 / 3.0

    return 0.0


def get_target_model_b(row: pd.Series) -> float:
    """
    Model B (Pyramid Accumulation 4 Tranches: 10% / 20% / 30% / 40%):
    - Tranche 1 (10%): Price > Daily EMA 50
    - Tranche 2 (20%): Price > Daily EMA 100
    - Tranche 3 (30%): Price > Daily EMA 200
    - Tranche 4 (40%): Price > Weekly EMA 50 (Daily EMA 250)
    """
    p = row['close']
    e50, e100, e200, e250 = row['ema50'], row['ema100'], row['ema200'], row['ema250']

    if pd.isna(e50) or pd.isna(e100) or pd.isna(e200) or pd.isna(e250):
        return 0.0

    dist_ema200 = (p / e200 - 1.0) * 100.0 if e200 > 0 else 0.0
    if dist_ema200 < -2.0:
        return 0.0

    exp = 0.0
    if p > e50:
        exp += 0.10
    if p > e100:
        exp += 0.20
    if p > e200:
        exp += 0.30
    if p > e250:
        exp += 0.40

    return min(1.0, exp)


def get_target_model_c(row: pd.Series) -> float:
    """
    Model C (Back-heavy / High Conviction: 20% / 30% / 50%):
    - Tranche 3 (100%): Price > EMA 200 and EMA 50 > EMA 100 > EMA 200 (Full Bull Stack)
    - Tranche 2 (50%):  Price > EMA 100 and EMA 50 > EMA 100 (Bull Stack)
    - Tranche 1 (20%):  Price > EMA 50
    - Hard breakdown: Price < EMA 200 by -2% -> 0.0
    """
    p = row['close']
    e50, e100, e200 = row['ema50'], row['ema100'], row['ema200']

    if pd.isna(e50) or pd.isna(e100) or pd.isna(e200):
        return 0.0

    dist_ema200 = (p / e200 - 1.0) * 100.0 if e200 > 0 else 0.0
    if dist_ema200 < -2.0:
        return 0.0

    if p > e200 and (e50 > e100 > e200):
        return 1.0
    if p > e100 and (e50 > e100):
        return 0.50
    if p > e50:
        return 0.20

    return 0.0


def get_target_model_d(row: pd.Series) -> float:
    """
    Model D (Front-balanced: 30% / 30% / 40%):
    - Tranche 3 (100%): Price > EMA 200 and EMA 50 > EMA 100 > EMA 200 (Full Bull Stack)
    - Tranche 2 (60%):  Price > EMA 100 and EMA 50 > EMA 100 (Bull Stack)
    - Tranche 1 (30%):  Price > EMA 50
    - Hard breakdown: Price < EMA 200 by -2% -> 0.0
    """
    p = row['close']
    e50, e100, e200 = row['ema50'], row['ema100'], row['ema200']

    if pd.isna(e50) or pd.isna(e100) or pd.isna(e200):
        return 0.0

    dist_ema200 = (p / e200 - 1.0) * 100.0 if e200 > 0 else 0.0
    if dist_ema200 < -2.0:
        return 0.0

    if p > e200 and (e50 > e100 > e200):
        return 1.0
    if p > e100 and (e50 > e100):
        return 0.60
    if p > e50:
        return 0.30

    return 0.0


def simulate_model(
    data: Dict[str, pd.DataFrame],
    model_type: str = 'model_a',
    start_date: str = '2015-01-01',
    end_date: str = '2026-02-28',
    settlement_lag: int = 1,
    cash_yield_annual: float = 0.0175,
    switching_friction: float = 0.0005,
) -> Tuple[pd.DataFrame, int, float, Dict[str, int]]:
    universe = UNIVERSE_TICKERS
    all_needed = list(set(universe + ['TIP']))
    all_dates = sorted(list(set.intersection(*[set(data[t].index) for t in all_needed])))
    all_dates = [d for d in all_dates if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)]

    if len(all_dates) < 2:
        return pd.DataFrame(), 0, 0.0, {}

    # Precompute TIP Canary 13612 Momentum (unweighted)
    tip_mom = calc_momentum_13612(data['TIP']['close'], weighted=False)

    # Precalculate indicators if needed
    for t in universe:
        if 'ema250' not in data[t].columns:
            data[t]['ema250'] = ema(data[t]['close'], 250)

    # Daily target exposure per asset
    daily_targets = {t: {} for t in universe}
    for t in universe:
        df = data[t]
        for d in all_dates:
            row = df.loc[d]
            if model_type == 'model_a':
                target = get_target_model_a(row)
            elif model_type == 'model_b':
                target = get_target_model_b(row)
            elif model_type == 'model_c':
                target = get_target_model_c(row)
            elif model_type == 'model_d':
                target = get_target_model_d(row)
            else:
                target = 0.0
            daily_targets[t][d] = target

    current_weights = {t: 0.0 for t in universe}
    pending_increases = {t: [] for t in universe}

    portfolio_value = 100000.0
    daily_records = []
    total_switches = 0
    switches_per_asset = {t: 0 for t in universe}
    total_friction_drag = 0.0

    daily_cash_rate = (1.0 + cash_yield_annual) ** (1.0 / 252.0) - 1.0
    n_assets = len(universe)

    for i in range(1, len(all_dates)):
        prev_date = all_dates[i - 1]
        curr_date = all_dates[i]

        # Check TIP canary at t-1
        tm = tip_mom.loc[prev_date] if prev_date in tip_mom.index else 0.0
        canary_cap = 1.0 if tm > 0 else 0.0

        # Target portfolio weights (each sleeve max 20% of portfolio)
        target_weights = {}
        for t in universe:
            tgt = min(daily_targets[t][prev_date], canary_cap)
            target_weights[t] = tgt / n_assets

        # Rebalance logic with settlement lag & friction
        for t in universe:
            tgt_w = target_weights[t]
            cur_w = current_weights[t]

            # Decrease: Instant exit / trim
            if tgt_w < cur_w:
                delta = cur_w - tgt_w
                current_weights[t] = tgt_w
                pending_increases[t].clear()
                total_switches += 1
                switches_per_asset[t] += 1
                drag = portfolio_value * delta * switching_friction
                portfolio_value -= drag
                total_friction_drag += drag

            # Increase: Sized entry with settlement lag
            elif tgt_w > cur_w:
                if settlement_lag > 0:
                    pending_increases[t] = [(settlement_lag, tgt_w)]
                else:
                    delta = tgt_w - cur_w
                    current_weights[t] = tgt_w
                    total_switches += 1
                    switches_per_asset[t] += 1
                    drag = portfolio_value * delta * switching_friction
                    portfolio_value -= drag
                    total_friction_drag += drag

            # Process pending increases
            if pending_increases[t]:
                rem_lag, dest_w = pending_increases[t][0]
                rem_lag -= 1
                if rem_lag <= 0:
                    delta = max(0.0, dest_w - current_weights[t])
                    current_weights[t] = dest_w
                    total_switches += 1
                    switches_per_asset[t] += 1
                    drag = portfolio_value * delta * switching_friction
                    portfolio_value -= drag
                    total_friction_drag += drag
                    pending_increases[t].clear()
                else:
                    pending_increases[t] = [(rem_lag, dest_w)]

        # Day t portfolio return
        total_inv_w = sum(current_weights.values())
        cash_w = max(0.0, 1.0 - total_inv_w)

        day_return = cash_w * daily_cash_rate
        for t in universe:
            w = current_weights[t]
            if w > 0:
                p_prev = data[t].loc[prev_date, 'close']
                p_curr = data[t].loc[curr_date, 'close']
                day_return += w * ((p_curr / p_prev) - 1.0)

        portfolio_value *= (1.0 + day_return)

        daily_records.append({
            'date': curr_date,
            'portfolio_value': portfolio_value,
            'day_return': day_return,
            'invested_weight': total_inv_w,
            'cash_weight': cash_w
        })

    res_df = pd.DataFrame(daily_records).set_index('date')
    return res_df, total_switches, total_friction_drag, switches_per_asset


def run_full_comparison():
    print("Loading data cache...")
    data = load_or_fetch_all_data()

    models = [
        ('model_a', 'Model A (33/33/33 V8.1)'),
        ('model_c', 'Model C (20/30/50 Back)'),
        ('model_d', 'Model D (30/30/40 Front)'),
    ]

    results = {}
    print("\nSimulating 3-Tranche Variations (2015-2026 Full Horizon)...")
    for m_key, m_name in models:
        df, sw, drag, sw_asset = simulate_model(data, m_key, start_date='2015-01-01', end_date='2026-02-28')
        kpis = compute_kpis(df, sw)

        # 2020 COVID
        df_20, sw_20, _, _ = simulate_model(data, m_key, start_date='2020-01-01', end_date='2020-12-31')
        k_20 = compute_kpis(df_20, sw_20)

        # 2022 Bear
        df_22, sw_22, _, _ = simulate_model(data, m_key, start_date='2022-01-01', end_date='2022-12-31')
        k_22 = compute_kpis(df_22, sw_22)

        # 2026 YTD
        df_26, sw_26, _, _ = simulate_model(data, m_key, start_date='2026-01-01', end_date='2026-02-28')
        k_26 = compute_kpis(df_26, sw_26)

        results[m_key] = {
            'name': m_name,
            'df': df,
            'kpis': kpis,
            'switches': sw,
            'drag': drag,
            'k_2020': k_20,
            'k_2022': k_22,
            'k_2026': k_26
        }

    # Benchmark SPY
    common = results['model_a']['df'].index.intersection(data['SPY'].index)
    spy_sub = data['SPY'].loc[common[0]:common[-1]]
    spy_ret = spy_sub['close'].pct_change().dropna()
    spy_cum = (spy_sub['close'].iloc[-1] / spy_sub['close'].iloc[0] - 1.0) * 100.0
    spy_years = (spy_sub.index[-1] - spy_sub.index[0]).days / 365.25
    spy_cagr = ((1.0 + spy_cum / 100.0) ** (1.0 / spy_years) - 1.0) * 100.0
    spy_peak = spy_sub['close'].cummax()
    spy_mdd = ((spy_sub['close'] - spy_peak) / spy_peak).min() * 100.0
    spy_vol = spy_ret.std() * np.sqrt(252)
    spy_sharpe = (spy_ret.mean() * 252) / spy_vol if spy_vol > 0 else 0.0

    print("\n" + "="*116)
    print("  QUANT RESEARCH REPORT: 3-TRANCHE WEIGHTING COMPARISON (2015 - 2026)")
    print("="*116)

    headers = [
        "Metric",
        "Model A (33/33/33)",
        "Model C (20/30/50)",
        "Model D (30/30/40)",
        "SPY Benchmark"
    ]

    r_a = results['model_a']
    r_c = results['model_c']
    r_d = results['model_d']

    rows = [
        ("CAGR (%)", f"+{r_a['kpis']['cagr']:.2f}%", f"+{r_c['kpis']['cagr']:.2f}%", f"+{r_d['kpis']['cagr']:.2f}%", f"+{spy_cagr:.2f}%"),
        ("Cumulative Return (%)", f"+{r_a['kpis']['cum_ret']:.1f}%", f"+{r_c['kpis']['cum_ret']:.1f}%", f"+{r_d['kpis']['cum_ret']:.1f}%", f"+{spy_cum:.1f}%"),
        ("Max Drawdown (%)", f"{r_a['kpis']['mdd']:.2f}%", f"{r_c['kpis']['mdd']:.2f}%", f"{r_d['kpis']['mdd']:.2f}%", f"{spy_mdd:.2f}%"),
        ("Sharpe Ratio", f"{r_a['kpis']['sharpe']:.2f}", f"{r_c['kpis']['sharpe']:.2f}", f"{r_d['kpis']['sharpe']:.2f}", f"{spy_sharpe:.2f}"),
        ("Sortino Ratio", f"{r_a['kpis']['sortino']:.2f}", f"{r_c['kpis']['sortino']:.2f}", f"{r_d['kpis']['sortino']:.2f}", "-"),
        ("Calmar Ratio", f"{r_a['kpis']['calmar']:.2f}", f"{r_c['kpis']['calmar']:.2f}", f"{r_d['kpis']['calmar']:.2f}", f"{spy_cagr / abs(spy_mdd):.2f}"),
        ("Total Order Switches", f"{r_a['switches']}", f"{r_c['switches']}", f"{r_d['switches']}", "0"),
        ("Annual Switches / Yr", f"{r_a['kpis']['annual_switches']:.1f}", f"{r_c['kpis']['annual_switches']:.1f}", f"{r_d['kpis']['annual_switches']:.1f}", "0"),
        ("Cumulative Friction Drag", f"THB {r_a['drag']:,.0f}", f"THB {r_c['drag']:,.0f}", f"THB {r_d['drag']:,.0f}", "THB 0"),
        ("2020 Return (COVID)", f"+{r_a['k_2020']['cum_ret']:.2f}%", f"+{r_c['k_2020']['cum_ret']:.2f}%", f"+{r_d['k_2020']['cum_ret']:.2f}%", "-"),
        ("2022 Return (Bear)", f"{r_a['k_2022']['cum_ret']:+.2f}%", f"{r_c['k_2022']['cum_ret']:+.2f}%", f"{r_d['k_2022']['cum_ret']:+.2f}%", "-"),
        ("2026 YTD Return", f"+{r_a['k_2026']['cum_ret']:.2f}%", f"+{r_c['k_2026']['cum_ret']:.2f}%", f"+{r_d['k_2026']['cum_ret']:.2f}%", "-")
    ]

    col_w = [26, 22, 22, 22, 16]
    fmt_row = "".join(f"{{:<{w}}}" for w in col_w)

    print(fmt_row.format(*headers))
    print("-" * 108)
    for r in rows:
        print(fmt_row.format(*r))
    print("=" * 108)

    return results

if __name__ == '__main__':
    run_full_comparison()
