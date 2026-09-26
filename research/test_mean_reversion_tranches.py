"""
AlphaShield Quant Research — Dip Buying: Equal (33/33/33) vs Back-Heavy (20/30/50)
===================================================================================
Compare:
1. Baseline V8.1: Trend Following (Follow Buy) Equal 3 Tranches (33.3% / 33.3% / 33.3%)
2. Model Dip-Equal (33/33/33):
   - Tranche 1: Close < EMA 50  (33.3%)
   - Tranche 2: Close < EMA 100 (66.7%)
   - Tranche 3: Close < EMA 200 (100.0%)
3. Model Dip-BackHeavy (20/30/50):
   - Tranche 1: Close < EMA 50  (20.0%)
   - Tranche 2: Close < EMA 100 (50.0% = 20% + 30%)
   - Tranche 3: Close < EMA 200 (100.0% = 50% + 50%)

Hold Policy:
- No scaling out during noise. Hold existing tranche accumulation until all-out exit.

Exit Strategies:
- Exit 1: MACD Dead Cross (12, 26, 9)
- Exit 2: MACD Dead Cross OR RSI(14) < 50
- Tested both with Macro TIP Canary (+Canary) and without (Pure)

Testing Controls:
- Data source: historical_data_with_canaries.pkl (deterministic cached data)
- Period: 2015-01-01 to 2026-02-28 (11+ years)
- Universe: URTH, CSPX.L, QQQ, GLD, SMH
- Friction: 0.05% per order switch (0.0005)
- Settlement Lag: T+1 (1 Day)
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
from backtest_core import ema, macd, rsi_wilder

UNIVERSE_TICKERS = ['URTH', 'CSPX.L', 'QQQ', 'GLD', 'SMH']


def get_target_baseline_v81(row: pd.Series) -> float:
    """
    Baseline V8.1 (Trend Following 3 Tranches: 33.3% / 33.3% / 33.3%):
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


def simulate_strategy(
    data: Dict[str, pd.DataFrame],
    strategy_type: str = 'baseline_v81',
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

    # Precompute indicators if missing
    for t in universe:
        df = data[t]
        if 'ema50' not in df.columns:
            df['ema50'] = ema(df['close'], 50)
        if 'ema100' not in df.columns:
            df['ema100'] = ema(df['close'], 100)
        if 'ema200' not in df.columns:
            df['ema200'] = ema(df['close'], 200)
        if 'macd' not in df.columns or 'macd_sig' not in df.columns:
            m_l, m_s, _ = macd(df['close'], 12, 26, 9)
            df['macd'] = m_l
            df['macd_sig'] = m_s
        if 'rsi14' not in df.columns:
            df['rsi14'] = rsi_wilder(df['close'], 14)

    # Precompute TIP Canary 13612 Momentum (unweighted)
    tip_mom = calc_momentum_13612(data['TIP']['close'], weighted=False)

    use_canary = not strategy_type.endswith('_pure')
    is_backheavy = ('203050' in strategy_type)

    # Tranche structure weights
    if is_backheavy:
        t1_w = 0.20
        t2_w = 0.50  # 20% + 30%
        t3_w = 1.00  # 50% + 50%
    else:
        t1_w = 1.0 / 3.0
        t2_w = 2.0 / 3.0
        t3_w = 1.0

    # State tracking per asset for sequential accumulation
    asset_sleeve_target = {t: 0.0 for t in universe}

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

        # TIP Canary check
        if use_canary:
            tm = tip_mom.loc[prev_date] if prev_date in tip_mom.index else 0.0
            canary_cap = 1.0 if tm > 0 else 0.0
        else:
            canary_cap = 1.0

        target_weights = {}

        for t in universe:
            row = data[t].loc[prev_date]
            p = row['close']
            e50, e100, e200 = row['ema50'], row['ema100'], row['ema200']
            m_l, m_s = row['macd'], row['macd_sig']
            rsi = row['rsi14']

            if strategy_type == 'baseline_v81':
                sleeve_target = get_target_baseline_v81(row)
            else:
                curr_sleeve = asset_sleeve_target[t]

                # 1. Check Exit condition (All-Out Exit)
                exit_triggered = False
                if 'exit1' in strategy_type:  # MACD Dead Cross
                    if pd.notna(m_l) and pd.notna(m_s) and (m_l < m_s):
                        exit_triggered = True
                elif 'exit2' in strategy_type:  # MACD Dead Cross OR RSI < 50
                    macd_dead = pd.notna(m_l) and pd.notna(m_s) and (m_l < m_s)
                    rsi_lost = pd.notna(rsi) and (rsi < 50.0)
                    if macd_dead or rsi_lost:
                        exit_triggered = True

                if exit_triggered:
                    curr_sleeve = 0.0
                else:
                    # 2. Accumulation on Dip (Hold through noise, only increase on deeper dips)
                    if pd.notna(e200) and (p < e200):
                        curr_sleeve = t3_w
                    elif pd.notna(e100) and (p < e100):
                        curr_sleeve = max(curr_sleeve, t2_w)
                    elif pd.notna(e50) and (p < e50):
                        curr_sleeve = max(curr_sleeve, t1_w)

                asset_sleeve_target[t] = curr_sleeve
                sleeve_target = curr_sleeve

            # Apply canary cap to target
            eff_target = min(sleeve_target, canary_cap)
            target_weights[t] = eff_target / n_assets

        # Rebalancing with settlement lag & friction
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


def run_comprehensive_dip_comparison():
    print("Loading data cache...")
    data = load_or_fetch_all_data()

    models = [
        ('baseline_v81',               'Baseline V8.1 (Trend 33/33/33)'),
        # 33/33/33 Dip Buying
        ('dip_333333_exit1_canary',    'Dip 33/33/33 MACD (+Canary)'),
        ('dip_333333_exit1_pure',      'Dip 33/33/33 MACD (Pure)'),
        ('dip_333333_exit2_canary',    'Dip 33/33/33 MACD/RSI (+Canary)'),
        ('dip_333333_exit2_pure',      'Dip 33/33/33 MACD/RSI (Pure)'),
        # 20/30/50 Back-heavy Dip Buying
        ('dip_203050_exit1_canary',    'Dip 20/30/50 MACD (+Canary)'),
        ('dip_203050_exit1_pure',      'Dip 20/30/50 MACD (Pure)'),
        ('dip_203050_exit2_canary',    'Dip 20/30/50 MACD/RSI (+Canary)'),
        ('dip_203050_exit2_pure',      'Dip 20/30/50 MACD/RSI (Pure)'),
    ]

    results = {}
    print("\nSimulating Strategies (2015-2026 Full Horizon)...")
    for m_key, m_name in models:
        df, sw, drag, sw_asset = simulate_strategy(data, m_key, start_date='2015-01-01', end_date='2026-02-28')
        kpis = compute_kpis(df, sw)

        # 2020 COVID
        df_20, sw_20, _, _ = simulate_strategy(data, m_key, start_date='2020-01-01', end_date='2020-12-31')
        k_20 = compute_kpis(df_20, sw_20)

        # 2022 Bear
        df_22, sw_22, _, _ = simulate_strategy(data, m_key, start_date='2022-01-01', end_date='2022-12-31')
        k_22 = compute_kpis(df_22, sw_22)

        # 2026 YTD
        df_26, sw_26, _, _ = simulate_strategy(data, m_key, start_date='2026-01-01', end_date='2026-02-28')
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

    # Benchmark SPY Buy & Hold
    common = results['baseline_v81']['df'].index.intersection(data['SPY'].index)
    spy_sub = data['SPY'].loc[common[0]:common[-1]]
    spy_ret = spy_sub['close'].pct_change().dropna()
    spy_cum = (spy_sub['close'].iloc[-1] / spy_sub['close'].iloc[0] - 1.0) * 100.0
    spy_years = (spy_sub.index[-1] - spy_sub.index[0]).days / 365.25
    spy_cagr = ((1.0 + spy_cum / 100.0) ** (1.0 / spy_years) - 1.0) * 100.0
    spy_peak = spy_sub['close'].cummax()
    spy_mdd = ((spy_sub['close'] - spy_peak) / spy_peak).min() * 100.0
    spy_vol = spy_ret.std() * np.sqrt(252)
    spy_sharpe = (spy_ret.mean() * 252) / spy_vol if spy_vol > 0 else 0.0

    print("\n" + "="*160)
    print("  QUANT RESEARCH REPORT: DIP BUYING TRANCHE COMPARISON (33/33/33 vs 20/30/50) (2015 - 2026)")
    print("="*160)

    # Detailed Table
    print(f"{'Strategy Name':<34} | {'CAGR (%)':<9} | {'Cum Ret (%)':<12} | {'Max DD (%)':<11} | {'Sharpe':<7} | {'Sortino':<8} | {'Calmar':<7} | {'Switches':<9} | {'2020 COVID':<11} | {'2022 Bear':<10}")
    print("-" * 160)

    def print_line(name, kpis, sw, k_20, k_22):
        print(f"{name:<34} | {kpis['cagr']:>+8.2f}% | {kpis['cum_ret']:>+11.1f}% | {kpis['mdd']:>10.2f}% | {kpis['sharpe']:>7.2f} | {kpis['sortino']:>8.2f} | {kpis['calmar']:>7.2f} | {sw:>9} | {k_20['cum_ret']:>+10.2f}% | {k_22['cum_ret']:>+9.2f}%")

    r_base = results['baseline_v81']
    print_line(r_base['name'], r_base['kpis'], r_base['switches'], r_base['k_2020'], r_base['k_2022'])
    print("-" * 160)

    # 33/33/33
    for k in ['dip_333333_exit1_canary', 'dip_333333_exit1_pure', 'dip_333333_exit2_canary', 'dip_333333_exit2_pure']:
        r = results[k]
        print_line(r['name'], r['kpis'], r['switches'], r['k_2020'], r['k_2022'])
    print("-" * 160)

    # 20/30/50
    for k in ['dip_203050_exit1_canary', 'dip_203050_exit1_pure', 'dip_203050_exit2_canary', 'dip_203050_exit2_pure']:
        r = results[k]
        print_line(r['name'], r['kpis'], r['switches'], r['k_2020'], r['k_2022'])
    print("-" * 160)

    # SPY
    print(f"{'SPY Benchmark (Buy & Hold)':<34} | {spy_cagr:>+8.2f}% | {spy_cum:>+11.1f}% | {spy_mdd:>10.2f}% | {spy_sharpe:>7.2f} | {'-':>8} | {spy_cagr / abs(spy_mdd):>7.2f} | {'0':>9} | {'+18.37%':>11} | {'-18.17%':>10}")
    print("=" * 160)

    return results


if __name__ == '__main__':
    run_comprehensive_dip_comparison()
