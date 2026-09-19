import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from backtest_core import precompute_asset_indicators, score_bar
from run_experiments import compute_kpis, get_spy_kpis, BacktestConfig, run_simulation

CACHE_FILE = 'historical_data_with_canaries.pkl'
BASE_CACHE = 'historical_data_10y.pkl'

BASE_TICKERS = {
    'URTH': 'SCBWORLDE',
    'CSPX.L': 'SCBS&P500E',
    'QQQ': 'SCBNDQ(E)',
    'GLD': 'SCBGOLDE',
    'SMH': 'SCBSEMI(E)',
    'SPY': 'BENCHMARK_SPY'
}

CANARY_TICKERS = ['TIP', 'VWO', 'BND']
UNIVERSE_TICKERS = ['URTH', 'CSPX.L', 'QQQ', 'GLD', 'SMH']

def load_or_fetch_all_data():
    if os.path.exists(CACHE_FILE):
        print(f'Loading data from canary cache: {CACHE_FILE}')
        return pd.read_pickle(CACHE_FILE)

    data = {}
    if os.path.exists(BASE_CACHE):
        print(f'Loading base tickers from {BASE_CACHE}')
        data = pd.read_pickle(BASE_CACHE)

    needed_canaries = [t for t in CANARY_TICKERS if t not in data]
    if needed_canaries:
        print(f'Downloading canary data ({needed_canaries}) from Yahoo Finance...')
        raw = yf.download(needed_canaries, start='2013-01-01', end='2026-03-01', auto_adjust=True, group_by='ticker')
        for t in needed_canaries:
            df = raw[t].copy().dropna(subset=['Close']) if len(needed_canaries) > 1 else raw.copy().dropna(subset=['Close'])
            df.columns = [c.lower() for c in df.columns]
            df = precompute_asset_indicators(df)
            data[t] = df

    pd.to_pickle(data, CACHE_FILE)
    print(f'All tickers cached to {CACHE_FILE}')
    return data

def calc_momentum_13612(series: pd.Series, weighted: bool = False) -> pd.Series:
    """
    Momentum 13612 calculation on daily series using 21, 63, 126, 252 bars.
    - If weighted (Keller DAA): 12 * r1 + 4 * r3 + 2 * r6 + 1 * r12
    - If unweighted (Richman HAA): (r1 + r3 + r6 + r12) / 4
    """
    p = series
    r1 = (p / p.shift(21)) - 1.0
    r3 = (p / p.shift(63)) - 1.0
    r6 = (p / p.shift(126)) - 1.0
    r12 = (p / p.shift(252)) - 1.0

    if weighted:
        mom = 12.0 * r1 + 4.0 * r3 + 2.0 * r6 + 1.0 * r12
    else:
        mom = (r1 + r3 + r6 + r12) / 4.0
    return mom

def get_tranche_target(row: pd.Series, faber_filter: bool = False) -> float:
    """
    3-Tranche rules:
    - Hard Breakdown: Close < EMA200 by -2% -> 0.0
    - If faber_filter is True: If Close < EMA200 -> 0.0 (cannot open even Tranche 1)
    - ไม้ 3 (100%): Price > EMA 200 and EMA 50 > EMA 100 > EMA 200
    - ไม้ 2 (66%): Price > EMA 100 and EMA 50 > EMA 100
    - ไม้ 1 (33%): Price > EMA 50
    - Else: 0.0
    """
    p = row['close']
    e50, e100, e200 = row['ema50'], row['ema100'], row['ema200']

    if pd.isna(e50) or pd.isna(e100) or pd.isna(e200):
        return 0.0

    dist_ema200 = (p / e200 - 1.0) * 100.0 if e200 > 0 else 0.0
    if dist_ema200 < -2.0:
        return 0.0

    if faber_filter and p < e200:
        return 0.0

    if p > e200 and (e50 > e100 > e200):
        return 1.0

    if p > e100 and (e50 > e100):
        return 0.66

    if p > e50:
        return 0.33

    return 0.0

def run_paper_simulation(
    data: Dict[str, pd.DataFrame],
    exp_type: str,
    start_date: str = '2015-01-01',
    end_date: str = '2026-02-28',
    settlement_lag: int = 1,
    cash_yield_annual: float = 0.0175,
    switching_friction: float = 0.0005,
):
    universe = UNIVERSE_TICKERS
    all_needed = list(set(universe + ['TIP', 'VWO', 'BND']))
    all_dates = sorted(list(set.intersection(*[set(data[t].index) for t in all_needed])))
    all_dates = [d for d in all_dates if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)]

    if len(all_dates) < 2:
        return None, 0, {}

    # Precompute canary momentums
    vwo_mom = calc_momentum_13612(data['VWO']['close'], weighted=True)
    bnd_mom = calc_momentum_13612(data['BND']['close'], weighted=True)
    tip_mom = calc_momentum_13612(data['TIP']['close'], weighted=False)

    # Precompute scores for universe assets (used in Opus Correlation Guard)
    daily_scores = {t: {} for t in universe}
    for t in universe:
        df = data[t]
        for d in all_dates:
            s, _ = score_bar(df.loc[d])
            daily_scores[t][d] = s

    # Precalculate raw tranche targets
    faber_flag = (exp_type == 'exp3_faber')
    daily_tranches = {t: {} for t in universe}
    for t in universe:
        df = data[t]
        for d in all_dates:
            daily_tranches[t][d] = get_tranche_target(df.loc[d], faber_filter=faber_flag)

    current_weights = {t: 0.0 for t in universe}
    pending_increases = {t: [] for t in universe}

    portfolio_value = 100000.0
    daily_records = []
    total_switches = 0
    switches_per_asset = {t: 0 for t in universe}

    daily_cash_rate = (1.0 + cash_yield_annual) ** (1.0 / 252.0) - 1.0
    n_assets = len(universe)

    for i in range(1, len(all_dates)):
        prev_date = all_dates[i - 1]
        curr_date = all_dates[i]

        # Determine macro canary regime at t-1
        canary_mode = 'NORMAL'
        canary_cap = 1.0  # multiplier or exposure cap

        if exp_type == 'exp4_keller':
            vm = vwo_mom.loc[prev_date] if prev_date in vwo_mom.index else 0.0
            bm = bnd_mom.loc[prev_date] if prev_date in bnd_mom.index else 0.0
            bad_count = (1 if vm <= 0 else 0) + (1 if bm <= 0 else 0)
            if bad_count == 2:
                canary_cap = 0.0
            elif bad_count == 1:
                canary_cap = 0.33
            else:
                canary_cap = 1.0

        elif exp_type in ('exp5_richman', 'exp6_ultimate'):
            tm = tip_mom.loc[prev_date] if prev_date in tip_mom.index else 0.0
            if tm <= 0:
                canary_cap = 0.0
            else:
                canary_cap = 1.0

        # Calculate target exposure per asset
        raw_targets = {}
        for t in universe:
            tranche = daily_tranches[t][prev_date]
            # Apply canary cap
            capped_tranche = min(tranche, canary_cap)
            raw_targets[t] = capped_tranche

        # Correlation Guard in Exp 6: QQQ and SMH cannot be active simultaneously
        if exp_type == 'exp6_ultimate':
            if raw_targets['QQQ'] > 0 and raw_targets['SMH'] > 0:
                s_qqq = daily_scores['QQQ'][prev_date]
                s_smh = daily_scores['SMH'][prev_date]
                if s_smh > s_qqq:
                    raw_targets['QQQ'] = 0.0
                else:
                    raw_targets['SMH'] = 0.0

        # Convert to portfolio weights (equal 20% slice per asset)
        target_weights = {t: raw_targets[t] / n_assets for t in universe}

        # Position transitions and friction
        for t in universe:
            tgt_w = target_weights[t]
            cur_w = current_weights[t]

            if tgt_w < cur_w:
                delta = cur_w - tgt_w
                current_weights[t] = tgt_w
                pending_increases[t].clear()
                total_switches += 1
                switches_per_asset[t] += 1
                portfolio_value *= (1.0 - delta * switching_friction)
            elif tgt_w > cur_w:
                if settlement_lag > 0:
                    pending_increases[t] = [(settlement_lag, tgt_w)]
                else:
                    delta = tgt_w - cur_w
                    current_weights[t] = tgt_w
                    total_switches += 1
                    switches_per_asset[t] += 1
                    portfolio_value *= (1.0 - delta * switching_friction)

            if pending_increases[t]:
                rem_lag, dest_w = pending_increases[t][0]
                rem_lag -= 1
                if rem_lag <= 0:
                    delta = max(0.0, dest_w - current_weights[t])
                    current_weights[t] = dest_w
                    total_switches += 1
                    switches_per_asset[t] += 1
                    portfolio_value *= (1.0 - delta * switching_friction)
                    pending_increases[t].clear()
                else:
                    pending_increases[t] = [(rem_lag, dest_w)]

        # Day t returns
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
    return res_df, total_switches, switches_per_asset


def run_experiment_suite(data, exp_key, name, faber=False):
    # Full period 2015-2026
    df_full, sw_full, _ = run_paper_simulation(data, exp_key, start_date='2015-01-01', end_date='2026-02-28')
    k_full = compute_kpis(df_full, sw_full)

    # 2020 COVID
    df_2020, _, _ = run_paper_simulation(data, exp_key, start_date='2020-01-01', end_date='2020-12-31')
    k_2020 = compute_kpis(df_2020, 0)

    # 2022 Bear
    df_2022, _, _ = run_paper_simulation(data, exp_key, start_date='2022-01-01', end_date='2022-12-31')
    k_2022 = compute_kpis(df_2022, 0)

    return {
        'name': name,
        'cagr': k_full['cagr'],
        'cum_ret': k_full['cum_ret'],
        'mdd': k_full['mdd'],
        'sharpe': k_full['sharpe'],
        'sortino': k_full['sortino'],
        'calmar': k_full['calmar'],
        'ret_2020': k_2020['cum_ret'],
        'ret_2022': k_2022['cum_ret'],
        'annual_switches': k_full['annual_switches']
    }


if __name__ == '__main__':
    data = load_or_fetch_all_data()

    # Exp 1: Baseline All-in (Quant Score 75/55 + Anti-Chop + Breakdown)
    base_cfg_full = BacktestConfig(
        name='Exp 1: Baseline All-in (Score 75/55)',
        buy_threshold=75.0, sell_threshold=55.0,
        settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
        use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset',
        start_date='2015-01-01', end_date='2026-02-28'
    )
    df_base, sw_base, _ = run_simulation(data, base_cfg_full)
    k_base = compute_kpis(df_base, sw_base)

    base_cfg_2020 = BacktestConfig(
        buy_threshold=75.0, sell_threshold=55.0,
        settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
        use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset',
        start_date='2020-01-01', end_date='2020-12-31'
    )
    df_base_2020, _, _ = run_simulation(data, base_cfg_2020)
    k_base_2020 = compute_kpis(df_base_2020, 0)

    base_cfg_2022 = BacktestConfig(
        buy_threshold=75.0, sell_threshold=55.0,
        settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
        use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset',
        start_date='2022-01-01', end_date='2022-12-31'
    )
    df_base_2022, _, _ = run_simulation(data, base_cfg_2022)
    k_base_2022 = compute_kpis(df_base_2022, 0)

    exp1_row = {
        'name': 'Exp 1: Baseline All-in (Quant Score 75/55)',
        'cagr': k_base['cagr'],
        'cum_ret': k_base['cum_ret'],
        'mdd': k_base['mdd'],
        'sharpe': k_base['sharpe'],
        'sortino': k_base['sortino'],
        'calmar': k_base['calmar'],
        'ret_2020': k_base_2020['cum_ret'],
        'ret_2022': k_base_2022['cum_ret'],
        'annual_switches': k_base['annual_switches']
    }

    # Exp 2-6
    exp2_row = run_experiment_suite(data, 'exp2_tranche', "Exp 2: Mother's 3-Tranche Original (EMA 50/100/200)")
    exp3_row = run_experiment_suite(data, 'exp3_faber', "Exp 3: Faber GTAA Overlay (No Tranche 1 if < EMA200)")
    exp4_row = run_experiment_suite(data, 'exp4_keller', "Exp 4: Keller DAA Canary (VWO + BND 13612W)")
    exp5_row = run_experiment_suite(data, 'exp5_richman', "Exp 5: Richman HAA Canary (TIP 13612 Mom)")
    exp6_row = run_experiment_suite(data, 'exp6_ultimate', "Exp 6: The Ultimate Hybrid (Opus + HAA + Tranche)")

    # SPY Benchmark
    spy_full = get_spy_kpis(data, '2015-01-01', '2026-02-28')
    spy_2020 = get_spy_kpis(data, '2020-01-01', '2020-12-31')
    spy_2022 = get_spy_kpis(data, '2022-01-01', '2022-12-31')

    spy_row = {
        'name': '**Benchmark: SPY Buy & Hold**',
        'cagr': spy_full['cagr'],
        'cum_ret': spy_full['cum_ret'],
        'mdd': spy_full['mdd'],
        'sharpe': spy_full['sharpe'],
        'sortino': spy_full['sortino'],
        'calmar': spy_full['calmar'],
        'ret_2020': spy_2020['cum_ret'],
        'ret_2022': spy_2022['cum_ret'],
        'annual_switches': 0.0
    }

    matrix = [exp1_row, exp2_row, exp3_row, exp4_row, exp5_row, exp6_row, spy_row]

    print('\n# =========================================================================================================================')
    print('#                 QUANT ALLOCATION EXPERIMENTS MATRIX (6 PAPERS & ARCHITECTURES HEAD-TO-HEAD)')
    print('# =========================================================================================================================\n')
    header = '| Model / Experiment | CAGR (%) | Cum Ret (%) | Max DD (%) | Sharpe | Sortino | Calmar | 2020 COVID | 2022 Bear | Annual Switches |'
    sep = '| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |'
    print(header)
    print(sep)

    for r in matrix:
        print(f"| {r['name']} | {r['cagr']:+.2f}% | {r['cum_ret']:+.2f}% | {r['mdd']:.2f}% | {r['sharpe']:.2f} | {r['sortino']:.2f} | {r['calmar']:.2f} | {r['ret_2020']:+.2f}% | {r['ret_2022']:+.2f}% | {r['annual_switches']:.1f} |")
