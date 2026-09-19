import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from run_experiments import load_or_fetch_data, compute_kpis, get_spy_kpis, run_simulation, BacktestConfig

def get_tranche_target(row: pd.Series) -> float:
    """
    Computes target exposure (0.0, 0.33, 0.66, 1.0) based on staged EMA rules.
    - Hard Breakdown Guard: If Price < EMA200 by more than -2% -> 0.0 (Cut 100% to Cash Park)
    - ไม้ 3 (100%): Price > EMA200 and EMA50 > EMA100 > EMA200
    - ไม้ 2 (66%): Price > EMA100 and EMA50 > EMA100
    - ไม้ 1 (33%): Price > EMA50
    - Else: 0.0
    """
    p = row['close']
    e50, e100, e200 = row['ema50'], row['ema100'], row['ema200']

    if pd.isna(e50) or pd.isna(e100) or pd.isna(e200):
        return 0.0

    # Hard Breakdown Guard: หลุด EMA 200 เกิน -2%
    dist_ema200 = (p / e200 - 1.0) * 100.0 if e200 > 0 else 0.0
    if dist_ema200 < -2.0:
        return 0.0

    # Priority from highest stage to lowest
    # ไม้ 3 (100%): Price > EMA 200 และเรียงแถวสมบูรณ์ EMA 50 > 100 > 200
    if p > e200 and (e50 > e100 > e200):
        return 1.0

    # ไม้ 2 (66%): Price > EMA 100 และ EMA 50 > EMA 100
    if p > e100 and (e50 > e100):
        return 0.66

    # ไม้ 1 (33%): Price > EMA 50
    if p > e50:
        return 0.33

    return 0.0


def run_staged_simulation(
    data: Dict[str, pd.DataFrame],
    start_date: str = '2015-01-01',
    end_date: str = '2026-02-28',
    settlement_lag: int = 1,
    cash_yield_annual: float = 0.0175,
    switching_friction: float = 0.0005,
    equal_weight_across_universe: bool = True
):
    universe_tickers = ['URTH', 'CSPX.L', 'QQQ', 'GLD', 'SMH']
    all_dates = sorted(list(set.intersection(*[set(data[t].index) for t in universe_tickers])))
    all_dates = [d for d in all_dates if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)]

    if len(all_dates) < 2:
        return None, 0, {}

    # Precalculate targets for each day
    daily_targets = {t: {} for t in universe_tickers}
    for t in universe_tickers:
        df = data[t]
        for d in all_dates:
            row = df.loc[d]
            daily_targets[t][d] = get_tranche_target(row)

    current_weights = {t: 0.0 for t in universe_tickers}
    # Pending increases due to T+1 settlement lag: list of (remaining_days, target_w)
    pending_increases = {t: [] for t in universe_tickers}

    portfolio_value = 100000.0
    daily_records = []
    total_switches = 0
    switches_per_asset = {t: 0 for t in universe_tickers}

    daily_cash_rate = (1.0 + cash_yield_annual) ** (1.0 / 252.0) - 1.0

    for i in range(1, len(all_dates)):
        prev_date = all_dates[i - 1]
        curr_date = all_dates[i]

        # 1. Update positions based on t-1 signals
        for t in universe_tickers:
            tgt_raw = daily_targets[t][prev_date]
            tgt_w = (tgt_raw / len(universe_tickers)) if equal_weight_across_universe else (tgt_raw / len(universe_tickers))
            cur_w = current_weights[t]

            if tgt_w < cur_w:
                # Immediate cut / reduction
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

            # Process pending lag countdown
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

        # 2. Portfolio return on day t
        total_invested_weight = sum(current_weights.values())
        cash_weight = max(0.0, 1.0 - total_invested_weight)

        day_return = cash_weight * daily_cash_rate
        for t in universe_tickers:
            w = current_weights[t]
            if w > 0:
                p_prev = data[t].loc[prev_date, 'close']
                p_curr = data[t].loc[curr_date, 'close']
                ret = (p_curr / p_prev) - 1.0
                day_return += w * ret

        portfolio_value *= (1.0 + day_return)

        daily_records.append({
            'date': curr_date,
            'portfolio_value': portfolio_value,
            'day_return': day_return,
            'invested_weight': total_invested_weight,
            'cash_weight': cash_weight
        })

    res_df = pd.DataFrame(daily_records).set_index('date')
    return res_df, total_switches, switches_per_asset


if __name__ == '__main__':
    data = load_or_fetch_data()

    # 1. Baseline Model (Run 2: Realistic MF Friction T+1, 1.75% Cash Yield, 0.05% Friction)
    baseline_cfg = BacktestConfig(
        name='Baseline Run 2 (All-in 100% per asset, T+1, Yield=1.75%, Frict=0.05%)',
        buy_threshold=75.0, sell_threshold=55.0,
        settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
        use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset',
        start_date='2015-01-01', end_date='2026-02-28'
    )
    baseline_df, base_switches, _ = run_simulation(data, baseline_cfg)
    base_kpi = compute_kpis(baseline_df, base_switches)
    base_kpi['name'] = baseline_cfg.name

    # 2. Staged Entry Model (Fixed 5-Bucket Universe: Max 20% per asset, Tranches 0%, 33%, 66%, 100%)
    staged_df, staged_switches, staged_sw_asset = run_staged_simulation(
        data,
        start_date='2015-01-01',
        end_date='2026-02-28',
        settlement_lag=1,
        cash_yield_annual=0.0175,
        switching_friction=0.0005,
        equal_weight_across_universe=True
    )
    staged_kpi = compute_kpis(staged_df, staged_switches)
    staged_kpi['name'] = 'Staged 3-Tranche Entry (EMA 50/100/200, T+1, Yield=1.75%, Frict=0.05%)'

    # 3. Crisis Window: 2020 COVID Crash
    base_2020_cfg = BacktestConfig(
        name='Baseline Run 2 (2020 COVID)',
        buy_threshold=75.0, sell_threshold=55.0,
        settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
        use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset',
        start_date='2020-01-01', end_date='2020-12-31'
    )
    base_2020_df, base_2020_sw, _ = run_simulation(data, base_2020_cfg)
    base_2020_kpi = compute_kpis(base_2020_df, base_2020_sw)
    base_2020_kpi['name'] = 'Baseline Run 2 (2020 COVID)'

    staged_2020_df, staged_2020_sw, _ = run_staged_simulation(
        data, start_date='2020-01-01', end_date='2020-12-31',
        settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005
    )
    staged_2020_kpi = compute_kpis(staged_2020_df, staged_2020_sw)
    staged_2020_kpi['name'] = 'Staged Entry (2020 COVID)'

    # 4. Crisis Window: 2022 Inflation Bear
    base_2022_cfg = BacktestConfig(
        name='Baseline Run 2 (2022 Bear)',
        buy_threshold=75.0, sell_threshold=55.0,
        settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
        use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset',
        start_date='2022-01-01', end_date='2022-12-31'
    )
    base_2022_df, base_2022_sw, _ = run_simulation(data, base_2022_cfg)
    base_2022_kpi = compute_kpis(base_2022_df, base_2022_sw)
    base_2022_kpi['name'] = 'Baseline Run 2 (2022 Bear)'

    staged_2022_df, staged_2022_sw, _ = run_staged_simulation(
        data, start_date='2022-01-01', end_date='2022-12-31',
        settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005
    )
    staged_2022_kpi = compute_kpis(staged_2022_df, staged_2022_sw)
    staged_2022_kpi['name'] = 'Staged Entry (2022 Bear)'

    # Benchmarks
    spy_full = get_spy_kpis(data, '2015-01-01', '2026-02-28')

    print('\n# =========================================================================================')
    print('#       HEAD-TO-HEAD: STAGED 3-TRANCHE ALLOCATION vs BASELINE ALL-IN (2015-2026)')
    print('# =========================================================================================\n')
    header = '| Strategy / Model | CAGR (%) | Cum. Return (%) | Max DD (%) | Sharpe | Sortino | Calmar | Total Switches | Annual Switches |'
    sep = '| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |'
    print(header)
    print(sep)

    for k in [base_kpi, staged_kpi, base_2020_kpi, staged_2020_kpi, base_2022_kpi, staged_2022_kpi]:
        print(f"| {k['name']} | {k['cagr']:+.2f}% | {k['cum_ret']:+.2f}% | {k['mdd']:.2f}% | {k['sharpe']:.2f} | {k['sortino']:.2f} | {k['calmar']:.2f} | {k['total_switches']} | {k['annual_switches']:.1f} |")

    print(f"| **Benchmark: SPY Buy & Hold (2015-2026)** | {spy_full['cagr']:+.2f}% | {spy_full['cum_ret']:+.2f}% | {spy_full['mdd']:.2f}% | {spy_full['sharpe']:.2f} | {spy_full['sortino']:.2f} | {spy_full['calmar']:.2f} | 0 | 0.0 |")
    print('\nBreakdown switches per asset (Staged Entry 2015-2026):', staged_sw_asset)
