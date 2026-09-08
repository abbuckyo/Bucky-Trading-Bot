import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from backtest_core import precompute_asset_indicators, score_bar

CACHE_FILE = 'historical_data_10y.pkl'
TICKERS = {
    'URTH': 'SCBWORLDE',
    'CSPX.L': 'SCBS&P500E',
    'QQQ': 'SCBNDQ(E)',
    'GLD': 'SCBGOLDE',
    'SMH': 'SCBSEMI(E)',
    'SPY': 'BENCHMARK_SPY'
}

def load_or_fetch_data():
    if os.path.exists(CACHE_FILE):
        return pd.read_pickle(CACHE_FILE)
    print('Downloading 10-year historical data from Yahoo Finance...')
    raw = yf.download(list(TICKERS.keys()), start='2014-01-01', end='2026-03-01', auto_adjust=True, group_by='ticker')
    cleaned = {}
    for t in TICKERS:
        df = raw[t].copy().dropna(subset=['Close'])
        df.columns = [c.lower() for c in df.columns]
        df = precompute_asset_indicators(df)
        cleaned[t] = df
    pd.to_pickle(cleaned, CACHE_FILE)
    return cleaned

@dataclass
class BacktestConfig:
    name: str = 'Config'
    buy_threshold: float = 75.0
    sell_threshold: float = 55.0
    settlement_lag: int = 0
    cash_yield_annual: float = 0.0
    switching_friction: float = 0.0
    use_anti_chop: bool = True
    use_hard_breakdown: bool = True
    mode: str = 'multi_asset'        # 'multi_asset' or 'best_momentum'
    start_date: str = '2015-01-01'
    end_date: str = '2026-02-28'

def run_simulation(data: Dict[str, pd.DataFrame], cfg: BacktestConfig):
    universe_tickers = ['URTH', 'CSPX.L', 'QQQ', 'GLD', 'SMH']
    all_dates = sorted(list(set.intersection(*[set(data[t].index) for t in universe_tickers])))
    all_dates = [d for d in all_dates if pd.to_datetime(cfg.start_date) <= d <= pd.to_datetime(cfg.end_date)]

    if len(all_dates) < 2:
        return None, 0, {}

    daily_scores = {t: {} for t in universe_tickers}
    daily_breakdown = {t: {} for t in universe_tickers}

    for t in universe_tickers:
        df = data[t]
        for d in all_dates:
            row = df.loc[d]
            s, b = score_bar(
                row,
                buy_th=cfg.buy_threshold,
                sell_th=cfg.sell_threshold,
                use_anti_chop=cfg.use_anti_chop,
                use_hard_breakdown=cfg.use_hard_breakdown
            )
            daily_scores[t][d] = s
            daily_breakdown[t][d] = b

    positions = {t: 'OUT' for t in universe_tickers}
    pending_in = {t: 0 for t in universe_tickers}
    portfolio_value = 100000.0
    daily_records = []
    total_switches = 0
    switches_per_asset = {t: 0 for t in universe_tickers}

    daily_cash_rate = (1.0 + cfg.cash_yield_annual) ** (1.0 / 252.0) - 1.0

    for i in range(1, len(all_dates)):
        prev_date = all_dates[i - 1]
        curr_date = all_dates[i]

        # 1. Decision at t-1 close
        target_positions = {}
        for t in universe_tickers:
            score = daily_scores[t][prev_date]
            b_down = daily_breakdown[t][prev_date]
            prev_pos = positions[t]

            # In hysteresis logic
            effective_prev = 'IN' if prev_pos in ('IN', 'PENDING') else 'OUT'
            if b_down or score < cfg.sell_threshold:
                target_positions[t] = 'OUT'
            elif score >= cfg.buy_threshold:
                target_positions[t] = 'IN'
            else:
                target_positions[t] = effective_prev

        if cfg.mode == 'best_momentum':
            eligible = [t for t in universe_tickers if target_positions[t] == 'IN']
            if eligible:
                best_t = max(eligible, key=lambda x: daily_scores[x][prev_date])
                for t in universe_tickers:
                    target_positions[t] = 'IN' if t == best_t else 'OUT'
            else:
                for t in universe_tickers:
                    target_positions[t] = 'OUT'

        # Process position transitions
        for t in universe_tickers:
            tgt = target_positions[t]
            cur = positions[t]

            if cur == 'OUT' and tgt == 'IN':
                total_switches += 1
                switches_per_asset[t] += 1
                if cfg.settlement_lag > 0:
                    pending_in[t] = cfg.settlement_lag
                    positions[t] = 'PENDING'
                else:
                    positions[t] = 'IN'
                    portfolio_value *= (1.0 - cfg.switching_friction)
            elif cur in ('IN', 'PENDING') and tgt == 'OUT':
                total_switches += 1
                switches_per_asset[t] += 1
                positions[t] = 'OUT'
                pending_in[t] = 0
                portfolio_value *= (1.0 - cfg.switching_friction)
            elif cur == 'PENDING':
                pending_in[t] -= 1
                if pending_in[t] <= 0:
                    positions[t] = 'IN'
                    portfolio_value *= (1.0 - cfg.switching_friction)

        # 2. Portfolio return on day t
        active_assets = [t for t in universe_tickers if positions[t] == 'IN']
        n_active = len(active_assets)

        if n_active > 0:
            weight = 1.0 / n_active
            day_return = 0.0
            for t in active_assets:
                p_prev = data[t].loc[prev_date, 'close']
                p_curr = data[t].loc[curr_date, 'close']
                day_return += weight * ((p_curr / p_prev) - 1.0)
        else:
            day_return = daily_cash_rate

        portfolio_value *= (1.0 + day_return)

        daily_records.append({
            'date': curr_date,
            'portfolio_value': portfolio_value,
            'day_return': day_return,
            'active_count': n_active,
        })

    res_df = pd.DataFrame(daily_records).set_index('date')
    return res_df, total_switches, switches_per_asset

def compute_kpis(res_df: pd.DataFrame, total_switches: int):
    years = (res_df.index[-1] - res_df.index[0]).days / 365.25
    if years <= 0:
        years = 1.0 / 252.0
    pv = res_df['portfolio_value']
    cum_ret = (pv.iloc[-1] / pv.iloc[0]) - 1.0
    
    # CAGR (for < 1 yr period, annualized can be volatile, but keep standard formula)
    if years >= 1.0:
        cagr = (pv.iloc[-1] / pv.iloc[0]) ** (1.0 / years) - 1.0
    else:
        cagr = (1.0 + cum_ret) ** (1.0 / years) - 1.0
        
    peak = pv.cummax()
    drawdown = (pv - peak) / peak
    mdd = drawdown.min()
    
    daily_rets = res_df['day_return']
    vol = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / vol if vol > 0 else 0.0
    
    neg_rets = daily_rets[daily_rets < 0]
    downside_dev = neg_rets.std() * np.sqrt(252)
    sortino = (daily_rets.mean() * 252) / downside_dev if downside_dev > 0 else 0.0
    
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    annual_switches = total_switches / years

    return {
        'years': years,
        'cagr': cagr * 100.0,
        'cum_ret': cum_ret * 100.0,
        'mdd': mdd * 100.0,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'total_switches': total_switches,
        'annual_switches': annual_switches,
    }

def get_spy_kpis(data, start_date, end_date):
    spy = data['SPY'].copy()
    spy = spy[(spy.index >= pd.to_datetime(start_date)) & (spy.index <= pd.to_datetime(end_date))]
    if len(spy) < 2:
        return {}
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    if years <= 0:
        years = 1.0 / 252.0
    p = spy['close']
    cum_ret = (p.iloc[-1] / p.iloc[0]) - 1.0
    if years >= 1.0:
        cagr = (p.iloc[-1] / p.iloc[0]) ** (1.0 / years) - 1.0
    else:
        cagr = (1.0 + cum_ret) ** (1.0 / years) - 1.0
    peak = p.cummax()
    mdd = ((p - peak) / peak).min()
    rets = p.pct_change().dropna()
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / vol if vol > 0 else 0.0
    neg = rets[rets < 0]
    sortino = (rets.mean() * 252) / (neg.std() * np.sqrt(252)) if len(neg) > 0 and neg.std() > 0 else 0.0
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    return {
        'years': years,
        'cagr': cagr * 100.0,
        'cum_ret': cum_ret * 100.0,
        'mdd': mdd * 100.0,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
    }

# Main Batch Runner
if __name__ == '__main__':
    data = load_or_fetch_data()

    experiments = [
        BacktestConfig(
            name='Run 1: Pure Baseline (Lag=0, Yield=0%, Frict=0%)',
            buy_threshold=75.0, sell_threshold=55.0,
            settlement_lag=0, cash_yield_annual=0.0, switching_friction=0.0,
            use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset'
        ),
        BacktestConfig(
            name='Run 2: Realistic MF Friction (T+1, Yield=1.75%, Frict=0.05%)',
            buy_threshold=75.0, sell_threshold=55.0,
            settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
            use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset'
        ),
        BacktestConfig(
            name='Run 3: Heavy Friction Stress (T+2, Yield=1.75%, Frict=0.10%)',
            buy_threshold=75.0, sell_threshold=55.0,
            settlement_lag=2, cash_yield_annual=0.0175, switching_friction=0.0010,
            use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset'
        ),
        BacktestConfig(
            name='Run 4: Fast Threshold 70/50 (T+1, Yield=1.75%, Frict=0.05%)',
            buy_threshold=70.0, sell_threshold=50.0,
            settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
            use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset'
        ),
        BacktestConfig(
            name='Run 5: Strict Threshold 80/60 (T+1, Yield=1.75%, Frict=0.05%)',
            buy_threshold=80.0, sell_threshold=60.0,
            settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
            use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset'
        ),
        BacktestConfig(
            name='Run 6: Without Anti-Chop Guard (T+1, Yield=1.75%, Frict=0.05%)',
            buy_threshold=75.0, sell_threshold=55.0,
            settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
            use_anti_chop=False, use_hard_breakdown=True, mode='multi_asset'
        ),
        BacktestConfig(
            name='Run 7: Without Hard Breakdown (-2% EMA200 Off)',
            buy_threshold=75.0, sell_threshold=55.0,
            settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
            use_anti_chop=True, use_hard_breakdown=False, mode='multi_asset'
        ),
        BacktestConfig(
            name='Run 8: Crisis Window: COVID-19 (2020)',
            buy_threshold=75.0, sell_threshold=55.0,
            settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
            use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset',
            start_date='2020-01-01', end_date='2020-12-31'
        ),
        BacktestConfig(
            name='Run 9: Crisis Window: Inflation Bear (2022)',
            buy_threshold=75.0, sell_threshold=55.0,
            settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
            use_anti_chop=True, use_hard_breakdown=True, mode='multi_asset',
            start_date='2022-01-01', end_date='2022-12-31'
        ),
        BacktestConfig(
            name='Run 10: Best Momentum Picker (Top 1 Pick)',
            buy_threshold=75.0, sell_threshold=55.0,
            settlement_lag=1, cash_yield_annual=0.0175, switching_friction=0.0005,
            use_anti_chop=True, use_hard_breakdown=True, mode='best_momentum'
        ),
    ]

    results = []
    for exp in experiments:
        res_df, total_sw, sw_asset = run_simulation(data, exp)
        k = compute_kpis(res_df, total_sw)
        k['name'] = exp.name
        results.append(k)

    spy_full = get_spy_kpis(data, '2015-01-01', '2026-02-28')
    spy_2020 = get_spy_kpis(data, '2020-01-01', '2020-12-31')
    spy_2022 = get_spy_kpis(data, '2022-01-01', '2022-12-31')

    # Output formatted markdown table
    print('\n# =========================================================================================')
    print('#              10-ITERATION QUANT STRESS TEST & FRICTION SIMULATION RESULTS')
    print('# =========================================================================================\n')
    header = '| Iteration / Test Case | CAGR (%) | Cum. Return (%) | Max DD (%) | Sharpe | Sortino | Calmar | Annual Switches |'
    sep = '| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |'
    print(header)
    print(sep)
    for r in results:
        name = r["name"]
        cagr = r["cagr"]
        cum = r["cum_ret"]
        mdd = r["mdd"]
        sh = r["sharpe"]
        so = r["sortino"]
        cal = r["calmar"]
        sw = r["annual_switches"]
        print(f"| {name} | {cagr:+.2f}% | {cum:+.2f}% | {mdd:.2f}% | {sh:.2f} | {so:.2f} | {cal:.2f} | {sw:.1f} |")

    print(f"| **Benchmark: SPY Buy & Hold (2015-2026)** | {spy_full['cagr']:+.2f}% | {spy_full['cum_ret']:+.2f}% | {spy_full['mdd']:.2f}% | {spy_full['sharpe']:.2f} | {spy_full['sortino']:.2f} | {spy_full['calmar']:.2f} | 0.0 |")
    print(f"| **Benchmark: SPY (2020 COVID Crash)** | {spy_2020['cagr']:+.2f}% | {spy_2020['cum_ret']:+.2f}% | {spy_2020['mdd']:.2f}% | {spy_2020['sharpe']:.2f} | {spy_2020['sortino']:.2f} | {spy_2020['calmar']:.2f} | 0.0 |")
    print(f"| **Benchmark: SPY (2022 Inflation Bear)** | {spy_2022['cagr']:+.2f}% | {spy_2022['cum_ret']:+.2f}% | {spy_2022['mdd']:.2f}% | {spy_2022['sharpe']:.2f} | {spy_2022['sortino']:.2f} | {spy_2022['calmar']:.2f} | 0.0 |")
