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
        print(f'Loading data from cache: {CACHE_FILE}')
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
    print('Data cached successfully.')
    return cleaned

@dataclass
class BacktestConfig:
    buy_threshold: float = 75.0
    sell_threshold: float = 55.0
    settlement_lag: int = 0          # Phase 1 = 0, Phase 2 = 1 or 2
    cash_yield_annual: float = 0.0   # Phase 1 = 0.0, Phase 2 = 1.75%
    switching_friction: float = 0.0  # Slippage / spread per switch
    use_anti_chop: bool = True
    use_hard_breakdown: bool = True
    mode: str = 'multi_asset'        # 'multi_asset' (Equal Weight of all IN) or 'best_momentum' (Top 1 score)
    start_date: str = '2015-01-01'
    end_date: str = '2026-02-28'

def run_simulation(data: Dict[str, pd.DataFrame], cfg: BacktestConfig):
    # Align dates across all 5 universe assets
    universe_tickers = ['URTH', 'CSPX.L', 'QQQ', 'GLD', 'SMH']
    all_dates = sorted(list(set.intersection(*[set(data[t].index) for t in universe_tickers])))
    all_dates = [d for d in all_dates if pd.to_datetime(cfg.start_date) <= d <= pd.to_datetime(cfg.end_date)]

    # Compute scores for all assets on each date (at close of t-1)
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

    # Simulation state
    positions = {t: 'OUT' for t in universe_tickers}
    pending_in = {t: 0 for t in universe_tickers}   # Settlement lag countdown
    
    portfolio_value = 100000.0
    daily_records = []
    total_switches = 0
    switches_per_asset = {t: 0 for t in universe_tickers}

    daily_cash_rate = (1.0 + cfg.cash_yield_annual) ** (1.0 / 252.0) - 1.0

    # Main loop over trading days t
    for i in range(1, len(all_dates)):
        prev_date = all_dates[i - 1]   # Day t - 1 (bar close used for decision)
        curr_date = all_dates[i]       # Day t (market open / day return)

        # 1. Update decisions based strictly on t-1 scores (NO LOOKAHEAD)
        target_positions = {}
        for t in universe_tickers:
            score = daily_scores[t][prev_date]
            b_down = daily_breakdown[t][prev_date]
            prev_pos = positions[t]

            if b_down or score < cfg.sell_threshold:
                target_positions[t] = 'OUT'
            elif score >= cfg.buy_threshold:
                target_positions[t] = 'IN'
            else:
                target_positions[t] = prev_pos  # Stay in hysteresis dead-zone

        # Handle 'best_momentum' mode: only the single highest score > buy_threshold can be IN
        if cfg.mode == 'best_momentum':
            eligible = [t for t in universe_tickers if target_positions[t] == 'IN']
            if eligible:
                best_t = max(eligible, key=lambda x: daily_scores[x][prev_date])
                for t in universe_tickers:
                    target_positions[t] = 'IN' if t == best_t else 'OUT'
            else:
                for t in universe_tickers:
                    target_positions[t] = 'OUT'

        # Check for switches and handle settlement lag
        for t in universe_tickers:
            if positions[t] == 'OUT' and target_positions[t] == 'IN':
                total_switches += 1
                switches_per_asset[t] += 1
                if cfg.settlement_lag > 0:
                    pending_in[t] = cfg.settlement_lag
                    positions[t] = 'PENDING'
                else:
                    positions[t] = 'IN'
                    portfolio_value *= (1.0 - cfg.switching_friction)
            elif positions[t] in ('IN', 'PENDING') and target_positions[t] == 'OUT':
                total_switches += 1
                switches_per_asset[t] += 1
                positions[t] = 'OUT'
                pending_in[t] = 0
                portfolio_value *= (1.0 - cfg.switching_friction)
            elif positions[t] == 'PENDING':
                pending_in[t] -= 1
                if pending_in[t] <= 0:
                    positions[t] = 'IN'
                    portfolio_value *= (1.0 - cfg.switching_friction)

        # 2. Compute portfolio return on day t
        active_assets = [t for t in universe_tickers if positions[t] == 'IN']
        n_active = len(active_assets)

        if n_active > 0:
            weight_per_asset = 1.0 / n_active
            day_return = 0.0
            for t in active_assets:
                p_prev = data[t].loc[prev_date, 'close']
                p_curr = data[t].loc[curr_date, 'close']
                ret = (p_curr / p_prev) - 1.0
                day_return += weight_per_asset * ret
        else:
            # 100% Cash park in SCBTMFPLUS-E
            day_return = daily_cash_rate

        portfolio_value *= (1.0 + day_return)

        daily_records.append({
            'date': curr_date,
            'portfolio_value': portfolio_value,
            'day_return': day_return,
            'active_count': n_active,
            'active_assets': ','.join(active_assets) if active_assets else 'CASH',
        })

    res_df = pd.DataFrame(daily_records).set_index('date')
    return res_df, total_switches, switches_per_asset

def compute_kpis(res_df: pd.DataFrame, total_switches: int, years: float):
    pv = res_df['portfolio_value']
    cum_ret = (pv.iloc[-1] / pv.iloc[0]) - 1.0
    cagr = (pv.iloc[-1] / pv.iloc[0]) ** (1.0 / years) - 1.0
    
    # Drawdown
    peak = pv.cummax()
    drawdown = (pv - peak) / peak
    mdd = drawdown.min()
    
    # Sharpe & Sortino (assumes 0% risk free for simple comparison)
    daily_rets = res_df['day_return']
    vol = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / vol if vol > 0 else 0.0
    
    neg_rets = daily_rets[daily_rets < 0]
    downside_dev = neg_rets.std() * np.sqrt(252)
    sortino = (daily_rets.mean() * 252) / downside_dev if downside_dev > 0 else 0.0
    
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    annual_switches = total_switches / years

    return {
        'CAGR': cagr * 100.0,
        'Cumulative Return': cum_ret * 100.0,
        'Max Drawdown': mdd * 100.0,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Total Switches': total_switches,
        'Annual Switches': annual_switches,
    }

if __name__ == '__main__':
    data = load_or_fetch_data()
    cfg_base = BacktestConfig(
        buy_threshold=75.0,
        sell_threshold=55.0,
        settlement_lag=0,
        cash_yield_annual=0.0,
        switching_friction=0.0,
        use_anti_chop=True,
        use_hard_breakdown=True,
        start_date='2015-01-01',
        end_date='2026-02-28'
    )
    
    res_df, total_sw, sw_per_asset = run_simulation(data, cfg_base)
    years = (res_df.index[-1] - res_df.index[0]).days / 365.25
    kpis = compute_kpis(res_df, total_sw, years)
    
    # Compare with SPY Buy & Hold
    spy = data['SPY'].loc[res_df.index[0]:res_df.index[-1]]
    spy_cum = (spy['close'] / spy['close'].iloc[0]) * 100000.0
    spy_cagr = (spy_cum.iloc[-1] / spy_cum.iloc[0]) ** (1.0 / years) - 1.0
    spy_peak = spy_cum.cummax()
    spy_mdd = ((spy_cum - spy_peak) / spy_peak).min()
    spy_rets = spy['close'].pct_change().dropna()
    spy_sharpe = (spy_rets.mean() * 252) / (spy_rets.std() * np.sqrt(252))
    
    print('=====================================================')
    print('   PHASE 1: BASELINE BACKTEST RESULTS (2015 - 2026)   ')
    print('=====================================================')
    print(f'Time Span: {res_df.index[0].date()} to {res_df.index[-1].date()} ({years:.1f} years)')
    print('-----------------------------------------------------')
    print(f'Metric                  Strategy (Baseline)     SPY (Buy&Hold)')
    print(f'CAGR:                   {kpis['CAGR']:>10.2f}%         {spy_cagr*100:>10.2f}%')
    print(f'Cumulative Return:      {kpis['Cumulative Return']:>10.2f}%         {((spy_cum.iloc[-1]/spy_cum.iloc[0])-1)*100:>10.2f}%')
    print(f'Max Drawdown (MDD):     {kpis['Max Drawdown']:>10.2f}%         {spy_mdd*100:>10.2f}%')
    print(f'Sharpe Ratio:           {kpis['Sharpe']:>10.2f}          {spy_sharpe:>10.2f}')
    print(f'Sortino Ratio:          {kpis['Sortino']:>10.2f}')
    print(f'Calmar Ratio:           {kpis['Calmar']:>10.2f}          {abs(spy_cagr/spy_mdd):>10.2f}')
    print(f'Total Switches:         {kpis['Total Switches']:>10d}')
    print(f'Annual Switches:        {kpis['Annual Switches']:>10.1f} switches/yr')
    print('-----------------------------------------------------')
    print('Switches per Asset:')
    for a, c in sw_per_asset.items():
        print(f'  - {a:8s}: {c:3d} switches ({c/years:.1f}/yr)')
    print('=====================================================')
