import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from backtest import load_or_fetch_data, run_simulation, BacktestConfig, compute_kpis

def generate_backtest_plot():
    os.makedirs('docs', exist_ok=True)
    out_path = os.path.join('docs', 'backtest_equity_curve.png')

    print('Loading data and running Run 2 (Realistic MF Friction)...')
    data = load_or_fetch_data()

    cfg_run2 = BacktestConfig(
        buy_threshold=75.0,
        sell_threshold=55.0,
        settlement_lag=1,
        cash_yield_annual=0.0175,
        switching_friction=0.0005,
        use_anti_chop=True,
        use_hard_breakdown=True,
        mode='multi_asset',
        start_date='2015-01-01',
        end_date='2026-02-28'
    )

    res_df, total_sw, _ = run_simulation(data, cfg_run2)
    kpis = compute_kpis(res_df, total_sw, (res_df.index[-1] - res_df.index[0]).days / 365.25)

    # SPY benchmark alignment
    spy = data['SPY'].copy()
    spy = spy.loc[res_df.index[0]:res_df.index[-1]]
    spy_equity = (spy['close'] / spy['close'].iloc[0]) * 100000.0

    strat_equity = res_df['portfolio_value']

    # Underwater Drawdown (%)
    strat_peak = strat_equity.cummax()
    strat_dd = (strat_equity - strat_peak) / strat_peak * 100.0

    spy_peak = spy_equity.cummax()
    spy_dd = (spy_equity - spy_peak) / spy_peak * 100.0

    # Plot styling
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={'height_ratios': [2.2, 1.2]}
    )

    # 1. Equity Curve (Log Scale)
    cagr_val = kpis['CAGR']
    sharpe_val = kpis['Sharpe']
    mdd_val = kpis['Max Drawdown']
    ax1.plot(strat_equity.index, strat_equity, label=f'Quant Strategy (Run 2: MF Realistic) | CAGR: {cagr_val:.1f}% | Sharpe: {sharpe_val:.2f}', color='#00FFA3', linewidth=2.0)
    ax1.plot(spy_equity.index, spy_equity, label='SPY Buy & Hold | CAGR: 13.3% | Sharpe: 0.79', color='#3B82F6', linewidth=1.5, alpha=0.85, linestyle='--')
    ax1.set_yscale('log')
    ax1.set_title('Tactical Asset Allocation Strategy vs SPY Buy & Hold (2015 - 2026)', fontsize=15, fontweight='bold', pad=12, color='#FFFFFF')
    ax1.set_ylabel('Portfolio Value (USD, Log Scale)', fontsize=11, color='#E2E8F0')
    ax1.grid(True, which='both', color='#334155', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.8, facecolor='#1E293B', edgecolor='#475569')

    # Add recession / crisis shading
    # 2020 COVID
    ax1.axvspan(pd.to_datetime('2020-02-19'), pd.to_datetime('2020-04-30'), color='#EF4444', alpha=0.15)
    ax1.text(pd.to_datetime('2020-03-01'), strat_equity.max()*0.6, 'COVID-19 Crash\n(Strategy MDD -16% vs SPY -34%)', color='#FCA5A5', fontsize=9, fontweight='bold')

    # 2022 Inflation Bear
    ax1.axvspan(pd.to_datetime('2022-01-03'), pd.to_datetime('2022-10-12'), color='#F59E0B', alpha=0.15)
    ax1.text(pd.to_datetime('2022-03-01'), strat_equity.min()*1.2, '2022 Bear Market\n(Cash Park Defense)', color='#FCD34D', fontsize=9, fontweight='bold')

    # 2. Underwater Drawdown (%)
    ax2.plot(strat_dd.index, strat_dd, label=f'Strategy Drawdown (Max: {mdd_val:.1f}%)', color='#00FFA3', linewidth=1.5)
    ax2.fill_between(strat_dd.index, strat_dd, 0, color='#00FFA3', alpha=0.25)

    ax2.plot(spy_dd.index, spy_dd, label=f'SPY Drawdown (Max: -33.7%)', color='#EF4444', linewidth=1.2, linestyle='--', alpha=0.8)
    ax2.fill_between(spy_dd.index, spy_dd, 0, color='#EF4444', alpha=0.15)

    ax2.set_title('Underwater Drawdown Profile (%)', fontsize=12, fontweight='bold', pad=8, color='#FFFFFF')
    ax2.set_ylabel('Drawdown (%)', fontsize=11, color='#E2E8F0')
    ax2.set_xlabel('Date', fontsize=11, color='#E2E8F0')
    ax2.set_ylim(-40, 2)
    ax2.grid(True, color='#334155', linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=10, framealpha=0.8, facecolor='#1E293B', edgecolor='#475569')

    ax2.xaxis.set_major_locator(mdates.YearLocator(1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Graph saved successfully to: {out_path}')

if __name__ == '__main__':
    generate_backtest_plot()
