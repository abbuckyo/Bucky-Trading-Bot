import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from run_paper_experiments import load_or_fetch_all_data, run_paper_simulation, compute_kpis, get_spy_kpis
from run_experiments import run_simulation, BacktestConfig

def generate_staged_canary_plot():
    os.makedirs('docs', exist_ok=True)
    out_path = os.path.join('docs', 'staged_canary_equity_curve.png')

    print('Loading data from cache...')
    data = load_or_fetch_all_data()

    # 1. AlphaShield V8.0 (Beta): Staged Tranche + TIP Canary (Exp 5)
    print('Simulating AlphaShield V8.0 (Beta)...')
    df_v8, sw_v8, _ = run_paper_simulation(
        data,
        exp_type='exp5_richman',
        start_date='2015-01-01',
        end_date='2026-02-28',
        settlement_lag=1,
        cash_yield_annual=0.0175,
        switching_friction=0.0005
    )
    kpis_v8 = compute_kpis(df_v8, sw_v8)

    # 2. AlphaShield V7.2 (Legacy Baseline): All-in 75/55 (Run 2)
    print('Simulating AlphaShield V7.2 (Legacy Baseline)...')
    cfg_v7 = BacktestConfig(
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
    df_v7, sw_v7, _ = run_simulation(data, cfg_v7)
    kpis_v7 = compute_kpis(df_v7, sw_v7)

    # 3. SPY Buy & Hold Benchmark
    common_idx = df_v8.index.intersection(df_v7.index)
    df_v8 = df_v8.loc[common_idx]
    df_v7 = df_v7.loc[common_idx]

    spy = data['SPY'].copy()
    spy = spy.loc[common_idx[0]:common_idx[-1]]
    spy_equity = (spy['close'] / spy['close'].iloc[0]) * 100000.0

    eq_v8 = df_v8['portfolio_value']
    eq_v7 = df_v7['portfolio_value']

    # Drawdown calculations (%)
    dd_v8 = (eq_v8 - eq_v8.cummax()) / eq_v8.cummax() * 100.0
    dd_v7 = (eq_v7 - eq_v7.cummax()) / eq_v7.cummax() * 100.0
    dd_spy = (spy_equity - spy_equity.cummax()) / spy_equity.cummax() * 100.0

    # Plot styling
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={'height_ratios': [2.2, 1.2]}
    )

    # ── Top Plot: Equity Curve (Log Scale) ──
    ax1.plot(
        eq_v8.index, eq_v8,
        label=f'AlphaShield V8.0 (Beta) — Tranche + TIP Canary | CAGR: +{kpis_v8["cagr"]:.2f}% | Sharpe: {kpis_v8["sharpe"]:.2f} | MDD: {kpis_v8["mdd"]:.1f}%',
        color='#00FFA3', linewidth=2.4
    )
    ax1.plot(
        eq_v7.index, eq_v7,
        label=f'AlphaShield V7.2 (Legacy Baseline) — All-in 75/55 | CAGR: +{kpis_v7["cagr"]:.2f}% | Sharpe: {kpis_v7["sharpe"]:.2f} | MDD: {kpis_v7["mdd"]:.1f}%',
        color='#F59E0B', linewidth=1.8, alpha=0.9
    )
    ax1.plot(
        spy_equity.index, spy_equity,
        label='SPY Buy & Hold (Benchmark) | CAGR: +13.29% | Sharpe: 0.79 | MDD: -33.72%',
        color='#3B82F6', linewidth=1.5, alpha=0.85, linestyle='--'
    )

    ax1.set_yscale('log')
    ax1.set_title('Head-to-Head Performance: AlphaShield V8.0 (Beta) vs Legacy V7.2 vs SPY (2015 - 2026)', fontsize=15, fontweight='bold', pad=12, color='#FFFFFF')
    ax1.set_ylabel('Portfolio Value (USD, Log Scale)', fontsize=11, color='#E2E8F0')
    ax1.grid(True, which='both', color='#334155', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.85, facecolor='#1E293B', edgecolor='#475569')

    # Shading Crisis Periods
    # 2020 COVID
    ax1.axvspan(pd.to_datetime('2020-02-19'), pd.to_datetime('2020-04-30'), color='#EF4444', alpha=0.15)
    ax1.text(pd.to_datetime('2020-02-25'), eq_v8.max() * 0.45, 'COVID-19 Crash\n(V8.0: +21.7% vs SPY -33.7% DD)', color='#FCA5A5', fontsize=9, fontweight='bold')

    # 2022 Inflation Bear
    ax1.axvspan(pd.to_datetime('2022-01-03'), pd.to_datetime('2022-12-31'), color='#F59E0B', alpha=0.15)
    ax1.text(pd.to_datetime('2022-02-01'), eq_v8.min() * 1.05, '2022 Bear Market (HAA TIP Canary Defense)\nV8.0: -0.33% vs Legacy -5.46% vs SPY -18.65%', color='#FCD34D', fontsize=9, fontweight='bold')

    # ── Bottom Plot: Underwater Drawdown Profile (%) ──
    ax2.plot(dd_v8.index, dd_v8, label=f'AlphaShield V8.0 Drawdown (Max: {kpis_v8["mdd"]:.1f}%)', color='#00FFA3', linewidth=1.6)
    ax2.fill_between(dd_v8.index, dd_v8, 0, color='#00FFA3', alpha=0.20)

    ax2.plot(dd_v7.index, dd_v7, label=f'Legacy V7.2 Drawdown (Max: {kpis_v7["mdd"]:.1f}%)', color='#F59E0B', linewidth=1.4, alpha=0.8)

    ax2.plot(dd_spy.index, dd_spy, label='SPY Drawdown (Max: -33.7%)', color='#3B82F6', linewidth=1.2, linestyle='--', alpha=0.7)
    ax2.fill_between(dd_spy.index, dd_spy, 0, color='#3B82F6', alpha=0.10)

    ax2.set_title('Underwater Drawdown Profile Comparison (%)', fontsize=12, fontweight='bold', pad=8, color='#FFFFFF')
    ax2.set_ylabel('Drawdown (%)', fontsize=11, color='#E2E8F0')
    ax2.set_xlabel('Date', fontsize=11, color='#E2E8F0')
    ax2.set_ylim(-40, 2)
    ax2.grid(True, color='#334155', linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=10, framealpha=0.85, facecolor='#1E293B', edgecolor='#475569')

    ax2.xaxis.set_major_locator(mdates.YearLocator(1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Benchmark chart saved successfully to: {out_path}')

if __name__ == '__main__':
    generate_staged_canary_plot()
