# -*- coding: utf-8 -*-
"""Position Sizing, Signal Construction & Strict T+1 Execution Ledger.

1. Fixed Fractional Risk Parity Sizing:
   - Target Risk per Trade = 0.75% of current portfolio equity (NAV).
   - Stop Distance = 3.0 * ATR_20.
   - Target Notional Weight = min(0.15, 0.0075 / (Stop Distance / Close))
     (equivalently, Target Notional = min(0.15 * NAV, (0.0075 * NAV) / (Stop Distance / Close))).
2. Signal State Machine (Close(t)):
   - Weekly Rebalance (Mondays): rank universe by Momentum Score, select Top 20%
     (5 to 10 stocks max).
   - Macro Gate (SPY > EMA200): if False, freeze new entries (hold existing valid
     winners without increasing risk).
   - Earnings Blackout (within 5 trading days of earnings): block new entries;
     if already holding, tighten Chandelier stop to 1.5 * ATR_20.
   - Daily Chandelier Trailing Stop: if Close(t) < Stop Price(t), set target
     weight to 0.0 at Close(t) -> executes full exit at Open(t+1).
3. Execution Ledger:
   - Strict T+1 execution lag (Close(t) signals execute at Open(t+1)).
   - 5 bps transaction cost per side on all rebalances.
   - Unallocated cash parked at 2.0% annual yield.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_CASH_YIELD,
    DEFAULT_COST_BPS,
    EngineConfig,
    MAX_SINGLE_STOCK_WEIGHT,
    TARGET_RISK_PER_TRADE,
    TRADING_DAYS_PER_YEAR,
)
from .ranking_engine import RankingOutput
from .regime_guard import RegimeState
from .risk_guards import RiskGuardState

_MAX_COST_ITERATIONS = 60


@dataclass
class Ledger:
    """Complete accounting record for a backtest run."""

    returns: pd.Series
    nav: pd.Series
    weights: pd.DataFrame
    signal_weights: pd.DataFrame
    turnover: pd.Series
    friction: pd.Series
    components: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def risky_exposure(self) -> pd.Series:
        return self.weights.sum(axis=1).rename("exposure")

    @property
    def cash_weight(self) -> pd.Series:
        return (1.0 - self.risky_exposure).clip(lower=0.0).rename("cash_weight")

    @property
    def active_count(self) -> pd.Series:
        return (self.weights > 1e-9).sum(axis=1).rename("active_holdings")


def compute_position_weight(
    close: float | np.ndarray | pd.Series | pd.DataFrame,
    atr20: float | np.ndarray | pd.Series | pd.DataFrame,
    target_risk: float = TARGET_RISK_PER_TRADE,
    atr_mult: float = 3.0,
    max_weight: float = MAX_SINGLE_STOCK_WEIGHT,
) -> float | np.ndarray | pd.Series | pd.DataFrame:
    """Compute Fixed Fractional Risk Parity target weight capped at ``max_weight`` (15%).

    Formula:
        Stop Distance = atr_mult * ATR_20
        Weight = min(max_weight, (target_risk * NAV) / (Stop Distance / Close) / NAV)
               = min(max_weight, target_risk / (Stop Distance / Close))
    """
    if max_weight > MAX_SINGLE_STOCK_WEIGHT + 1e-12 or max_weight <= 0.0:
        raise ValueError(f"max_weight must be in (0, {MAX_SINGLE_STOCK_WEIGHT}]")

    c = np.asarray(close, dtype=float)
    a = np.asarray(atr20, dtype=float)
    stop_dist = atr_mult * a

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_stop = np.where((c > 0.0) & (stop_dist > 0.0), stop_dist / c, np.nan)
        raw_weight = np.where(
            np.isfinite(pct_stop) & (pct_stop > 0.0),
            target_risk / pct_stop,
            np.where((c > 0.0) & (stop_dist == 0.0), max_weight, 0.0),
        )
    clipped = np.clip(np.where(np.isfinite(raw_weight), raw_weight, 0.0), 0.0, max_weight)

    if isinstance(close, pd.DataFrame):
        return pd.DataFrame(clipped, index=close.index, columns=close.columns)
    if isinstance(close, pd.Series):
        return pd.Series(clipped, index=close.index, name=close.name)
    if np.ndim(close) == 0:
        return float(clipped.item())
    return clipped


def compute_target_notional(
    portfolio_nav: float,
    close: float,
    atr20: float,
    target_risk: float = TARGET_RISK_PER_TRADE,
    atr_mult: float = 3.0,
    max_weight: float = MAX_SINGLE_STOCK_WEIGHT,
) -> float:
    """Compute dollar notional allocation for a single stock given current ``portfolio_nav``."""
    if portfolio_nav <= 0.0:
        return 0.0
    w = float(
        compute_position_weight(
            close=close,
            atr20=atr20,
            target_risk=target_risk,
            atr_mult=atr_mult,
            max_weight=max_weight,
        )
    )
    return w * float(portfolio_nav)


def build_signal_weights(
    closes: pd.DataFrame,
    ranking: RankingOutput,
    regime: RegimeState,
    guards: RiskGuardState,
    config: EngineConfig | None = None,
) -> pd.DataFrame:
    """Generate end-of-day (Close(t)) target portfolio weights prior to T+1 execution.

    Rules enforced at each session ``t``:
      1. Daily Exit Check (all days):
         - Any currently held position with ``stop_breached[t, i] == True``
           (``Close(t) < High_22(t) - m * ATR_20(t)``, where ``m = 1.5`` during
           5-day pre-earnings blackout and ``3.0`` otherwise) or ``Close(t) <= EMA100(t)``
           is immediately exited (target weight = 0.0 at Close(t), executed at Open(t+1)).
      2. Weekly Rebalance (when ``ranking.rebalance_mask[t] == True``):
         - Candidate stocks must be in ``ranking.eligible_pool[t]``, pass
           ``guards.trend_eligible[t]`` (``Close > EMA100``), and have `` stop_breached == False``.
         - Ranked descending by ``ranking.ranks[t]`` up to ``config.max_active_holdings``.
         - For a NEW position (not held at ``t-1``):
             * Blocked if ``regime.macro_bullish[t] == False`` (SPY <= EMA 200).
             * Blocked if ``regime.earnings_blackout[t, i] == True`` (earnings within 5 days).
         - For an ALREADY HELD position:
             * If ``regime.macro_bullish[t] == False``: hold existing valid winner
               (provided it still passes ``Close > EMA100`` and ``~stop_breached``)
               without increasing its weight (`min(prev_weight, new_sized_weight)`).
             * If ``regime.earnings_blackout[t, i] == True``: do not add risk
               (`min(prev_weight, new_sized_weight)`), and its stop is tightened to 1.5x ATR.
      3. Portfolio Cap:
         - Every single stock weight is strictly ``<= 0.15`` (15.0%).
         - Total portfolio weight is capped at ``1.0`` (100% of NAV).
    """
    cfg = config or EngineConfig()
    dates = closes.index
    tickers = list(closes.columns)
    n_dates = len(dates)
    n_tickers = len(tickers)

    raw_sized = compute_position_weight(
        close=closes,
        atr20=guards.atr20,
        target_risk=cfg.target_risk_per_trade,
        atr_mult=cfg.chandelier_atr_mult,
        max_weight=cfg.max_weight_per_stock,
    ).to_numpy(dtype=float)

    eligible_arr = ranking.eligible_pool.reindex(index=dates, columns=tickers).fillna(False).to_numpy(dtype=bool)
    ranks_arr = ranking.ranks.reindex(index=dates, columns=tickers).fillna(np.inf).to_numpy(dtype=float)
    rebal_arr = ranking.rebalance_mask.reindex(dates).fillna(False).to_numpy(dtype=bool)

    macro_arr = regime.macro_bullish.reindex(dates).fillna(False).to_numpy(dtype=bool)
    blackout_arr = regime.earnings_blackout.reindex(index=dates, columns=tickers).fillna(False).to_numpy(dtype=bool)

    trend_arr = guards.trend_eligible.reindex(index=dates, columns=tickers).fillna(False).to_numpy(dtype=bool)
    breach_arr = guards.stop_breached.reindex(index=dates, columns=tickers).fillna(False).to_numpy(dtype=bool)

    out_weights = np.zeros((n_dates, n_tickers), dtype=float)
    current_w = np.zeros(n_tickers, dtype=float)

    for t in range(n_dates):
        # Step 1: Daily risk exits on existing holdings (Chandelier stop breach or trend loss)
        surviving_w = current_w.copy()
        for j in range(n_tickers):
            if surviving_w[j] > 0.0:
                if breach_arr[t, j] or (not trend_arr[t, j]):
                    surviving_w[j] = 0.0

        # Step 2: Weekly rebalance on Monday (or first trading day of week)
        if rebal_arr[t]:
            next_w = np.zeros(n_tickers, dtype=float)
            # Determine ordered candidate indices by momentum rank
            cand_indices = [
                j
                for j in range(n_tickers)
                if eligible_arr[t, j] and trend_arr[t, j] and (not breach_arr[t, j]) and raw_sized[t, j] > 0.0
            ]
            cand_indices.sort(key=lambda j: ranks_arr[t, j])
            cand_indices = cand_indices[: cfg.max_active_holdings]

            if macro_arr[t]:
                for j in cand_indices:
                    was_held = surviving_w[j] > 0.0
                    if not was_held:
                        # New entry: must NOT be in 5-day pre-earnings blackout
                        if blackout_arr[t, j]:
                            continue
                        next_w[j] = min(cfg.max_weight_per_stock, raw_sized[t, j])
                    else:
                        # Existing winner: if in earnings blackout, do not increase risk
                        if blackout_arr[t, j]:
                            next_w[j] = min(surviving_w[j], raw_sized[t, j], cfg.max_weight_per_stock)
                        else:
                            next_w[j] = min(cfg.max_weight_per_stock, raw_sized[t, j])
            else:
                # SPY <= EMA 200: Freeze new entries; retain existing valid winners without adding risk
                for j in range(n_tickers):
                    if surviving_w[j] > 0.0:
                        sized_j = raw_sized[t, j] if raw_sized[t, j] > 0.0 else surviving_w[j]
                        next_w[j] = min(surviving_w[j], sized_j, cfg.max_weight_per_stock)

            total_w = float(next_w.sum())
            if total_w > 1.0:
                next_w = next_w / total_w
            next_w = np.minimum(next_w, cfg.max_weight_per_stock)
            current_w = next_w
        else:
            current_w = surviving_w

        out_weights[t, :] = current_w

    return pd.DataFrame(out_weights, index=dates, columns=tickers)


class PortfolioEngine:
    """Self-financing T+1 execution ledger with 5 bps transaction costs and 2% cash park."""

    def __init__(
        self,
        opens: pd.DataFrame,
        closes: pd.DataFrame,
        config: EngineConfig | None = None,
    ) -> None:
        if not opens.index.equals(closes.index):
            raise ValueError("opens and closes must share the exact trading calendar.")
        if list(opens.columns) != list(closes.columns):
            raise ValueError("opens and closes must have identical asset columns.")
        self._config = config or EngineConfig()
        self._columns = list(opens.columns)
        self._index = closes.index
        self._opens = opens.to_numpy(dtype=float)
        self._closes = closes.to_numpy(dtype=float)
        if not np.all(self._opens > 0.0) or not np.all(self._closes > 0.0):
            raise ValueError("Execution prices (Open/Close) must be strictly positive.")

    def run(
        self,
        target_weights: pd.DataFrame,
        start: str | pd.Timestamp | None = None,
        cost_bps: float | None = None,
        execution_lag: int | None = None,
    ) -> Ledger:
        """Execute ``target_weights`` with strict ``T + execution_lag`` delay."""
        if list(target_weights.columns) != self._columns:
            raise ValueError("target_weights columns must match PortfolioEngine asset columns.")
        lag = self._config.execution_lag if execution_lag is None else execution_lag
        if lag < 1:
            raise ValueError("execution_lag must be >= 1 (strict T+1 requirement; zero look-ahead).")
        cost_rate = (self._config.cost_bps if cost_bps is None else cost_bps) * 1e-4

        run_index = self._index
        if start is not None:
            run_index = run_index[run_index >= pd.Timestamp(start)]
        if len(run_index) == 0:
            raise ValueError("Backtest window contains zero trading sessions.")

        positions = self._index.get_indexer(run_index)
        opens = self._opens[positions]
        closes = self._closes[positions]

        # Enforce hard 15% cap and [0, 1] portfolio sum on raw signal weights before shifting
        clean_signals = target_weights.clip(lower=0.0, upper=self._config.max_weight_per_stock)
        row_sums = clean_signals.sum(axis=1)
        over_one = row_sums > 1.0
        if over_one.any():
            clean_signals.loc[over_one] = clean_signals.loc[over_one].div(row_sums.loc[over_one], axis=0)

        # Shift by `lag` on the full calendar so signals at Close(t - lag) execute at Open(t)
        executed_df = clean_signals.shift(lag).reindex(run_index).fillna(0.0)
        w_exec = executed_df.to_numpy(dtype=float)

        n_assets = len(self._columns)
        nav = float(self._config.initial_capital)
        values = np.zeros(n_assets, dtype=float)
        cash = float(nav)
        daily_yield = (1.0 + self._config.cash_yield_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

        rows: list[dict[str, float]] = []
        for pos in range(len(run_index)):
            prev_nav = nav
            if pos == 0:
                open_values = np.zeros(n_assets, dtype=float)
            else:
                open_values = values * (opens[pos] / closes[pos - 1])
            open_nav = float(open_values.sum()) + cash
            overnight_ret = open_nav / prev_nav - 1.0

            w = w_exec[pos]
            post = open_nav
            for _ in range(_MAX_COST_ITERATIONS):
                traded = float(np.abs(w * post - open_values).sum())
                nxt = open_nav - cost_rate * traded
                if abs(nxt - post) <= 1e-12 * max(1.0, open_nav):
                    post = nxt
                    break
                post = nxt
            traded = float(np.abs(w * post - open_values).sum())
            friction = cost_rate * traded
            if abs(post + friction - open_nav) > 1e-8 * max(1.0, open_nav):
                raise AssertionError("Self-financing identity failed at rebalance.")

            new_open = w * post
            new_cash = max(0.0, post - float(new_open.sum()))
            close_values = new_open * (closes[pos] / opens[pos])
            close_nav = float(close_values.sum()) + new_cash
            pre_nav = open_nav - friction
            intraday_ret = close_nav / pre_nav - 1.0 if pre_nav > 0.0 else 0.0

            cash_weight = max(0.0, 1.0 - float(w.sum()))
            cash_yield = cash_weight * post * daily_yield
            nav = close_nav + cash_yield

            values = close_values
            cash = new_cash + cash_yield

            rows.append(
                {
                    "prev_nav": prev_nav,
                    "open_nav": open_nav,
                    "overnight_ret": overnight_ret,
                    "friction": friction,
                    "intraday_ret": intraday_ret,
                    "cash_yield": cash_yield,
                    "net_ret": nav / prev_nav - 1.0,
                    "turnover": traded / open_nav if open_nav > 0.0 else 0.0,
                    "exposure": float(w.sum()),
                }
            )

        components = pd.DataFrame(rows, index=run_index)
        ledger = Ledger(
            returns=components["net_ret"].astype(float).rename("net_return"),
            nav=pd.Series(
                components["prev_nav"].to_numpy() * (1.0 + components["net_ret"].to_numpy()),
                index=run_index,
                name="nav",
            ),
            weights=pd.DataFrame(w_exec, index=run_index, columns=self._columns),
            signal_weights=clean_signals.reindex(run_index).fillna(0.0),
            turnover=components["turnover"].rename("turnover"),
            friction=components["friction"].rename("friction"),
            components=components,
        )
        self._verify_invariants(ledger, cost_rate, lag, clean_signals)
        return ledger

    def _verify_invariants(
        self,
        ledger: Ledger,
        cost_rate: float,
        lag: int,
        clean_signals: pd.DataFrame,
    ) -> None:
        w = ledger.weights.to_numpy(dtype=float)
        if (w < -1e-12).any() or (w > self._config.max_weight_per_stock + 1e-9).any():
            raise AssertionError(
                f"Executed weights breached max single-stock cap ({self._config.max_weight_per_stock:.2%})."
            )
        if not np.isfinite(ledger.returns).all():
            raise AssertionError("Non-finite returns encountered in ledger.")
        expected = clean_signals.shift(lag).reindex(ledger.returns.index).fillna(0.0)
        np.testing.assert_allclose(
            w,
            expected.to_numpy(dtype=float),
            atol=1e-12,
            err_msg="Execution lag invariant violated: executed weights != signal(t - lag).",
        )
        recon = (
            ledger.friction.to_numpy()
            - cost_rate * ledger.turnover.to_numpy() * ledger.components["open_nav"].to_numpy()
        )
        if float(np.abs(recon).max()) > 1e-6:
            raise AssertionError("Transaction cost reconciliation failed.")
