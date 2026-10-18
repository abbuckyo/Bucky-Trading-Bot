# -*- coding: utf-8 -*-
"""Module C — Portfolio construction, execution ledger & accounting.

Strict T+1 execution ledger: signals computed at Close(t) are deployed at
Open(t+lag). Each session is split into an Overnight leg (previous Close ->
today's Open, on *old* holdings) and an Intraday Post-Rebalance leg
(today's Open -> Close on *new* holdings). Friction (5 bps default) is
deducted on absolute turnover of the rebalance. Unallocated weight is
parked in a money-market yield (default 2.0% annualized, proxying the
SCBTMFPLUS-E / BIL cash park).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PortfolioConfig, TRADING_DAYS_PER_YEAR

_MAX_COST_ITERATIONS = 60


@dataclass
class Ledger:
    """Full accounting trail of one backtest run."""

    returns: pd.Series  # net daily portfolio return
    nav: pd.Series  # closing NAV in currency units
    weights: pd.DataFrame  # executed target weights per asset
    turnover: pd.Series  # traded notional / open NAV per session
    friction: pd.Series  # friction paid per session (currency units)
    components: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def risky_exposure(self) -> pd.Series:
        return self.weights.sum(axis=1).rename("exposure")

    @property
    def cash_weight(self) -> pd.Series:
        return (1.0 - self.risky_exposure).clip(lower=0.0).rename("cash_weight")


class PortfolioEngine:
    """Rebalance an Equal-Cap tranche portfolio with strict execution lag."""

    def __init__(self, opens: pd.DataFrame, closes: pd.DataFrame, config: PortfolioConfig) -> None:
        if not opens.index.equals(closes.index):
            raise ValueError("opens and closes must share one aligned calendar")
        if list(opens.columns) != list(closes.columns):
            raise ValueError("opens and closes must cover the same assets")
        self._config = config
        self._columns = list(opens.columns)
        self._index = closes.index
        self._opens = opens.to_numpy(dtype=float)
        self._closes = closes.to_numpy(dtype=float)
        if not np.all(self._opens > 0.0) or not np.all(self._closes > 0.0):
            raise ValueError("non-positive prices in execution data")

    # -- public API ---------------------------------------------------------

    def run(
        self,
        target_weights: pd.DataFrame,
        start: str | pd.Timestamp | None = None,
        cost_bps: float | None = None,
        execution_lag: int | None = None,
    ) -> Ledger:
        """Simulate the strategy from ``start`` (default: calendar start).

        ``cost_bps`` / ``execution_lag`` overrides exist only for stress
        tests (2x cost, T+2); defaults come from the frozen config.
        """
        if list(target_weights.columns) != self._columns:
            raise ValueError("target weights columns must match execution assets")
        lag = self._config.execution_lag if execution_lag is None else execution_lag
        if lag < 1:
            raise ValueError("execution lag must be >= 1 (strict T+1 invariant)")
        cost = (self._config.cost_bps if cost_bps is None else cost_bps) * 1e-4

        run_index = self._index
        if start is not None:
            run_index = run_index[run_index >= pd.Timestamp(start)]
        if len(run_index) == 0:
            raise ValueError("empty backtest window")
        positions = self._index.get_indexer(run_index)
        opens = self._opens[positions]
        closes = self._closes[positions]

        # Shift on the FULL signal calendar so pre-window signals feed the
        # first execution day with zero look-ahead: Close(t-lag) -> Open(t).
        executed = target_weights.shift(lag).reindex(run_index)
        w_exec = executed.to_numpy(dtype=float)
        w_exec = np.where(np.isfinite(w_exec), w_exec, 0.0)

        n_assets = len(self._columns)
        nav = self._config.initial_capital
        values = np.zeros(n_assets, dtype=float)  # risky book; book starts in cash
        cash = float(nav)
        rows: list[dict[str, float]] = []
        daily_yield = (1.0 + self._config.cash_yield_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

        for pos in range(len(run_index)):
            prev_nav = nav
            # Overnight leg: old holdings marked from prev close to today's open.
            if pos == 0:  # fresh book, fully in cash: nothing to mark
                open_values = np.zeros(n_assets, dtype=float)
            else:
                open_values = values * (opens[pos] / closes[pos - 1])
            open_nav = float(open_values.sum()) + cash
            overnight_ret = open_nav / prev_nav - 1.0

            # Fixed-point solve: postcost NAV satisfies
            #   post = open_nav - cost * sum(|w*post - old_open|)
            # because the traded notional itself depends on the post-cost NAV.
            w = w_exec[pos]
            post = open_nav
            for _ in range(_MAX_COST_ITERATIONS):
                traded = float(np.abs(w * post - open_values).sum())
                nxt = open_nav - cost * traded
                if abs(nxt - post) <= 1e-12 * max(1.0, open_nav):
                    post = nxt
                    break
                post = nxt
            traded = float(np.abs(w * post - open_values).sum())
            friction = cost * traded
            if abs(post + friction - open_nav) > 1e-8 * max(1.0, open_nav):
                raise AssertionError("self-financing reconciliation failed")

            # Intraday leg: new holdings from open to close.
            new_open = w * post
            new_cash = max(0.0, post - float(new_open.sum()))
            close_values = new_open * (closes[pos] / opens[pos])
            close_nav = float(close_values.sum()) + new_cash
            pre_nav = open_nav - friction
            intraday_ret = close_nav / pre_nav - 1.0 if pre_nav > 0 else 0.0

            # Cash park: money-market accrual on the cash weight held into
            # the close (one accrual per session).
            cash_weight = max(0.0, 1.0 - float(w.sum()))
            cash_yield = cash_weight * post * daily_yield
            nav = close_nav + cash_yield

            values = close_values
            cash = new_cash + cash_yield  # yield accrues into the cash balance

            rows.append(
                {
                    "prev_nav": prev_nav,
                    "open_nav": open_nav,
                    "overnight_ret": overnight_ret,
                    "friction": friction,
                    "intraday_ret": intraday_ret,
                    "cash_yield": cash_yield,
                    "net_ret": nav / prev_nav - 1.0,
                    "turnover": traded / open_nav,
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
            turnover=components["turnover"].rename("turnover"),
            friction=components["friction"].rename("friction"),
            components=components,
        )
        self._verify_invariants(ledger, cost, lag, target_weights)
        return ledger

    # -- invariant audit -----------------------------------------------------

    def _verify_invariants(
        self,
        ledger: Ledger,
        cost: float,
        lag: int,
        target_weights: pd.DataFrame,
    ) -> None:
        w = ledger.weights.to_numpy(dtype=float)
        if (w < -1e-12).any() or (w > 0.20 + 1e-9).any():
            raise AssertionError("executed weights violate the equal-cap 20% bound")
        if np.isfinite(ledger.returns).all() is False:
            raise AssertionError("non-finite ledger returns")
        if (ledger.returns <= -1.0 + 1e-12).any():
            raise AssertionError("ledger return below -100%")
        # Strict T+lag: executed weights must be exactly signal(t - lag).
        expected = target_weights.shift(lag).reindex(ledger.returns.index).fillna(0.0)
        np.testing.assert_allclose(
            ledger.weights.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            atol=1e-12,
            err_msg="execution lag violated: weights are not signal(t - lag)",
        )
        # Friction identity: friction == cost * turnover * open NAV.
        recon = ledger.friction.to_numpy() - cost * ledger.turnover.to_numpy() * ledger.components[
            "open_nav"
        ].to_numpy()
        if float(np.abs(recon).max()) > 1e-6:
            raise AssertionError("friction reconciliation failed")
