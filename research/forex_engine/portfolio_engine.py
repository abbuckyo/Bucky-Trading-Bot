# -*- coding: utf-8 -*-
"""Inverse-Volatility Position Sizing & Symmetric Long/Short T+1 Execution Ledger.

1. Volatility-Targeted Sizing:
   - Target annualized volatility per active pair = 2.5% of portfolio NAV.
   - ``Weight_{i,t} = Direction_{i,t} * min(0.20, 0.025 / (sigma_{20D,i} * sqrt(252)))``
   - Gross leverage normalization: if ``sum(|w_i|) > 1.0``, scale proportionally
     so ``sum(|w_i|) <= 1.0``.
2. Symmetric Long/Short T+1 Execution Ledger:
   - Strict T+1 execution lag (signals at Close(t) execute for session t+1).
   - Supports signed weights in ``[-0.20, +0.20]``: Long (+w) profits when pair
     appreciates; Short (-w) profits when pair depreciates.
   - Deducts 3 bps (0.03%) per unit of rebalance turnover ``sum(|w_t - w_{t-1}|)``.
   - Unallocated gross collateral ``(1.0 - sum(|w_i|))`` accrues 2.0% annual yield.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_CASH_YIELD,
    DEFAULT_COST_BPS,
    ForexConfig,
    MAX_GROSS_LEVERAGE,
    MAX_PAIR_WEIGHT,
    TARGET_PAIR_VOL_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)


@dataclass
class ForexLedger:
    """Full accounting record of a G10 Forex backtest run."""

    returns: pd.Series
    nav: pd.Series
    weights: pd.DataFrame          # executed signed weights (T+lag)
    signal_weights: pd.DataFrame   # pre-lag signal weights at Close(t)
    turnover: pd.Series            # one-way notional turnover per session
    friction: pd.Series            # transaction cost in currency units
    components: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def gross_exposure(self) -> pd.Series:
        return self.weights.abs().sum(axis=1).rename("gross_exposure")

    @property
    def net_exposure(self) -> pd.Series:
        return self.weights.sum(axis=1).rename("net_exposure")

    @property
    def cash_weight(self) -> pd.Series:
        return (1.0 - self.gross_exposure).clip(lower=0.0).rename("cash_weight")


def compute_target_weights(
    direction_signal: pd.DataFrame,
    vol20_annual: pd.DataFrame,
    target_pair_vol: float = TARGET_PAIR_VOL_ANNUAL,
    max_pair_weight: float = MAX_PAIR_WEIGHT,
    max_gross_leverage: float = MAX_GROSS_LEVERAGE,
) -> pd.DataFrame:
    """Compute signed inverse-volatility target weights obeying per-pair and gross caps.

    Formula:
        Weight_{i,t} = Direction_{i,t} * min(0.20, 0.025 / vol20_annual_{i,t})
        Normalized so sum_i |Weight_{i,t}| <= 1.0.
    """
    if not direction_signal.index.equals(vol20_annual.index):
        raise ValueError("direction_signal and vol20_annual must share the same index")

    d_arr = np.clip(direction_signal.to_numpy(dtype=float), -1.0, 1.0)
    v_arr = vol20_annual.to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_mag = np.where(
            np.isfinite(v_arr) & (v_arr > 0.0),
            np.minimum(max_pair_weight, target_pair_vol / v_arr),
            0.0,
        )
    w_mat = d_arr * raw_mag
    w_mat = np.clip(np.nan_to_num(w_mat, nan=0.0), -max_pair_weight, max_pair_weight)

    gross = np.abs(w_mat).sum(axis=1, keepdims=True)
    scale = np.where(gross > max_gross_leverage, max_gross_leverage / np.maximum(gross, 1e-12), 1.0)
    w_mat = w_mat * scale

    return pd.DataFrame(w_mat, index=direction_signal.index, columns=direction_signal.columns)


class ForexPortfolioEngine:
    """Symmetric Long/Short T+1 execution ledger for G10 Forex."""

    def __init__(
        self,
        closes: pd.DataFrame,
        config: ForexConfig | None = None,
        opens: pd.DataFrame | None = None,
    ) -> None:
        self._config = config or ForexConfig()
        self._columns = list(closes.columns)
        self._index = closes.index
        self._closes = closes.to_numpy(dtype=float)
        if not np.all(self._closes > 0.0):
            raise ValueError("FX close prices must be strictly positive.")
        if opens is not None and (not opens.index.equals(closes.index)):
            raise ValueError("opens and closes must share the same trading calendar.")

    def run(
        self,
        target_weights: pd.DataFrame,
        start: str | pd.Timestamp | None = None,
        cost_bps: float | None = None,
        execution_lag: int | None = None,
    ) -> ForexLedger:
        """Execute signed ``target_weights`` with strict ``T + execution_lag`` delay."""
        if list(target_weights.columns) != self._columns:
            raise ValueError("target_weights columns must match ForexPortfolioEngine pairs.")
        lag = self._config.execution_lag if execution_lag is None else execution_lag
        if lag < 1:
            raise ValueError("execution_lag must be >= 1 (strict T+1 requirement; zero look-ahead).")
        cost_rate = (self._config.cost_bps if cost_bps is None else cost_bps) * 1e-4

        # Enforce per-pair [-0.20, +0.20] and gross leverage <= 1.0 on signal weights
        clean_sig = target_weights.clip(
            lower=-self._config.max_pair_weight,
            upper=self._config.max_pair_weight,
        ).fillna(0.0)
        gross_sig = clean_sig.abs().sum(axis=1)
        over_gross = gross_sig > self._config.max_gross_leverage
        if over_gross.any():
            clean_sig.loc[over_gross] = clean_sig.loc[over_gross].mul(
                self._config.max_gross_leverage / gross_sig.loc[over_gross], axis=0
            )

        # Pair daily spot returns r_{i,t} = Close(t) / Close(t-1) - 1
        pair_rets_full = pd.DataFrame(
            self._closes, index=self._index, columns=self._columns
        ).pct_change(fill_method=None).fillna(0.0)

        run_index = self._index
        if start is not None:
            run_index = run_index[run_index >= pd.Timestamp(start)]
        if len(run_index) == 0:
            raise ValueError("Empty backtest window.")

        # Shift on FULL calendar so Close(t - lag) signal governs session t return
        executed_df = clean_sig.shift(lag).reindex(run_index).fillna(0.0)
        w_exec = executed_df.to_numpy(dtype=float)
        r_mat = pair_rets_full.reindex(run_index).to_numpy(dtype=float)

        n_sessions, n_pairs = w_exec.shape
        nav = float(self._config.initial_capital)
        prev_w = np.zeros(n_pairs, dtype=float)
        daily_rf = (1.0 + self._config.cash_yield_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

        rows: list[dict[str, float]] = []
        for t in range(n_sessions):
            prev_nav = nav
            w = w_exec[t]

            # Rebalance turnover & friction
            turnover = float(np.abs(w - prev_w).sum())
            friction_ret = cost_rate * turnover
            friction_dollars = prev_nav * friction_ret

            # Symmetric Long/Short PnL on active currency pairs
            fx_pnl_ret = float(np.dot(w, r_mat[t]))

            # Unencumbered cash yield on (1 - gross_exposure)
            gross_exp = float(np.abs(w).sum())
            cash_w = max(0.0, 1.0 - gross_exp)
            cash_ret = cash_w * daily_rf

            net_ret = fx_pnl_ret + cash_ret - friction_ret
            nav = prev_nav * (1.0 + net_ret)

            # Update end-of-day drifted weights before next session's rebalance
            if abs(1.0 + net_ret) > 1e-12:
                prev_w = w * (1.0 + r_mat[t]) / (1.0 + net_ret)
            else:
                prev_w = w.copy()

            rows.append(
                {
                    "prev_nav": prev_nav,
                    "fx_pnl_ret": fx_pnl_ret,
                    "cash_ret": cash_ret,
                    "friction": friction_dollars,
                    "net_ret": net_ret,
                    "turnover": turnover,
                    "gross_exposure": gross_exp,
                    "net_exposure": float(w.sum()),
                }
            )

        comp = pd.DataFrame(rows, index=run_index)
        ledger = ForexLedger(
            returns=comp["net_ret"].astype(float).rename("net_return"),
            nav=pd.Series(
                comp["prev_nav"].to_numpy() * (1.0 + comp["net_ret"].to_numpy()),
                index=run_index,
                name="nav",
            ),
            weights=pd.DataFrame(w_exec, index=run_index, columns=self._columns),
            signal_weights=clean_sig.reindex(run_index).fillna(0.0),
            turnover=comp["turnover"].rename("turnover"),
            friction=comp["friction"].rename("friction"),
            components=comp,
        )
        self._verify_invariants(ledger, lag, clean_sig)
        return ledger

    def _verify_invariants(
        self,
        ledger: ForexLedger,
        lag: int,
        clean_sig: pd.DataFrame,
    ) -> None:
        w = ledger.weights.to_numpy(dtype=float)
        if (np.abs(w) > self._config.max_pair_weight + 1e-9).any():
            raise AssertionError("Per-pair weight exceeded max_pair_weight (20.0%).")
        if (ledger.gross_exposure.to_numpy() > self._config.max_gross_leverage + 1e-9).any():
            raise AssertionError("Gross exposure exceeded max_gross_leverage (1.0x).")
        if not np.isfinite(ledger.returns).all():
            raise AssertionError("Non-finite returns detected in ForexLedger.")

        expected = clean_sig.shift(lag).reindex(ledger.returns.index).fillna(0.0)
        np.testing.assert_allclose(
            w,
            expected.to_numpy(dtype=float),
            atol=1e-12,
            err_msg="Strict T+lag invariant violated in ForexLedger.",
        )
