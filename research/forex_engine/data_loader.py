# -*- coding: utf-8 -*-
"""Sequential Yahoo Finance data loader & Parquet cache for G10 Forex Engine.

Fetches G10 spot FX proxies (``EURUSD=X``, ``GBPUSD=X``, ``USDJPY=X``,
``AUDUSD=X``, ``USDCAD=X``, ``USDCHF=X``, ``NZDUSD=X``) plus macro references
(``DX-Y.NYB`` and ``SPY``) sequentially with ``threads=False`` and caches the
aligned dataset to ``research/forex_engine/cache/g10_forex_ohlc.parquet``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .config import ForexConfig

_REQUIRED_OHLC = ("Open", "High", "Low", "Close")


@dataclass(slots=True)
class ForexMarketBundle:
    """Calendar-aligned G10 FX OHLC data and macro benchmark series."""

    opens: pd.DataFrame     # columns = G10_UNIVERSE
    highs: pd.DataFrame     # columns = G10_UNIVERSE
    lows: pd.DataFrame      # columns = G10_UNIVERSE
    closes: pd.DataFrame    # columns = G10_UNIVERSE
    dxy_close: pd.Series    # DX-Y.NYB close aligned to calendar
    spy_close: pd.Series    # SPY close aligned to calendar

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self.closes.index

    @property
    def pairs(self) -> list[str]:
        return list(self.closes.columns)


class ForexDataLoader:
    """Sequential Yahoo Finance downloader with local Parquet caching."""

    def __init__(self, config: ForexConfig | None = None) -> None:
        self._config = config or ForexConfig()
        self._cache_dir = Path(self._config.cache_dir)
        self._cache_path = self._cache_dir / "g10_forex_ohlc.parquet"

    def _required_tickers(self) -> list[str]:
        syms = list(self._config.universe)
        for extra in (self._config.dollar_index, self._config.equity_benchmark):
            if extra and extra not in syms:
                syms.append(extra)
        return syms

    def load(self) -> ForexMarketBundle:
        """Load aligned G10 FX market bundle from Parquet cache or sequential fetch."""
        snapshot = None if self._config.refresh else self._load_cache()
        if snapshot is None:
            snapshot = self._fetch_all()
            self._save_cache(snapshot)
        return self._assemble(snapshot)

    def _fetch_end(self) -> str:
        if self._config.backtest_end is not None:
            return str(pd.Timestamp(self._config.backtest_end).date())
        return str(pd.Timestamp.now(tz="America/New_York").date())

    def _fetch_one(self, symbol: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                df = yf.download(
                    symbol,
                    start=self._config.data_start,
                    end=self._fetch_end(),
                    auto_adjust=True,
                    actions=False,
                    progress=False,
                    threads=False,
                    timeout=30,
                )
                return self.validate_raw_ohlc(symbol, df)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self._config.max_retries:
                    print(f"  [ForexDataLoader] {symbol}: attempt {attempt} failed ({exc}); retrying...")
                    time.sleep(self._config.retry_delay_s * attempt)
        raise RuntimeError(f"Sequential FX download failed for {symbol}: {last_error}") from last_error

    def _fetch_all(self) -> pd.DataFrame:
        tickers = self._required_tickers()
        print(
            f"[ForexDataLoader] Sequentially downloading {len(tickers)} FX/macro tickers "
            f"from {self._config.data_start} -> {self._fetch_end()} ..."
        )
        frames: dict[str, pd.DataFrame] = {}
        for idx, sym in enumerate(tickers, start=1):
            print(f"  [{idx:02d}/{len(tickers):02d}] fetching {sym}")
            frames[sym] = self._fetch_one(sym)
            time.sleep(0.15)
        return pd.concat(frames, names=("Ticker", "Field")).sort_index()

    @staticmethod
    def validate_raw_ohlc(symbol: str, df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError(f"{symbol}: empty OHLC dataframe")
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            field_level = None
            for lvl in range(out.columns.nlevels):
                vals = out.columns.get_level_values(lvl)
                if all(f in vals for f in _REQUIRED_OHLC):
                    field_level = lvl
                    break
            if field_level is None:
                raise ValueError(f"{symbol}: cannot locate OHLC level in MultiIndex")
            out.columns = out.columns.get_level_values(field_level)
        missing = [f for f in _REQUIRED_OHLC if f not in out.columns]
        if missing:
            raise ValueError(f"{symbol}: missing required OHLC fields {missing}")

        out = out[list(_REQUIRED_OHLC)].copy()
        out.index = pd.to_datetime(out.index)
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out.index = out.index.normalize()
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out = out.dropna(how="any").astype(float)
        if (out <= 0.0).any().any():
            raise ValueError(f"{symbol}: non-positive prices detected")

        # Enforce OHLC envelope consistency (High >= max(O,C) and Low <= min(O,C))
        hi = np.maximum.reduce([out["Open"].to_numpy(), out["High"].to_numpy(), out["Low"].to_numpy(), out["Close"].to_numpy()])
        lo = np.minimum.reduce([out["Open"].to_numpy(), out["High"].to_numpy(), out["Low"].to_numpy(), out["Close"].to_numpy()])
        out["High"] = hi
        out["Low"] = lo
        return out

    def _save_cache(self, snapshot: pd.DataFrame) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot.to_parquet(self._cache_path)
        print(f"  [ForexDataLoader] Saved Parquet cache -> {self._cache_path}")

    def _load_cache(self) -> pd.DataFrame | None:
        if not self._cache_path.exists():
            return None
        try:
            snap = pd.read_parquet(self._cache_path)
            cached = set(snap.index.get_level_values("Ticker").unique())
            if not set(self._required_tickers()).issubset(cached):
                return None
            print(f"  [ForexDataLoader] Loaded FX snapshot from cache: {self._cache_path}")
            return snap
        except Exception:  # noqa: BLE001
            return None

    def _assemble(self, snapshot: pd.DataFrame) -> ForexMarketBundle:
        pairs = list(self._config.universe)
        dxy_sym = self._config.dollar_index
        spy_sym = self._config.equity_benchmark

        def _field_df(field_name: str, symbols: list[str]) -> pd.DataFrame:
            return pd.DataFrame(
                {sym: snapshot.xs(sym, level="Ticker")[field_name] for sym in symbols}
            ).sort_index()

        # Align on SPY trading sessions where all G10 FX pairs have valid observations
        spy_close_raw = snapshot.xs(spy_sym, level="Ticker")["Close"].sort_index()
        dxy_close_raw = snapshot.xs(dxy_sym, level="Ticker")["Close"].sort_index()

        closes_fx = _field_df("Close", pairs).reindex(spy_close_raw.index).ffill(limit=2)
        opens_fx = _field_df("Open", pairs).reindex(spy_close_raw.index).ffill(limit=2)
        highs_fx = _field_df("High", pairs).reindex(spy_close_raw.index).ffill(limit=2)
        lows_fx = _field_df("Low", pairs).reindex(spy_close_raw.index).ffill(limit=2)
        dxy_aligned = dxy_close_raw.reindex(spy_close_raw.index).ffill(limit=3)

        valid = (
            closes_fx.notna().all(axis=1)
            & opens_fx.notna().all(axis=1)
            & highs_fx.notna().all(axis=1)
            & lows_fx.notna().all(axis=1)
            & spy_close_raw.notna()
            & dxy_aligned.notna()
        )
        idx = spy_close_raw.index[valid]
        if len(idx) < self._config.warmup_days_required:
            raise RuntimeError(f"Insufficient aligned FX sessions ({len(idx)}).")

        return ForexMarketBundle(
            opens=opens_fx.loc[idx].astype(float),
            highs=highs_fx.loc[idx].astype(float),
            lows=lows_fx.loc[idx].astype(float),
            closes=closes_fx.loc[idx].astype(float),
            dxy_close=dxy_aligned.loc[idx].astype(float).rename(dxy_sym),
            spy_close=spy_close_raw.loc[idx].astype(float).rename(spy_sym),
        )
