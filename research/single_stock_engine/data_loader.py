# -*- coding: utf-8 -*-
"""Data ingestion, Parquet caching, calendar validation & earnings schedule loader.

Strictly sequential/synchronous calls to Yahoo Finance (``threads=False``) to
avoid HTTP 429 rate limits. Caches aligned OHLC snapshots and earnings dates as
local ``.parquet`` files under ``research/single_stock_engine/cache/``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import yfinance as yf

from .config import EngineConfig

_REQUIRED_OHLC = ("Open", "High", "Low", "Close")


@dataclass(slots=True)
class MarketBundle:
    """Calendar-aligned OHLC market bundle + earnings announcement dates."""

    opens: pd.DataFrame       # columns = universe tickers
    highs: pd.DataFrame       # columns = universe tickers
    lows: pd.DataFrame        # columns = universe tickers
    closes: pd.DataFrame      # columns = universe tickers
    spy_close: pd.Series      # benchmark close series aligned to ``closes.index``
    earnings_dates: dict[str, pd.DatetimeIndex] = field(default_factory=dict)

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self.closes.index

    @property
    def tickers(self) -> list[str]:
        return list(self.closes.columns)


class DataLoader:
    """Sequential Yahoo Finance fetcher with local Parquet cache and validation."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self._config = config or EngineConfig()
        self._cache_dir = Path(self._config.cache_dir)
        self._ohlc_cache_path = self._cache_dir / "single_stock_ohlc.parquet"
        self._earnings_cache_path = self._cache_dir / "earnings_dates.parquet"

    def _required_tickers(self) -> list[str]:
        tickers = list(self._config.universe)
        if self._config.benchmark not in tickers:
            tickers.append(self._config.benchmark)
        return tickers

    def load(self, fetch_earnings: bool = True) -> MarketBundle:
        """Load aligned market data from Parquet cache or sequential download."""
        snapshot = None if self._config.refresh else self._load_ohlc_cache()
        if snapshot is None:
            snapshot = self._fetch_all_ohlc()
            self._save_ohlc_cache(snapshot)

        earnings_map: dict[str, pd.DatetimeIndex] = {}
        if fetch_earnings:
            earnings_map = (
                {} if self._config.refresh else self._load_earnings_cache()
            )
            if not earnings_map:
                earnings_map = self._fetch_all_earnings()
                self._save_earnings_cache(earnings_map)

        return self._assemble(snapshot, earnings_map)

    # -- sequential OHLC fetching -------------------------------------------

    def _fetch_end(self) -> str:
        if self._config.backtest_end is not None:
            return str(pd.Timestamp(self._config.backtest_end).date())
        return str(pd.Timestamp.now(tz="America/New_York").date())

    def _fetch_one_ohlc(self, symbol: str) -> pd.DataFrame:
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
                    print(f"  [DataLoader] {symbol}: attempt {attempt} failed ({exc}); retrying...")
                    time.sleep(self._config.retry_delay_s * attempt)
        raise RuntimeError(f"Sequential download failed for {symbol}: {last_error}") from last_error

    def _fetch_all_ohlc(self) -> pd.DataFrame:
        tickers = self._required_tickers()
        print(
            f"[DataLoader] Sequentially downloading {len(tickers)} tickers "
            f"from {self._config.data_start} -> {self._fetch_end()} ..."
        )
        frames: dict[str, pd.DataFrame] = {}
        for idx, symbol in enumerate(tickers, start=1):
            print(f"  [{idx:02d}/{len(tickers):02d}] fetching {symbol}")
            frames[symbol] = self._fetch_one_ohlc(symbol)
            time.sleep(0.15)  # polite pacing to prevent HTTP 429
        snapshot = pd.concat(frames, names=("Ticker", "Field")).sort_index()
        return snapshot

    @staticmethod
    def validate_raw_ohlc(symbol: str, df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError(f"{symbol}: empty OHLC response")
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            field_level = None
            for lvl in range(out.columns.nlevels):
                vals = out.columns.get_level_values(lvl)
                if all(f in vals for f in _REQUIRED_OHLC):
                    field_level = lvl
                    break
            if field_level is None:
                raise ValueError(f"{symbol}: cannot locate OHLC columns in MultiIndex")
            out.columns = out.columns.get_level_values(field_level)
        missing = [f for f in _REQUIRED_OHLC if f not in out.columns]
        if missing:
            raise ValueError(f"{symbol}: missing required OHLC fields {missing}")
        out = out[list(_REQUIRED_OHLC)].copy()
        out.index = pd.to_datetime(out.index)
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out = out.dropna(how="any")
        if (out <= 0.0).any().any():
            raise ValueError(f"{symbol}: non-positive OHLC prices found")
        return out.astype(float)

    # -- sequential earnings date fetching ----------------------------------

    def _fetch_all_earnings(self) -> dict[str, pd.DatetimeIndex]:
        result: dict[str, pd.DatetimeIndex] = {}
        print("[DataLoader] Fetching historical & upcoming earnings dates sequentially...")
        for symbol in self._config.universe:
            dates: list[pd.Timestamp] = []
            try:
                ticker_obj = yf.Ticker(symbol)
                ed = ticker_obj.get_earnings_dates(limit=60)
                if ed is not None and len(ed) > 0:
                    idx = pd.to_datetime(ed.index)
                    if idx.tz is not None:
                        idx = idx.tz_localize(None)
                    dates.extend(idx.normalize().tolist())
            except Exception:  # noqa: BLE001
                pass
            result[symbol] = pd.DatetimeIndex(sorted(set(dates)))
            time.sleep(0.10)
        return result

    # -- Parquet cache I/O --------------------------------------------------

    def _save_ohlc_cache(self, snapshot: pd.DataFrame) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot.to_parquet(self._ohlc_cache_path)
        print(f"  [DataLoader] OHLC cache saved -> {self._ohlc_cache_path}")

    def _load_ohlc_cache(self) -> pd.DataFrame | None:
        if not self._ohlc_cache_path.exists():
            return None
        try:
            snapshot = pd.read_parquet(self._ohlc_cache_path)
            cached_tickers = set(snapshot.index.get_level_values("Ticker").unique())
            if not set(self._required_tickers()).issubset(cached_tickers):
                print("  [DataLoader] Cache miss: required tickers not all present in cache.")
                return None
            print(f"  [DataLoader] Loaded OHLC snapshot from cache: {self._ohlc_cache_path}")
            return snapshot
        except Exception as exc:  # noqa: BLE001
            print(f"  [DataLoader] Cache unreadable ({exc}); refetching...")
            return None

    def _save_earnings_cache(self, earnings_map: Mapping[str, pd.DatetimeIndex]) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for sym, dts in earnings_map.items():
            for dt in dts:
                rows.append({"Ticker": sym, "EarningsDate": pd.Timestamp(dt).normalize()})
        df = pd.DataFrame(rows, columns=["Ticker", "EarningsDate"])
        df.to_parquet(self._earnings_cache_path, index=False)

    def _load_earnings_cache(self) -> dict[str, pd.DatetimeIndex]:
        if not self._earnings_cache_path.exists():
            return {}
        try:
            df = pd.read_parquet(self._earnings_cache_path)
            out: dict[str, pd.DatetimeIndex] = {sym: pd.DatetimeIndex([]) for sym in self._config.universe}
            if len(df) > 0:
                for sym, grp in df.groupby("Ticker"):
                    out[str(sym)] = pd.DatetimeIndex(
                        sorted(pd.to_datetime(grp["EarningsDate"]).dt.normalize().unique())
                    )
            return out
        except Exception:  # noqa: BLE001
            return {}

    # -- calendar alignment & assembly --------------------------------------

    def _assemble(
        self,
        snapshot: pd.DataFrame,
        earnings_map: Mapping[str, Sequence[pd.Timestamp | str]] | None = None,
    ) -> MarketBundle:
        universe = list(self._config.universe)
        benchmark = self._config.benchmark
        all_syms = self._required_tickers()

        def _extract_field(field_name: str, symbols: list[str]) -> pd.DataFrame:
            return pd.DataFrame(
                {sym: snapshot.xs(sym, level="Ticker")[field_name] for sym in symbols}
            ).sort_index()

        closes_all = _extract_field("Close", all_syms).dropna(axis=0, how="any")
        if closes_all.empty:
            raise RuntimeError("No overlapping trading sessions across universe + benchmark.")

        common_idx = closes_all.index
        opens = _extract_field("Open", universe).reindex(common_idx)
        highs = _extract_field("High", universe).reindex(common_idx)
        lows = _extract_field("Low", universe).reindex(common_idx)
        closes = closes_all[universe].copy()
        spy_close = closes_all[benchmark].copy().rename(benchmark)

        valid_mask = (
            opens.notna().all(axis=1)
            & highs.notna().all(axis=1)
            & lows.notna().all(axis=1)
            & closes.notna().all(axis=1)
            & spy_close.notna()
        )
        opens = opens.loc[valid_mask].astype(float)
        highs = highs.loc[valid_mask].astype(float)
        lows = lows.loc[valid_mask].astype(float)
        closes = closes.loc[valid_mask].astype(float)
        spy_close = spy_close.loc[valid_mask].astype(float)

        if len(closes) < self._config.warmup_days_required:
            raise RuntimeError(
                f"Insufficient aligned sessions ({len(closes)}) < warmup ({self._config.warmup_days_required})."
            )

        norm_earnings: dict[str, pd.DatetimeIndex] = {}
        raw_map = earnings_map or {}
        for sym in universe:
            dts = raw_map.get(sym, [])
            if len(dts) > 0:
                idx = pd.to_datetime(list(dts))
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_localize(None)
                norm_earnings[sym] = pd.DatetimeIndex(sorted(idx.normalize().unique()))
            else:
                norm_earnings[sym] = pd.DatetimeIndex([])

        return MarketBundle(
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            spy_close=spy_close,
            earnings_dates=norm_earnings,
        )
