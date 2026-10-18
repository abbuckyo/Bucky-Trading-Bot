# -*- coding: utf-8 -*-
"""Module A — Data ingestion & caching.

Sequential (rate-limit friendly) fetching via ``yfinance`` with retries,
robust OHLC validation, cross-ticker calendar alignment and a Parquet
snapshot cache so repeat backtests make zero network calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import CANARY_TICKER, CORE_UNIVERSE, DataConfig

_REQUIRED_FIELDS = ("Open", "Close")


@dataclass(slots=True)
class MarketData:
    """Aligned market snapshot ready for signal & portfolio engines."""

    closes: pd.DataFrame  # risk assets + canary, single aligned calendar
    opens: pd.DataFrame  # risk assets only, same calendar as ``closes``
    risk_assets: tuple[str, ...] = CORE_UNIVERSE

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self.closes.index


def _required_tickers() -> list[str]:
    return list(CORE_UNIVERSE) + [CANARY_TICKER]


class DataEngine:
    """Load the market snapshot from cache, or fetch it sequentially."""

    def __init__(self, config: DataConfig) -> None:
        self._config = config
        self._cache_path = Path(config.cache_dir) / "market_snapshot.parquet"

    # -- public API ---------------------------------------------------------

    def load(self) -> MarketData:
        """Return validated, calendar-aligned market data (cached first)."""
        snapshot = self._load_cache()
        if snapshot is None:
            snapshot = self._fetch_all()
            self._save_cache(snapshot)
        return self._assemble(snapshot)

    # -- fetching -----------------------------------------------------------

    def _fetch_end(self) -> str:
        if self._config.backtest_end is not None:
            return str(pd.Timestamp(self._config.backtest_end))
        # yfinance ``end`` is exclusive: requesting "today" returns rows
        # through the most recent *closed* session only.
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
                return self._validate_raw(symbol, df)
            except Exception as exc:  # noqa: BLE001 - retry any provider error
                last_error = exc
                if attempt < self._config.max_retries:
                    print(f"  {symbol}: attempt {attempt} failed ({exc}); retrying...")
                    time.sleep(self._config.retry_delay_s * attempt)
        raise RuntimeError(f"Download failed for {symbol}: {last_error}") from last_error

    def _fetch_all(self) -> pd.DataFrame:
        print(f"Downloading market data sequentially from {self._config.data_start} ...")
        frames: dict[str, pd.DataFrame] = {}
        for symbol in _required_tickers():
            print(f"  fetching {symbol}")
            frames[symbol] = self._fetch_one(symbol)
        # Column layout: level 0 = Ticker, level 1 = Field (parquet-safe).
        snapshot = pd.concat(frames, names=("Ticker", "Field")).sort_index()
        print(f"  snapshot: {snapshot.index[0].date()} -> {snapshot.index[-1].date()}")
        return snapshot

    @staticmethod
    def _validate_raw(symbol: str, df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError("empty response")
        if isinstance(df.columns, pd.MultiIndex):
            field_level = None
            for level in range(df.columns.nlevels):
                values = df.columns.get_level_values(level)
                if all(f in values for f in _REQUIRED_FIELDS):
                    field_level = level
                    break
            if field_level is None:
                raise ValueError("cannot identify OHLC field level")
            df.columns = df.columns.get_level_values(field_level)
        missing = [f for f in _REQUIRED_FIELDS if f not in df.columns]
        if missing:
            raise ValueError(f"missing fields {missing}")
        out = df[list(_REQUIRED_FIELDS)].copy()
        out.index = pd.to_datetime(out.index)
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out = out.sort_index()
        if out.index.duplicated().any():
            raise ValueError("duplicate timestamps")
        if out.isna().any().any():
            raise ValueError("missing Open/Close observations")
        if (out <= 0).any().any():
            raise ValueError("non-positive prices")
        return out

    # -- cache --------------------------------------------------------------

    def _save_cache(self, snapshot: pd.DataFrame) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_parquet(self._cache_path)
        print(f"  cache written: {self._cache_path}")

    def _load_cache(self) -> pd.DataFrame | None:
        if not self._cache_path.exists():
            return None
        try:
            snapshot = pd.read_parquet(self._cache_path)
            tickers = snapshot.index.get_level_values("Ticker").unique().tolist()
            if tickers != _required_tickers():
                print("  cache miss: ticker set changed")
                return None
            self._validate_snapshot(snapshot)
            print(f"  using cached snapshot: {self._cache_path}")
            return snapshot
        except Exception as exc:  # noqa: BLE001 - corrupted cache -> refetch
            print(f"  cache unreadable ({exc}); refetching")
            return None

    @staticmethod
    def _validate_snapshot(snapshot: pd.DataFrame) -> None:
        for ticker in _required_tickers():
            clean = snapshot.xs(ticker, level="Ticker")
            if clean.isna().any().any() or (clean <= 0).any().any():
                raise ValueError(f"cache holds invalid rows for {ticker}")

    # -- alignment ----------------------------------------------------------

    def _assemble(self, snapshot: pd.DataFrame) -> MarketData:
        closes_all = pd.DataFrame(
            {t: snapshot.xs(t, level="Ticker")["Close"] for t in _required_tickers()}
        ).sort_index()
        # Alignment check: keep only sessions where ALL required tickers traded.
        closes_all = closes_all.dropna(axis=0, how="any")
        if closes_all.empty:
            raise RuntimeError("no overlapping sessions across required tickers")
        if len(closes_all) < self._config.warmup_days_required:
            raise RuntimeError(f"insufficient data ({len(closes_all)} sessions)")

        bt_start = pd.Timestamp(self._config.backtest_start)
        warmup = int((closes_all.index < bt_start).sum())
        if warmup < self._config.warmup_days_required:
            raise RuntimeError(
                f"warm-up insufficient: {warmup} sessions before {bt_start.date()}, "
                f"need {self._config.warmup_days_required}. Raise data_start."
            )

        opens = pd.DataFrame(
            {t: snapshot.xs(t, level="Ticker")["Open"] for t in CORE_UNIVERSE}
        ).reindex(closes_all.index)
        if opens.isna().any().any():
            bad = opens.index[opens.isna().any(axis=1)]
            print(f"  dropping {len(bad)} session(s) with missing opens")
            closes_all = closes_all.drop(index=bad)
            opens = opens.drop(index=bad)

        gaps = closes_all.index.to_series().diff().dt.days.dropna()
        if (gaps > 10).any():
            print("  warning: calendar gap(s) > 10 days detected after alignment")

        return MarketData(
            closes=closes_all,
            opens=opens.astype(float),
            risk_assets=CORE_UNIVERSE,
        )
