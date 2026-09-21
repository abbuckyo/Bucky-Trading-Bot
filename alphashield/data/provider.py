"""
alphashield.data.provider
========================
Deep Module for market data retrieval.
Hides connection session lifecycles, curl_cffi impersonation, multi-source
waterfall fallbacks, data frame normalization, and pre-market quote recovery.
"""
from __future__ import annotations

import io
import logging
import math
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

LOG = logging.getLogger("quantbot.data")

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def is_nan(val: Any) -> bool:
    """Check if a numeric value is NaN, None, or infinite."""
    if val is None:
        return True
    try:
        f = float(val)
        return math.isnan(f) or math.isinf(f)
    except (ValueError, TypeError):
        return True


class MarketDataProvider:
    """
    Deep Module encapsulating all market data ingestion logic.

    Public Interface:
        - get_series(ticker, stooq_symbol=None, min_rows=240, stale_days_max=10) -> Tuple[Optional[pd.DataFrame], str]
        - get_quote(ticker) -> Optional[Dict[str, float]]
    """

    def __init__(
        self,
        finnhub_key: str = "",
        http_timeout: int = 25,
        history_period: str = "3y",
        session_factory: Optional[Callable[[], Any]] = None,
    ):
        self.finnhub_key = finnhub_key.strip() if finnhub_key else ""
        self.http_timeout = http_timeout
        self.history_period = history_period
        self._custom_session_factory = session_factory
        self._curl_session = None

    def get_curl_session(self):
        """Create or return a browser-fingerprinted session to bypass Cloudflare/Yahoo 429."""
        if self._custom_session_factory is not None:
            return self._custom_session_factory()

        if self._curl_session is not None:
            return self._curl_session

        try:
            from curl_cffi import requests as cureq
            self._curl_session = cureq.Session(impersonate="chrome124")
        except Exception:
            sess = requests.Session()
            sess.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            })
            self._curl_session = sess

        return self._curl_session

    # --------------------------------------------------------------------------
    # Normalization & Validation (Hidden implementation)
    # --------------------------------------------------------------------------

    def normalize_frame(self, df: pd.DataFrame, source: str) -> Optional[pd.DataFrame]:
        """Normalize any incoming DataFrame to standard schema [open, high, low, close, volume]."""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None

        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            flat = []
            for tup in df.columns:
                parts = [str(p) for p in tup if p is not None and str(p) != ""]
                chosen = parts[0] if parts else ""
                for p in parts:
                    if p.lower().replace(" ", "_") in REQUIRED_COLS + ["adj_close"]:
                        chosen = p
                        break
                flat.append(chosen)
            df.columns = flat

        df.columns = [
            str(c).strip().lower().replace(" ", "_").replace("-", "_")
            for c in df.columns
        ]

        if "close" not in df.columns and "adj_close" in df.columns:
            df["close"] = df["adj_close"]

        if not isinstance(df.index, pd.DatetimeIndex):
            for cand in ("date", "datetime", "index"):
                if cand in df.columns:
                    df.index = pd.to_datetime(df[cand], errors="coerce")
                    df = df.drop(columns=[cand])
                    break
            else:
                df.index = pd.to_datetime(df.index, errors="coerce")

        df.index = pd.to_datetime(df.index, errors="coerce")
        try:
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert(None)
        except (TypeError, AttributeError):
            try:
                df.index = df.index.tz_localize(None)
            except Exception:
                pass

        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep="last")].sort_index()

        if "close" not in df.columns:
            LOG.warning("[%s] normalization failed: no close column", source)
            return None
        for col in ("open", "high", "low"):
            if col not in df.columns:
                df[col] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = 0.0

        df = df[REQUIRED_COLS]
        for col in REQUIRED_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]

        return df if not df.empty else None

    def is_fresh(self, df: pd.DataFrame, stale_days_max: int = 10) -> bool:
        """Verify data freshness within allowed days threshold."""
        last = df.index[-1].to_pydatetime()
        age = (datetime.utcnow() - last).days
        if age > stale_days_max:
            LOG.warning("Stale data: last bar %s (%d days old)", last.date(), age)
            return False
        return True

    def validate(
        self,
        df: Optional[pd.DataFrame],
        source: str,
        ticker: str,
        min_rows: int = 240,
        stale_days_max: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Validate row count and freshness."""
        if df is None:
            return None
        if len(df) < min_rows:
            LOG.warning("[%s] %s rejected: %d rows < %d", source, ticker, len(df), min_rows)
            return None
        if not self.is_fresh(df, stale_days_max):
            return None
        LOG.info("[%s] %s OK -> %d rows, last=%s", source, ticker, len(df), df.index[-1].date())
        return df

    # --------------------------------------------------------------------------
    # Individual Historical Fetchers (Hidden implementation)
    # --------------------------------------------------------------------------

    def fetch_yahoo_chart_api(
        self,
        ticker: str,
        min_rows: int = 240,
        stale_days_max: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Primary: Direct v8 Chart API using curl_cffi Chrome impersonation (no crumbs/cookies required)."""
        session = self.get_curl_session()
        endpoints = [
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={self.history_period}&interval=1d",
            f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range={self.history_period}&interval=1d",
        ]

        for url in endpoints:
            try:
                resp = session.get(url, timeout=self.http_timeout)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                result = data.get("chart", {}).get("result", [])
                if not result:
                    continue
                res = result[0]
                timestamps = res.get("timestamp", [])
                quote = res.get("indicators", {}).get("quote", [{}])[0]
                adjclose = res.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", quote.get("close", []))

                if not timestamps or not quote.get("close"):
                    continue

                df = pd.DataFrame({
                    "open": quote.get("open", []),
                    "high": quote.get("high", []),
                    "low": quote.get("low", []),
                    "close": adjclose if adjclose else quote.get("close", []),
                    "volume": quote.get("volume", [0] * len(timestamps)),
                }, index=pd.to_datetime(timestamps, unit="s"))

                norm = self.normalize_frame(df, "yahoo_chart_api")
                valid = self.validate(norm, "yahoo_chart_api", ticker, min_rows=min_rows, stale_days_max=stale_days_max)
                if valid is not None:
                    return valid
            except Exception as exc:
                LOG.debug("Yahoo v8 failed for %s on %s: %s", ticker, url, exc)

        return None

    def fetch_yfinance_lib(
        self,
        ticker: str,
        min_rows: int = 240,
        stale_days_max: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Secondary: yfinance standard download."""
        try:
            import yfinance as yf
            raw = yf.download(
                tickers=ticker,
                period=self.history_period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
            )
            norm = self.normalize_frame(raw, "yfinance_lib")
            return self.validate(norm, "yfinance_lib", ticker, min_rows=min_rows, stale_days_max=stale_days_max)
        except Exception:
            return None

    def fetch_stooq(
        self,
        symbol: str,
        min_rows: int = 240,
        stale_days_max: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Tertiary: Stooq CSV via curl_cffi with anti-HTML validation."""
        session = self.get_curl_session()
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
        try:
            resp = session.get(url, timeout=self.http_timeout)
            if resp.status_code != 200:
                return None
            text = resp.text.strip()
            if not text or text.startswith(("<", "<!DOCTYPE", "<html")) or "No data" in text[:200]:
                LOG.warning("[stooq] blocked or empty response for %s", symbol)
                return None
            raw = pd.read_csv(io.StringIO(text))
            norm = self.normalize_frame(raw, "stooq")
            return self.validate(norm, "stooq", symbol, min_rows=min_rows, stale_days_max=stale_days_max)
        except Exception as exc:
            LOG.warning("[stooq] fetch failed for %s: %s", symbol, exc)
            return None

    def fetch_finnhub_candles(
        self,
        ticker: str,
        min_rows: int = 240,
        stale_days_max: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Quaternary: Finnhub daily candles for US ETFs."""
        if not self.finnhub_key:
            return None

        clean_sym = ticker.split(".")[0].upper()
        now_ts = int(time.time())
        start_ts = now_ts - (3 * 365 * 86400)
        url = f"https://finnhub.io/api/v1/stock/candle?symbol={clean_sym}&resolution=D&from={start_ts}&to={now_ts}&token={self.finnhub_key}"

        try:
            resp = requests.get(url, timeout=self.http_timeout)
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("s") != "ok" or not data.get("t"):
                return None

            df = pd.DataFrame({
                "open": data["o"],
                "high": data["h"],
                "low": data["l"],
                "close": data["c"],
                "volume": data.get("v", [0] * len(data["t"])),
            }, index=pd.to_datetime(data["t"], unit="s"))

            norm = self.normalize_frame(df, "finnhub_candles")
            return self.validate(norm, "finnhub_candles", ticker, min_rows=min_rows, stale_days_max=stale_days_max)
        except Exception as exc:
            LOG.warning("[finnhub] candles failed for %s: %s", clean_sym, exc)
            return None

    # --------------------------------------------------------------------------
    # Unified Public Interfaces
    # --------------------------------------------------------------------------

    def get_series(
        self,
        ticker: str,
        stooq_symbol: Optional[str] = None,
        min_rows: int = 240,
        stale_days_max: int = 10,
    ) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Primary entry point for historical price series.
        Cascades through 4 resilient sources until one succeeds.
        Returns: (df, source_name)
        """
        # 1. Direct Yahoo v8 Chart API
        df = self.fetch_yahoo_chart_api(ticker, min_rows=min_rows, stale_days_max=stale_days_max)
        if df is not None:
            return df, "yahoo_v8"

        # 2. Standard yfinance
        df = self.fetch_yfinance_lib(ticker, min_rows=min_rows, stale_days_max=stale_days_max)
        if df is not None:
            return df, "yfinance"

        # 3. Stooq
        if stooq_symbol:
            LOG.info("Falling back to Stooq for %s (%s)", ticker, stooq_symbol)
            df = self.fetch_stooq(stooq_symbol, min_rows=min_rows, stale_days_max=stale_days_max)
            if df is not None:
                return df, "stooq"

        # 4. Finnhub Candles
        LOG.info("Falling back to Finnhub Candles for %s", ticker)
        df = self.fetch_finnhub_candles(ticker, min_rows=min_rows, stale_days_max=stale_days_max)
        if df is not None:
            return df, "finnhub"

        LOG.error("ALL 4 DATA SOURCES FAILED for %s", ticker)
        return None, "none"

    def get_quote(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Fetch real-time or pre-market quote for a ticker (e.g. ES=F, NQ=F, SPY, QQQ).
        Returns dict with keys: {'price': float, 'prev_close': float, 'chg_pct': float}
        Fallback cascade:
          1. Direct Yahoo Chart API (curl_cffi Chrome impersonation)
          2. yfinance FastInfo / History
          3. Finnhub /quote API (resolves Yahoo 429 rate limit)
        """
        # 1. Direct Yahoo Chart API via curl_cffi
        try:
            session = self.get_curl_session()
            endpoints = [
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2d&interval=1d",
                f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=2d&interval=1d",
            ]
            for url in endpoints:
                try:
                    resp = session.get(url, timeout=self.http_timeout)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    res = data.get("chart", {}).get("result", [])
                    if not res:
                        continue
                    meta = res[0].get("meta", {})
                    last_p = meta.get("regularMarketPrice")
                    prev_c = meta.get("chartPreviousClose") or meta.get("previousClose")

                    if last_p is not None and prev_c is not None and prev_c > 0:
                        pct = ((last_p / prev_c) - 1.0) * 100.0
                        if not is_nan(pct):
                            return {
                                "price": float(last_p),
                                "prev_close": float(prev_c),
                                "chg_pct": round(float(pct), 2),
                                "source": "yahoo_v8",
                            }
                except Exception:
                    continue
        except Exception as exc:
            LOG.warning("Yahoo curl_cffi quote failed for %s: %s", ticker, exc)

        # 2. yfinance fallback
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = getattr(t, "fast_info", None)
            last_p = getattr(info, "last_price", None)
            prev_c = getattr(info, "previous_close", None)
            if last_p is None or prev_c is None or prev_c <= 0:
                h = t.history(period="2d")
                if len(h) >= 2:
                    prev_c = float(h["Close"].iloc[-2])
                    last_p = float(h["Close"].iloc[-1])
                elif len(h) == 1:
                    prev_c = float(h["Open"].iloc[0])
                    last_p = float(h["Close"].iloc[-1])

            if last_p is not None and prev_c is not None and prev_c > 0:
                pct = ((last_p / prev_c) - 1.0) * 100.0
                if not is_nan(pct):
                    return {
                        "price": float(last_p),
                        "prev_close": float(prev_c),
                        "chg_pct": round(float(pct), 2),
                        "source": "yfinance",
                    }
        except Exception as exc:
            LOG.warning("yfinance quote fallback failed for %s: %s", ticker, exc)

        # 3. Finnhub API Fallback (Bypasses Yahoo 429 entirely)
        if self.finnhub_key:
            # Map futures to proxy ETF if needed, or query direct symbol
            finnhub_symbol_map = {
                "ES=F": "SPY",
                "NQ=F": "QQQ",
            }
            clean_sym = finnhub_symbol_map.get(ticker, ticker.split(".")[0].upper())
            url = f"https://finnhub.io/api/v1/quote?symbol={clean_sym}&token={self.finnhub_key}"
            try:
                resp = requests.get(url, timeout=self.http_timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    curr = data.get("c")   # Current price
                    prev = data.get("pc")  # Previous close
                    dp = data.get("dp")    # Percent change

                    if curr and prev and prev > 0:
                        pct = dp if dp is not None else (((curr / prev) - 1.0) * 100.0)
                        if not is_nan(pct):
                            LOG.info("Finnhub quote fallback succeeded for %s (via %s): %+.2f%%", ticker, clean_sym, pct)
                            return {
                                "price": float(curr),
                                "prev_close": float(prev),
                                "chg_pct": round(float(pct), 2),
                                "source": f"finnhub_{clean_sym}",
                            }
            except Exception as exc:
                LOG.warning("Finnhub quote fallback failed for %s: %s", clean_sym, exc)

        return None
