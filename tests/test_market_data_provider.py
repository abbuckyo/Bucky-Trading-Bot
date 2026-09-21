from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest

from alphashield.data.provider import MarketDataProvider


def test_provider_normalize_frame_valid():
    provider = MarketDataProvider()
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    raw_df = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "High": [105.0, 106.0, 107.0, 108.0, 109.0],
        "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "Close": [102.0, 103.0, 104.0, 105.0, 106.0],
        "Volume": [1000, 1100, 1200, 1300, 1400],
    }, index=dates)

    norm = provider.normalize_frame(raw_df, "test")
    assert norm is not None
    assert list(norm.columns) == ["open", "high", "low", "close", "volume"]
    assert len(norm) == 5


def test_provider_quote_yahoo_success():
    provider = MarketDataProvider()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "chart": {
            "result": [{
                "meta": {
                    "regularMarketPrice": 5050.0,
                    "previousClose": 5000.0,
                }
            }]
        }
    }

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    provider.get_curl_session = MagicMock(return_value=mock_session)

    quote = provider.get_quote("ES=F")
    assert quote is not None
    assert quote["price"] == 5050.0
    assert quote["prev_close"] == 5000.0
    assert quote["chg_pct"] == 1.0
    assert quote["source"] == "yahoo_v8"


def test_provider_quote_finnhub_fallback_when_yahoo_fails():
    # Yahoo fails (e.g. 429), yfinance fails -> falls back to Finnhub
    provider = MarketDataProvider(finnhub_key="test_finnhub_key")

    mock_curl = MagicMock()
    mock_curl.get.side_effect = Exception("Yahoo 429 Rate Limit")
    provider.get_curl_session = MagicMock(return_value=mock_curl)

    mock_finnhub_resp = MagicMock()
    mock_finnhub_resp.status_code = 200
    mock_finnhub_resp.json.return_value = {
        "c": 505.0,
        "pc": 500.0,
        "dp": 1.0,
    }

    with patch("yfinance.Ticker", side_effect=Exception("yfinance down")):
        with patch("requests.get", return_value=mock_finnhub_resp) as mock_req_get:
            quote = provider.get_quote("ES=F")
            assert quote is not None
            assert quote["price"] == 505.0
            assert quote["prev_close"] == 500.0
            assert quote["chg_pct"] == 1.0
            assert "finnhub" in quote["source"]
            # Verify proxy mapping was used
            assert "symbol=SPY" in mock_req_get.call_args[0][0]
