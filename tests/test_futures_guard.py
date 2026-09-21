from unittest.mock import patch, MagicMock
from bot import evaluate_us_futures_guard, CFG

def test_futures_guard_normal_via_curl():
    # Test that evaluate_us_futures_guard returns ok status when curl_cffi succeeds
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "chart": {
            "result": [{
                "meta": {
                    "regularMarketPrice": 5000.0,
                    "previousClose": 5000.0,
                }
            }]
        }
    }
    with patch("bot._get_curl_session") as mock_session_getter:
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_getter.return_value = mock_session

        triggered, data, desc, status = evaluate_us_futures_guard()
        assert status == "ok"
        assert triggered is False
        assert CFG.FUTURES_ES_TICKER in data
        assert CFG.FUTURES_NQ_TICKER in data
        assert data[CFG.FUTURES_ES_TICKER] == 0.0
        assert data[CFG.FUTURES_NQ_TICKER] == 0.0


def test_futures_guard_circuit_breaker_triggers():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    # ES drops -1.5% (Limit: -1.2%)
    mock_resp.json.return_value = {
        "chart": {
            "result": [{
                "meta": {
                    "regularMarketPrice": 4925.0,
                    "previousClose": 5000.0,
                }
            }]
        }
    }
    with patch("bot._get_curl_session") as mock_session_getter:
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_getter.return_value = mock_session

        triggered, data, desc, status = evaluate_us_futures_guard()
        assert status == "ok"
        assert triggered is True
        assert "CIRCUIT BREAKER TRIGGERED" in desc
