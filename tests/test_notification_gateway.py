"""
Unit tests for alphashield.notification.gateway (NotificationGateway Deep Module)
================================================================================
Verifies:
  1. Auto-chunking text along line boundaries
  2. Discord delivery adapter with headers and post payload
  3. LINE delivery adapter with 5-message batching
  4. Dry-run mode suppressing HTTP requests
  5. Unified dispatch seam
"""
import json
from unittest.mock import patch, MagicMock
import pytest

from alphashield.notification.gateway import NotificationGateway
from bot import _chunk_text


def test_chunk_text_under_limit():
    text = "Line 1\nLine 2\nLine 3"
    chunks = NotificationGateway.chunk_text(text, 100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_over_limit():
    lines = [f"Item {i:03d} - some long description text" for i in range(50)]
    long_text = "\n".join(lines)
    chunks = NotificationGateway.chunk_text(long_text, 200)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 200


def test_chunk_text_line_exceeds_limit():
    # Single line/word exceeding the chunk limit
    long_line = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    limit = 10
    chunks = NotificationGateway.chunk_text(long_line, limit)
    assert chunks == ["ABCDEFGHIJ", "KLMNOPQRST", "UVWXYZ"]
    for c in chunks:
        assert len(c) <= limit


def test_chunk_text_empty_and_whitespace():
    assert NotificationGateway.chunk_text("", 100) == []
    assert NotificationGateway.chunk_text("   \n \n \n", 100) == []


def test_chunk_text_exact_limit_boundary():
    line1 = "A" * 10
    line2 = "B" * 10
    text = f"{line1}\n{line2}"
    chunks = NotificationGateway.chunk_text(text, 10)
    assert len(chunks) == 2
    assert chunks[0] == line1
    assert chunks[1] == line2


def test_chunk_text_mixed_long_and_short_lines():
    text = "Short line 1\n" + ("X" * 25) + "\nShort line 2"
    limit = 10
    chunks = NotificationGateway.chunk_text(text, limit)
    assert len(chunks) >= 4
    for c in chunks:
        assert len(c) <= limit
    joined = "".join(chunks)
    assert "Short line 1" in joined
    assert "Short line 2" in joined
    assert "X" * 25 in joined


def test_bot_chunk_text_wrapper():
    # Verify bot._chunk_text delegates to NotificationGateway.chunk_text
    long_line = "1234567890ABCDEFGHIJ"
    chunks = _chunk_text(long_line, 10)
    assert chunks == ["1234567890", "ABCDEFGHIJ"]


def test_send_discord_success():
    gw = NotificationGateway(discord_webhook_url="https://discord.mock/webhook")
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        ok = gw.send_discord("Hello Discord")
        assert ok is True
        mock_post.assert_called_once()
        assert mock_post.call_args[1]["json"] == {"content": "Hello Discord"}


def test_send_line_batching():
    # Long text that splits into 7 chunks -> should produce 2 broadcast requests (5 + 2)
    gw = NotificationGateway(
        line_access_token="MOCK_TOKEN",
        line_user_id="U1234567890",
        line_chunk_limit=20,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    long_msg = "\n".join([f"Line number {i}" for i in range(7)])
    with patch("requests.post", return_value=mock_resp) as mock_post:
        ok = gw.send_line(long_msg)
        assert ok is True
        assert mock_post.call_count == 2
        # Verify broadcast endpoint is targeted
        for call in mock_post.call_args_list:
            endpoint = call[0][0]
            assert endpoint == "https://api.line.me/v2/bot/message/broadcast"
            # Verify payload does not contain "to" key
            sent_data = json.loads(call[1]["data"].decode("utf-8"))
            assert "to" not in sent_data
            assert "messages" in sent_data


def test_dry_run_mode_suppresses_http():
    gw = NotificationGateway(
        discord_webhook_url="https://discord.mock",
        line_access_token="MOCK_TOKEN",
        line_user_id="U123",
        dry_run=True,
    )

    with patch("requests.post") as mock_post:
        res = gw.dispatch(title="Test", message="Body", notify_line=True, notify_discord=True)
        assert res["discord"] is True
        assert res["line"] is True
        mock_post.assert_not_called()


def test_dispatch_unified_seam():
    gw = NotificationGateway(
        discord_webhook_url="https://discord.mock",
        line_access_token="MOCK_TOKEN",
        line_user_id="U123",
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp):
        res = gw.dispatch("Title", "Message")
        assert res["discord"] is True
        assert res["line"] is True
