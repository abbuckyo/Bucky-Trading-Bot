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
from unittest.mock import patch, MagicMock
import pytest

from alphashield.notification.gateway import NotificationGateway


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
    # Long text that splits into 7 chunks -> should produce 2 push requests (5 + 2)
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
