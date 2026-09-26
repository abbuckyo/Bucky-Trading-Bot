"""
alphashield.notification.gateway
================================
Deep Module: Notification & Delivery Gateway
Encapsulates multi-channel message dispatching (Discord Webhook, LINE Messaging API),
message auto-chunking, retry/backoff, and dry-run safety behind a clean, simple seam.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional

import requests

LOG = logging.getLogger("quantbot.notification")


class NotificationGateway:
    """
    Deep Module for dispatching trading alerts and reports.

    Public Interface:
        - dispatch(title: str, message: str, *, notify_line: bool = True, notify_discord: bool = True) -> Dict[str, bool]
        - send_discord(message: str) -> bool
        - send_line(message: str) -> bool
    """

    def __init__(
        self,
        discord_webhook_url: str = "",
        line_access_token: str = "",
        line_user_id: str = "",
        discord_chunk_limit: int = 1900,
        line_chunk_limit: int = 4900,
        http_timeout: float = 25.0,
        line_timeout: float = 8.0,
        dry_run: bool = False,
    ):
        self.discord_webhook_url = discord_webhook_url.strip() if discord_webhook_url else ""
        self.line_access_token = line_access_token.strip() if line_access_token else ""
        self.line_user_id = line_user_id.strip() if line_user_id else ""
        self.discord_chunk_limit = discord_chunk_limit
        self.line_chunk_limit = line_chunk_limit
        self.http_timeout = http_timeout
        self.line_timeout = line_timeout
        self.dry_run = dry_run
        self.line_broadcast_endpoint = "https://api.line.me/v2/bot/message/broadcast"
        self.line_push_endpoint = self.line_broadcast_endpoint  # backward compatibility alias

    # --------------------------------------------------------------------------
    # Auto-Chunking (Hidden Implementation)
    # --------------------------------------------------------------------------

    @staticmethod
    def chunk_text(text: str, limit: int) -> List[str]:
        """
        Split a long string cleanly on newline boundaries without exceeding limit.
        """
        if not text:
            return []

        chunks: List[str] = []
        buf = ""
        for line in text.split("\n"):
            while len(line) > limit:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.append(line[:limit])
                line = line[limit:]
            if len(buf) + len(line) + 1 > limit:
                chunks.append(buf)
                buf = line
            else:
                buf = f"{buf}\n{line}" if buf else line
        if buf.strip():
            chunks.append(buf)
        return [c for c in chunks if c.strip()]

    # --------------------------------------------------------------------------
    # Discord Delivery Adapter
    # --------------------------------------------------------------------------

    def send_discord(self, message: str) -> bool:
        """
        Send formatted message to Discord webhook with chunking and 429 backoff.
        """
        if self.dry_run:
            LOG.info("[DRY-RUN] Discord message suppressed (%d chars)", len(message))
            return True

        if not self.discord_webhook_url:
            LOG.debug("Discord webhook URL not configured, skipping delivery.")
            return False

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        chunks = self.chunk_text(message, self.discord_chunk_limit)
        ok = True
        for chunk in chunks:
            for attempt in range(3):
                try:
                    r = requests.post(
                        self.discord_webhook_url,
                        json={"content": chunk},
                        headers=headers,
                        timeout=self.http_timeout,
                    )
                    if r.status_code == 429:
                        wait = float(r.json().get("retry_after", 2))
                        time.sleep(wait + 0.5)
                        continue
                    r.raise_for_status()
                    break
                except Exception as exc:
                    LOG.warning("Discord delivery failed (attempt %d/3): %s", attempt + 1, exc)
                    time.sleep(1.5 * (attempt + 1))
            else:
                ok = False
            time.sleep(0.5)
        return ok

    # --------------------------------------------------------------------------
    # LINE Messaging API Delivery Adapter (Broadcast)
    # --------------------------------------------------------------------------

    def send_line(self, message: str) -> bool:
        """
        Send formatted message to LINE broadcast API with auto-chunking (max 5 bubbles per payload).
        Broadcast delivers messages to all user friends of the official account.
        """
        # Check global environment toggle if configured
        enable_line_env = os.getenv("ENABLE_LINE")
        if enable_line_env is not None and enable_line_env.strip().lower() in ("false", "0", "no"):
            LOG.info("MOCK: ข้ามการส่ง LINE ตามการตั้งค่า (ENABLE_LINE=false)")
            return True

        if self.dry_run:
            LOG.info("[DRY-RUN] LINE message suppressed (%d chars)", len(message))
            return True

        if not self.line_access_token:
            LOG.warning("ข้ามการส่ง LINE: ไม่พบการตั้งค่า line_access_token")
            return False

        chunks = self.chunk_text(message, self.line_chunk_limit)
        if not chunks:
            LOG.warning("LINE: ข้อความว่างเปล่า ไม่สามารถส่งได้")
            return False

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.line_access_token}",
        }

        ok = True
        for start in range(0, len(chunks), 5):
            batch = chunks[start:start + 5]
            payload = {
                "messages": [{"type": "text", "text": c} for c in batch],
            }
            for attempt in range(2):
                try:
                    r = requests.post(
                        self.line_broadcast_endpoint,
                        headers=headers,
                        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        timeout=self.line_timeout,
                    )
                    if r.status_code == 429:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    LOG.info("LINE broadcast notification sent successfully (%d message chunks)", len(batch))
                    break
                except Exception as e:
                    LOG.error(
                        "LINE broadcast failed (attempt %d/2, timeout=%.1fs): %s",
                        attempt + 1,
                        self.line_timeout,
                        e,
                    )
                    time.sleep(1.0 * (attempt + 1))
            else:
                ok = False
            time.sleep(0.3)
        return ok

    # --------------------------------------------------------------------------
    # Unified Public Seam
    # --------------------------------------------------------------------------

    def dispatch(
        self,
        title: str,
        message: str,
        *,
        notify_line: bool = True,
        notify_discord: bool = True,
    ) -> Dict[str, bool]:
        """
        Unified dispatch seam for broadcasts and alerts.
        """
        full_content = f"{title}\n\n{message}" if title else message
        results = {"discord": False, "line": False}

        if notify_discord:
            try:
                results["discord"] = self.send_discord(full_content)
            except Exception as exc:
                LOG.error("NotificationGateway: Discord dispatch error: %s", exc)

        if notify_line:
            try:
                results["line"] = self.send_line(full_content)
            except Exception as exc:
                LOG.error("NotificationGateway: LINE dispatch error: %s", exc)

        return results
