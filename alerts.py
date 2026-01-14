import asyncio
import json
import smtplib
import ssl
import time
import urllib.request
from email.message import EmailMessage

from config import Config


class AlertManager:
    def __init__(self):
        self.enabled = bool(getattr(Config, "ALERT_ENABLED", False))
        provider = getattr(Config, "ALERT_PROVIDER", "telegram")
        self.provider = (provider or "").lower()
        levels = getattr(Config, "ALERT_LEVELS", ["ERROR", "WARN"]) or []
        self.levels = {str(lvl).upper() for lvl in levels}
        try:
            self.min_interval = float(getattr(Config, "ALERT_MIN_INTERVAL_SEC", 60))
        except Exception:
            self.min_interval = 60.0
        self._queue = None
        self._worker = None
        self._last_sent = 0.0

    def start(self):
        if not self.enabled:
            return
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._worker = asyncio.create_task(self._worker_loop())

    def notify(self, level: str, message: str):
        if not self.enabled:
            return
        lvl = (level or "INFO").upper()
        if lvl not in self.levels:
            return
        if self._queue is None:
            return
        try:
            self._queue.put_nowait((lvl, str(message)))
        except Exception:
            pass

    async def _worker_loop(self):
        while True:
            level, message = await self._queue.get()
            try:
                now = time.time()
                if self.min_interval > 0 and (now - self._last_sent) < self.min_interval:
                    continue
                self._last_sent = now
                await asyncio.to_thread(self._send, level, message)
            except Exception:
                continue

    def _send(self, level: str, message: str):
        text = f"[{level}] {message}"
        if self.provider == "telegram":
            self._send_telegram(text)
        elif self.provider == "discord":
            self._send_discord(text)
        elif self.provider == "email":
            self._send_email(text)
        elif self.provider == "webhook":
            self._send_webhook(text)

    def _send_telegram(self, text: str):
        token = getattr(Config, "TELEGRAM_BOT_TOKEN", "")
        chat_id = getattr(Config, "TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        self._post_json(url, payload)

    def _send_discord(self, text: str):
        url = getattr(Config, "DISCORD_WEBHOOK_URL", "")
        if not url:
            return
        payload = {"content": text}
        self._post_json(url, payload)

    def _send_webhook(self, text: str):
        url = getattr(Config, "ALERT_WEBHOOK_URL", "")
        if not url:
            return
        payload = {"content": text}
        self._post_json(url, payload)

    def _send_email(self, text: str):
        host = getattr(Config, "SMTP_HOST", "")
        to_addr = getattr(Config, "SMTP_TO", "")
        if not host or not to_addr:
            return
        try:
            port = int(getattr(Config, "SMTP_PORT", 587))
        except Exception:
            port = 587
        user = getattr(Config, "SMTP_USER", "")
        password = getattr(Config, "SMTP_PASS", "")
        from_addr = getattr(Config, "SMTP_FROM", "") or user or to_addr

        msg = EmailMessage()
        msg["Subject"] = "NEUROBOT Alert"
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg.set_content(text)

        context = ssl.create_default_context()
        try:
            with smtplib.SMTP(host, port, timeout=10) as server:
                server.starttls(context=context)
                if user and password:
                    server.login(user, password)
                server.send_message(msg)
        except Exception:
            return

    def _post_json(self, url: str, payload: dict):
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception:
            return
