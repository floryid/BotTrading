import logging
import queue
import threading
import time

import requests

from . import config

log = logging.getLogger("telegram")


class Notifier:
    """Non-blocking Telegram sender (messages go through a worker thread)."""

    def __init__(self) -> None:
        self.enabled = bool(config.TELEGRAM_TOKEN and config.CHAT_ID)
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=200)
        if self.enabled:
            threading.Thread(target=self._worker, daemon=True, name="telegram").start()
        else:
            log.info("Telegram disabled (TELEGRAM_TOKEN / CHAT_ID not set)")

    def send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait(text)
        except queue.Full:
            log.warning("Telegram queue full, message dropped")

    def flush(self, timeout: float = 5.0) -> None:
        """Wait briefly for pending messages (used at shutdown)."""
        if not self.enabled:
            return
        end = time.monotonic() + timeout
        while not self._q.empty() and time.monotonic() < end:
            time.sleep(0.1)
        time.sleep(1.0)  # let the in-flight request finish

    def _worker(self) -> None:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
        while True:
            text = self._q.get()
            try:
                resp = requests.post(url, json={"chat_id": config.CHAT_ID, "text": text[:4000]}, timeout=10)
                if resp.status_code != 200:
                    log.warning("Telegram HTTP %s", resp.status_code)
            except requests.RequestException as exc:
                # never log the exception URL: it contains the bot token
                log.warning("Telegram send failed: %s", type(exc).__name__)
