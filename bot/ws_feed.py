"""Bybit v5 public WebSocket price feed with backoff reconnect."""
import json
import logging
import threading
import time
from typing import Callable, Dict, Optional

import websocket

from . import config

log = logging.getLogger("ws")


class PriceFeed:
    def __init__(self, symbols, on_giveup: Optional[Callable[[], None]] = None) -> None:
        self.symbols = symbols
        self._by_ws = {config.ws_symbol(s): s for s in symbols}
        self.price_stream: Dict[str, float] = {}
        self.price_ts: Dict[str, float] = {}
        self.ws_connected = False
        self.gave_up = False
        self.retries = 0
        self.reconnects = 0
        self._ws_lock = threading.Lock()
        self._stop = threading.Event()
        self._app: Optional[websocket.WebSocketApp] = None
        self._on_giveup = on_giveup

    # --- public --------------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="ws-feed").start()

    def stop(self) -> None:
        self._stop.set()
        if self._app is not None:
            self._app.close()

    def get_price(self, symbol: str, max_age: Optional[float] = None) -> Optional[float]:
        with self._ws_lock:
            price = self.price_stream.get(symbol)
            ts = self.price_ts.get(symbol, 0)
        if price is None:
            return None
        if max_age is not None and time.time() - ts > max_age:
            return None
        return price

    def snapshot(self) -> dict:
        with self._ws_lock:
            return {"connected": self.ws_connected, "retries": self.retries, "reconnects": self.reconnects,
                    "gave_up": self.gave_up,
                    "prices": {s: {"price": p, "age": round(time.time() - self.price_ts.get(s, 0), 1)}
                               for s, p in self.price_stream.items()}}

    # --- internals -----------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            self._app = websocket.WebSocketApp(
                config.WS_URL, on_open=self._on_open, on_message=self._on_message,
                on_error=self._on_error, on_close=self._on_close)
            self._app.run_forever(ping_interval=20, ping_timeout=10)
            with self._ws_lock:
                self.ws_connected = False
            if self._stop.is_set():
                return
            self.retries += 1
            if self.retries > config.WS_MAX_RETRY:
                self.gave_up = True
                log.error("WebSocket: giving up after %d retries", config.WS_MAX_RETRY)
                if self._on_giveup:
                    self._on_giveup()
                return
            delay = min(5 * self.retries, 60)
            log.warning("WebSocket closed, reconnect #%d in %ds", self.retries, delay)
            self._stop.wait(delay)

    def _on_open(self, ws) -> None:
        self.retries = 0
        with self._ws_lock:
            self.ws_connected = True
            self.reconnects += 1
        args = [f"tickers.{config.ws_symbol(s)}" for s in self.symbols]
        ws.send(json.dumps({"op": "subscribe", "args": args}))
        log.info("WebSocket connected, subscribed to %d tickers", len(args))

    def _on_message(self, ws, message: str) -> None:
        try:
            msg = json.loads(message)
        except ValueError:
            return
        topic = msg.get("topic", "")
        if not topic.startswith("tickers."):
            return
        data = msg.get("data") or {}
        sym = self._by_ws.get(data.get("symbol", topic.split(".", 1)[1]))
        last = data.get("lastPrice")
        if sym and last:  # delta messages may omit lastPrice
            with self._ws_lock:
                self.price_stream[sym] = float(last)
                self.price_ts[sym] = time.time()

    def _on_error(self, ws, error) -> None:
        log.warning("WebSocket error: %s", error)

    def _on_close(self, ws, code, reason) -> None:
        with self._ws_lock:
            self.ws_connected = False
