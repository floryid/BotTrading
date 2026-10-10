"""Smart Money score: whale exchange inflow + liquidations + order-book imbalance."""
import logging
import threading
import time
from typing import Optional

import requests

from . import config

log = logging.getLogger("smart_money")


def _clip(v: float) -> float:
    return max(-1.0, min(1.0, v))


class SmartMoney:
    """Each component is cached (TTL 5 min). When an API fails the last cached
    value is kept; without API keys the component stays neutral (0)."""

    def __init__(self, exchange) -> None:
        self.exchange = exchange
        self._lock = threading.Lock()
        self._cache: dict = {"whale": 0.0, "liquidation": 0.0, "liq_spike": False, "liq_bias": None,
                             "whale_ts": 0.0, "liq_ts": 0.0, "whale_ok": False, "liq_ok": False}
        self._book: dict = {}  # symbol -> (ts, imbalance)

    # --- CryptoQuant: BTC exchange inflow ----------------------------------
    def _refresh_whale(self) -> None:
        if not config.CRYPTOQUANT_KEY:
            return
        try:
            resp = requests.get(
                "https://api.cryptoquant.com/v1/btc/exchange-flows/inflow",
                params={"exchange": "all_exchange", "window": "hour", "limit": 48},
                headers={"Authorization": f"Bearer {config.CRYPTOQUANT_KEY}"}, timeout=10)
            resp.raise_for_status()
            rows = resp.json()["result"]["data"]
            values = [float(r["inflow_total"]) for r in rows if r.get("inflow_total") is not None]
            if len(values) >= 5:
                baseline = sum(values[:-1]) / len(values[:-1])
                latest = sum(values[-3:]) / 3
                # inflow above baseline = coins moving to exchanges = bearish
                score = _clip(-(latest / baseline - 1.0)) if baseline > 0 else 0.0
                with self._lock:
                    self._cache.update(whale=score, whale_ts=time.time(), whale_ok=True)
        except Exception as exc:
            log.warning("CryptoQuant failed: %s", type(exc).__name__)

    # --- Coinglass: liquidations -------------------------------------------
    def _refresh_liquidation(self) -> None:
        if not config.COINGLASS_KEY:
            return
        try:
            resp = requests.get(
                "https://open-api-v4.coinglass.com/api/futures/liquidation/aggregated-history",
                params={"exchange_list": "Bybit,Binance,OKX", "symbol": "BTC", "interval": "1h", "limit": 4},
                headers={"CG-API-KEY": config.COINGLASS_KEY, "accept": "application/json"}, timeout=10)
            resp.raise_for_status()
            rows = resp.json()["data"]
            last = rows[-1]
            long_liq = float(last.get("aggregated_long_liquidation_usd", 0))
            short_liq = float(last.get("aggregated_short_liquidation_usd", 0))
            total = long_liq + short_liq
            # dominant long liquidations = forced selling = bearish
            score = _clip((short_liq - long_liq) / total) if total > 0 else 0.0
            spike = max(long_liq, short_liq) > config.LIQ_SPIKE_USD
            # a long-liquidation flush is a bear-trap candidate, short squeeze a bull-trap candidate
            bias = None
            if spike:
                bias = "bear_trap" if long_liq > short_liq else "bull_trap"
            with self._lock:
                self._cache.update(liquidation=score, liq_spike=spike, liq_bias=bias,
                                   liq_ts=time.time(), liq_ok=True)
        except Exception as exc:
            log.warning("Coinglass failed: %s", type(exc).__name__)

    # --- Order book --------------------------------------------------------
    def orderbook_imbalance(self, symbol: str) -> float:
        cached = self._book.get(symbol)
        if cached and time.time() - cached[0] < config.SMART_MONEY_TTL:
            return cached[1]
        try:
            book = self.exchange.fetch_order_book(symbol, 10)
            bids = sum(b[1] for b in book["bids"][:10])
            asks = sum(a[1] for a in book["asks"][:10])
            total = bids + asks
            value = _clip((bids - asks) / total) if total > 0 else 0.0
            self._book[symbol] = (time.time(), value)
            return value
        except Exception as exc:
            log.warning("Orderbook %s failed: %s", symbol, type(exc).__name__)
            return cached[1] if cached else 0.0

    def refresh(self) -> None:
        now = time.time()
        if now - self._cache["whale_ts"] > config.SMART_MONEY_TTL:
            self._refresh_whale()
        if now - self._cache["liq_ts"] > config.SMART_MONEY_TTL:
            self._refresh_liquidation()

    def score(self, symbol: str, refresh: bool = True) -> dict:
        if refresh:
            self.refresh()
        book = self.orderbook_imbalance(symbol) if refresh else self._book.get(symbol, (0, 0.0))[1]
        with self._lock:
            c = dict(self._cache)
        w = config.SM_WEIGHTS
        total = c["whale"] * w["whale"] + c["liquidation"] * w["liquidation"] + book * w["orderbook"]
        return {"total": round(_clip(total), 4), "whale": round(c["whale"], 4),
                "liquidation": round(c["liquidation"], 4), "orderbook": round(book, 4),
                "liq_spike": c["liq_spike"], "liq_bias": c["liq_bias"],
                "whale_api": c["whale_ok"], "liq_api": c["liq_ok"]}

    def snapshot(self) -> dict:
        with self._lock:
            return {k: v for k, v in self._cache.items()}
