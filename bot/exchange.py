"""Thin ccxt wrapper for Bybit linear perpetuals."""
import logging
import time
from typing import Optional

import ccxt
import pandas as pd

from . import config

log = logging.getLogger("exchange")


class Exchange:
    def __init__(self) -> None:
        params = {"enableRateLimit": True, "options": {"defaultType": "swap"}}
        if config.BYBIT_DOMAIN != "bybit.com":
            params["hostname"] = config.BYBIT_DOMAIN
        if config.API_KEY and config.SECRET:
            params.update(apiKey=config.API_KEY, secret=config.SECRET)
        self.ex = ccxt.bybit(params)
        self.markets_loaded = False

    # --- market data ---------------------------------------------------------
    def load_markets(self) -> None:
        try:
            self.ex.load_markets()
            self.markets_loaded = True
        except Exception as exc:
            log.warning("load_markets failed: %s", type(exc).__name__)

    def fetch_ohlcv(self, symbol: str, timeframe: str = config.TIMEFRAME, limit: int = 200,
                    drop_open: bool = True) -> Optional[pd.DataFrame]:
        raw = None
        for attempt in range(3):
            try:
                raw = self.ex.fetch_ohlcv(config.swap_symbol(symbol), timeframe, limit=limit)
                break
            except ccxt.NetworkError as exc:
                log.warning("fetch_ohlcv %s attempt %d failed: %s", symbol, attempt + 1, type(exc).__name__)
                time.sleep(2 * (attempt + 1))
            except Exception as exc:
                log.warning("fetch_ohlcv %s failed: %s", symbol, type(exc).__name__)
                return None
        if raw is None:
            return None
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if drop_open and len(df) > 1:
            df = df.iloc[:-1]  # last candle is still forming
        return df.reset_index(drop=True)

    def fetch_history(self, symbol: str, timeframe: str, total: int) -> Optional[pd.DataFrame]:
        """Paginated history (closed candles only), oldest first."""
        step = self.ex.parse_timeframe(timeframe) * 1000
        since = int(time.time() * 1000) - total * step
        rows: list = []
        try:
            while len(rows) < total + 2:
                raw = self.ex.fetch_ohlcv(config.swap_symbol(symbol), timeframe, since=since, limit=1000)
                if not raw:
                    break
                rows += raw
                nxt = raw[-1][0] + step
                if nxt <= since or len(raw) < 1000:
                    break
                since = nxt
        except Exception as exc:
            log.warning("fetch_history %s failed: %s", symbol, type(exc).__name__)
            if not rows:
                return None
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        return df.iloc[:-1].tail(total).reset_index(drop=True) if len(df) > 1 else None

    def fetch_order_book(self, symbol: str, limit: int = 10) -> dict:
        return self.ex.fetch_order_book(config.swap_symbol(symbol), limit)

    # --- precision -------------------------------------------------------------
    def round_qty(self, symbol: str, qty: float) -> float:
        """Round down to the exchange step; returns 0 if below the minimum amount."""
        sym = config.swap_symbol(symbol)
        if not self.markets_loaded:
            return float(int(qty * 1000)) / 1000
        try:
            market = self.ex.market(sym)
            qty = float(self.ex.amount_to_precision(sym, qty))
            minimum = (market.get("limits", {}).get("amount", {}) or {}).get("min")
            if minimum and qty < minimum:
                return 0.0
            return qty
        except Exception as exc:
            log.warning("round_qty %s failed: %s", symbol, exc)
            return 0.0

    # --- account (live only) -----------------------------------------------
    def setup_futures(self, symbols) -> None:
        """Cross margin, one-way position mode and configured leverage."""
        for sym in symbols:
            swap = config.swap_symbol(sym)
            for label, fn in (("position mode", lambda: self.ex.set_position_mode(False, swap)),
                              ("margin mode", lambda: self.ex.set_margin_mode("cross", swap)),
                              ("leverage", lambda: self.ex.set_leverage(config.LEVERAGE, swap))):
                try:
                    fn()
                except ccxt.BaseError as exc:
                    # Bybit answers with an error when the value is already set
                    log.info("setup %s %s: %s", sym, label, str(exc)[:120])
            log.info("Futures setup done for %s", sym)

    def fetch_equity(self) -> Optional[float]:
        try:
            bal = self.ex.fetch_balance()
            return float(bal["total"].get("USDT", 0.0))
        except Exception as exc:
            log.warning("fetch_balance failed: %s", type(exc).__name__)
            return None

    def fetch_position(self, symbol: str) -> Optional[dict]:
        """Open exchange position for the symbol, or None if flat. Raises on API errors."""
        positions = self.ex.fetch_positions([config.swap_symbol(symbol)])
        for p in positions:
            if float(p.get("contracts") or 0) > 0:
                return p
        return None

    def open_market(self, symbol: str, side: str, qty: float, sl: float, tp: float) -> dict:
        """Market entry with position-level SL/TP (market-triggered) attached on Bybit."""
        order_side = "buy" if side == "LONG" else "sell"
        return self.ex.create_order(config.swap_symbol(symbol), "market", order_side, qty, None,
                                    {"stopLoss": sl, "takeProfit": tp})

    def set_stop_loss(self, symbol: str, price: float) -> None:
        """Move the position-level stop loss (market-triggered). Raises on API failure."""
        swap = config.swap_symbol(symbol)
        self.ex.privatePostV5PositionTradingStop({
            "category": "linear", "symbol": self.ex.market(swap)["id"], "tpslMode": "Full", "positionIdx": 0,
            "stopLoss": self.ex.price_to_precision(swap, price), "slOrderType": "Market"})

    def close_market(self, symbol: str, side: str, qty: float) -> dict:
        order_side = "sell" if side == "LONG" else "buy"
        return self.ex.create_order(config.swap_symbol(symbol), "market", order_side, qty, None,
                                    {"reduceOnly": True})
