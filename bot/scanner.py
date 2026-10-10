"""Market scanner: monitors the most liquid Bybit USDT perpetuals (read-only, never trades)."""
import logging
import threading
import time
from collections import deque
from typing import Dict, Optional

import requests

from . import config
from .alerts import AlertManager
from .exchange import Exchange
from .indicators import add_indicators
from .profit_edge import scanner_row_edge

log = logging.getLogger("scanner")
SUFFIX = "/USDT:USDT"
EVENT_MIN_TURNOVER = 2e7   # only liquid coins generate automatic events
EVENT_COOLDOWN = 3 * 3600


class _NullNotifier:
    def send(self, text: str) -> None:
        pass


def _f(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify(trend_up: bool, rsi: float, adx: float) -> str:
    if rsi >= 70:
        return "OVERBOUGHT"
    if rsi <= 30:
        return "OVERSOLD"
    if adx >= 20 and trend_up and rsi >= 45:
        return "BULLISH"
    if adx >= 20 and not trend_up and rsi <= 55:
        return "BEARISH"
    return "NEUTRAL"


class MarketScanner:
    def __init__(self, notifier=None) -> None:
        self.exchange = Exchange()
        self.notifier = notifier
        self._lock = threading.Lock()
        self.rows: Dict[str, dict] = {}
        self.fng: Optional[dict] = None
        self.updated = 0.0
        self.analysis_updated = 0.0
        self.events: deque = deque(maxlen=150)
        self._event_id = 0
        self._cool: Dict[tuple, float] = {}
        self._primed = False
        self._stop = threading.Event()
        self.alerts = AlertManager(notifier or _NullNotifier(), self.add_event)

    # --- opportunity feed ----------------------------------------------------------
    def add_event(self, kind: str, symbol: str, text: str, side: Optional[str], cooldown: float = 0.0) -> None:
        now = time.time()
        key = (kind, symbol, side)
        if cooldown and now - self._cool.get(key, 0) < cooldown:
            return
        self._cool[key] = now
        with self._lock:
            self._event_id += 1
            self.events.appendleft({"id": self._event_id, "ts": now, "type": kind, "symbol": symbol,
                                    "side": side, "text": text})
        if config.SCANNER_TELEGRAM and self.notifier and kind != "ALERT":
            self.notifier.send(f"📡 {kind} {text}")

    def _detect(self, sym: str, old: dict, new: dict, row: dict) -> None:
        """Compare the previous and the fresh 1H analysis; only liquid coins generate events."""
        if row["turnover"] < EVENT_MIN_TURNOVER or old.get("signal") is None:
            return
        base = row["base"]
        if new["signal"] != old["signal"] and new["signal"] in ("BULLISH", "BEARISH"):
            side = "LONG" if new["signal"] == "BULLISH" else "SHORT"
            self.add_event("SIGNAL", sym, f"{base}: sinyal 1H berubah jadi {new['signal']} (RSI {new['rsi']}, ADX {new['adx']})",
                           side, EVENT_COOLDOWN)
        if new["vol_ratio"] and new["vol_ratio"] >= 3 and abs(new["chg1h"]) >= 1.5:
            side = "LONG" if new["chg1h"] > 0 else "SHORT"
            self.add_event("VOLUME", sym, f"{base}: volume {new['vol_ratio']}x dengan gerak {new['chg1h']:+.2f}% dalam 1 jam",
                           side, EVENT_COOLDOWN)
        pe = new.get("profit_edge")
        if pe and pe.get("hot") and pe.get("side") in ("LONG", "SHORT"):
            self.add_event("PROFIT", sym,
                           f"{base}: peluang profit AI {pe['edge']}/100 (grade {pe['grade']}) • {pe['side']} • E[R] {pe.get('expected_r', 0):+.2f}",
                           pe["side"], EVENT_COOLDOWN)
        if old.get("rsi") is not None:
            if old["rsi"] >= 30 > new["rsi"]:
                self.add_event("RSI", sym, f"{base}: RSI 1H masuk oversold ({new['rsi']})", "LONG", EVENT_COOLDOWN)
            elif old["rsi"] <= 70 < new["rsi"]:
                self.add_event("RSI", sym, f"{base}: RSI 1H masuk overbought ({new['rsi']})", "SHORT", EVENT_COOLDOWN)

    # --- public --------------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._ticker_loop, daemon=True, name="scanner-tickers").start()
        threading.Thread(target=self._analysis_loop, daemon=True, name="scanner-analysis").start()

    def stop(self) -> None:
        self._stop.set()

    def has(self, symbol: str) -> bool:
        with self._lock:
            return symbol in self.rows

    def row(self, symbol: str) -> Optional[dict]:
        with self._lock:
            r = self.rows.get(symbol)
            return dict(r) if r else None

    def snapshot(self) -> dict:
        with self._lock:
            rows = [dict(r) for r in self.rows.values()]
            hot = sorted((r for r in rows if r.get("profit_edge", {}).get("hot")),
                         key=lambda r: -(r.get("profit_edge") or {}).get("edge", 0))[:12]
            return {"rows": rows, "updated": self.updated, "analysis_updated": self.analysis_updated,
                    "fng": self.fng, "bot_symbols": config.SYMBOLS, "events": list(self.events),
                    "profit_picks": [{"symbol": r["symbol"], **r["profit_edge"]} for r in hot]}

    # --- tickers (one API call for every symbol) --------------------------------
    def _ticker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._refresh_tickers()
            except Exception as exc:
                log.warning("Scanner tickers failed: %s", type(exc).__name__)
            self._refresh_fng()
            self._stop.wait(config.SCANNER_TICKER_SECONDS)

    def _refresh_tickers(self) -> None:
        tickers = self.exchange.ex.fetch_tickers(params={"category": "linear"})
        items = [t for k, t in tickers.items() if k.endswith(SUFFIX) and t.get("last")]
        items.sort(key=lambda t: -(t.get("quoteVolume") or 0))
        wanted = {s for s in config.SYMBOLS}
        chosen = items[:config.SCANNER_TOP_N]
        chosen_syms = {t["symbol"].replace(":USDT", "") for t in chosen}
        chosen += [t for t in items[config.SCANNER_TOP_N:] if t["symbol"].replace(":USDT", "") in wanted - chosen_syms]
        fresh = {}
        for t in chosen:
            sym = t["symbol"].replace(":USDT", "")
            info = t.get("info") or {}
            high, low, last = _f(t.get("high")), _f(t.get("low")), _f(t["last"])
            fresh[sym] = {
                "symbol": sym, "base": sym.split("/")[0], "price": last,
                "chg24": _f(t.get("percentage"), 0.0), "turnover": _f(t.get("quoteVolume"), 0.0),
                "high": high, "low": low,
                "range_pos": round((last - low) / (high - low) * 100, 1) if high and low and high > low else None,
                "funding": (_f(info.get("fundingRate"), 0.0) or 0.0) * 100,
                "oi": _f(info.get("openInterestValue"), 0.0),
                "in_bot": sym in wanted}
        with self._lock:
            for sym, row in fresh.items():
                old = self.rows.get(sym, {})
                row.update({k: v for k, v in old.items() if k in ANALYSIS_KEYS})
                self.rows[sym] = row
            for sym in [s for s in self.rows if s not in fresh]:
                del self.rows[sym]
            self.updated = time.time()
            current = {s: dict(r) for s, r in self.rows.items()}
        self.alerts.check(current)
        if self._primed:
            for sym, r in current.items():
                if r["turnover"] < EVENT_MIN_TURNOVER or r["range_pos"] is None:
                    continue
                if r["range_pos"] >= 99.5:
                    self.add_event("BREAKOUT", sym, f"{r['base']}: harga menembus high 24 jam ({r['price']})", "LONG", EVENT_COOLDOWN)
                elif r["range_pos"] <= 0.5:
                    self.add_event("BREAKDOWN", sym, f"{r['base']}: harga menembus low 24 jam ({r['price']})", "SHORT", EVENT_COOLDOWN)
        self._primed = True

    def _refresh_fng(self) -> None:
        if self.fng and time.time() - self.fng["ts"] < 3600:
            return
        try:
            d = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8).json()["data"][0]
            self.fng = {"value": int(d["value"]), "label": d["value_classification"], "ts": time.time()}
        except Exception as exc:
            log.info("Fear&Greed unavailable: %s", type(exc).__name__)

    # --- technical analysis on 1h candles (rate-limited, in background) ----------
    def _analysis_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                ranked = sorted(self.rows, key=lambda s: -self.rows[s]["turnover"])[:config.SCANNER_ANALYZE_N]
                symbols = ranked + [s for s in config.SYMBOLS if s in self.rows and s not in ranked]
            for sym in symbols:
                if self._stop.is_set():
                    return
                try:
                    self._analyze(sym)
                except Exception as exc:
                    log.debug("Scanner analysis %s failed: %s", sym, type(exc).__name__)
            if symbols:
                self.analysis_updated = time.time()
            self._stop.wait(config.SCANNER_ANALYSIS_SECONDS if symbols else 5)

    def _analyze(self, sym: str) -> None:
        df = self.exchange.fetch_ohlcv(sym, "1h", 120)
        if df is None or len(df) < 60:
            return
        d = add_indicators(df)
        last = d.iloc[-1]
        trend_up = bool(last["ema20"] > last["ema50"])
        rsi, adx = float(last["rsi"]), float(last["adx"])
        vol_ma = float(last["vol_ma"]) if last["vol_ma"] == last["vol_ma"] else 0.0
        closes = d["close"].tail(24).tolist()
        low, high = min(closes), max(closes)
        spark = [round((c - low) / (high - low) * 100, 1) if high > low else 50 for c in closes]
        result = {"trend": "UP" if trend_up else "DOWN", "rsi": round(rsi, 1), "adx": round(adx, 1),
                  "vol_ratio": round(float(last["volume"]) / vol_ma, 2) if vol_ma > 0 else None,
                  "chg1h": round((float(last["close"]) / float(d["close"].iloc[-2]) - 1) * 100, 2),
                  "signal": classify(trend_up, rsi, adx), "spark": spark}
        with self._lock:
            row_funding = self.rows.get(sym, {})
        result["profit_edge"] = scanner_row_edge({**row_funding, **result})
        with self._lock:
            row = self.rows.get(sym)
            if row is None:
                return
            old = {k: row.get(k) for k in ("signal", "rsi")}
            row.update(result)
            snap = dict(row)
        self._detect(sym, old, result, snap)


ANALYSIS_KEYS = {"trend", "rsi", "adx", "vol_ratio", "chg1h", "signal", "spark", "profit_edge"}
