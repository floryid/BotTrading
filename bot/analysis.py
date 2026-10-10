"""Multi-timeframe analysis and trade-plan builder for the dashboard (decision support only)."""
import threading
import time
from typing import Dict, List, Optional

import pandas as pd

from . import config
from .exchange import Exchange
from .indicators import add_indicators
from .profit_edge import pick_best_edge

TFS = ["15m", "1h", "4h"]
CACHE_TTL = 30
SWING_K = 3


def _f(v) -> Optional[float]:
    return None if v is None or v != v else float(v)


def tf_view(df: pd.DataFrame) -> dict:
    d = add_indicators(df)
    last = d.iloc[-1]
    vol_ma = _f(last["vol_ma"]) or 0.0
    return {"close": _f(last["close"]), "trend": "UP" if last["ema20"] > last["ema50"] else "DOWN",
            "rsi": round(_f(last["rsi"]), 1), "adx": round(_f(last["adx"]), 1), "atr": _f(last["atr"]),
            "vol_ratio": round(float(last["volume"]) / vol_ma, 2) if vol_ma > 0 else None,
            "ema20": _f(last["ema20"]), "ema50": _f(last["ema50"])}


def swing_levels(df: pd.DataFrame, atr_value: float, k: int = SWING_K) -> Dict[str, List[dict]]:
    """Support/resistance from swing pivots, merged when closer than 0.35 ATR."""
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    raw = {"res": [], "sup": []}
    for i in range(k, len(df) - k):
        if high[i] == high[i - k:i + k + 1].max():
            raw["res"].append(float(high[i]))
        if low[i] == low[i - k:i + k + 1].min():
            raw["sup"].append(float(low[i]))
    out = {}
    for kind, vals in raw.items():
        merged: List[dict] = []
        for v in sorted(vals):
            if merged and abs(v - merged[-1]["price"]) <= 0.35 * atr_value:
                m = merged[-1]
                m["price"] = (m["price"] * m["touches"] + v) / (m["touches"] + 1)
                m["touches"] += 1
            else:
                merged.append({"price": v, "touches": 1})
        out[kind] = merged
    return out


def nearest(levels: Dict[str, List[dict]], price: float, n: int = 3) -> Dict[str, List[dict]]:
    above = sorted((l for l in levels["res"] + levels["sup"] if l["price"] > price), key=lambda l: l["price"])[:n]
    below = sorted((l for l in levels["res"] + levels["sup"] if l["price"] < price), key=lambda l: -l["price"])[:n]
    return {"resistance": above, "support": below}


def plan(price: float, atr_value: float) -> dict:
    sl = config.SL_ATR_MULT * atr_value
    tp1, tp2 = sl, config.TP_ATR_MULT * atr_value
    return {"atr": atr_value, "risk_dist": sl,
            "long": {"entry": price, "sl": price - sl, "tp1": price + tp1, "tp2": price + tp2},
            "short": {"entry": price, "sl": price + sl, "tp1": price - tp1, "tp2": price - tp2}}


def score_side(side: str, views: Dict[str, dict], funding: Optional[float], room_r: Optional[float]):
    """Returns (score, reasons). Positive score favours `side`."""
    sign = 1 if side == "LONG" else -1
    want = "UP" if side == "LONG" else "DOWN"
    score, reasons = 0.0, []

    def add(points, ok, text):
        nonlocal score
        score += points
        reasons.append({"ok": ok, "text": text})

    for tf, weight in (("4h", 2), ("1h", 2), ("15m", 1)):
        v = views[tf]
        aligned = v["trend"] == want
        add(weight if aligned else -weight, aligned, f"Trend {tf} {'searah' if aligned else 'berlawanan'} ({v['trend']}, EMA20/50)")
    h = views["1h"]
    rsi_h = h["rsi"] if side == "LONG" else 100 - h["rsi"]
    if 40 <= rsi_h <= 65:
        add(1, True, f"RSI 1H {h['rsi']} di zona sehat")
    elif rsi_h > 70:
        add(-1, False, f"RSI 1H {h['rsi']} {'overbought' if side == 'LONG' else 'oversold'} (rawan berbalik)")
    else:
        reasons.append({"ok": None, "text": f"RSI 1H {h['rsi']} netral"})
    if h["adx"] >= 20:
        add(1, True, f"ADX 1H {h['adx']}: trend cukup kuat")
    else:
        reasons.append({"ok": None, "text": f"ADX 1H {h['adx']}: pasar cenderung sideways"})
    if h["vol_ratio"] and h["vol_ratio"] >= 1.3:
        add(1, True, f"Volume 1H {h['vol_ratio']}x rata-rata (ada konfirmasi)")
    if funding is not None:
        crowded = funding * sign > 0.03
        if crowded:
            add(-1, False, f"Funding {funding:+.4f}%: posisi {side} terlalu ramai")
    if room_r is not None:
        if room_r >= 1.5:
            add(1, True, f"Ruang ke level S/R terdekat {room_r:.1f}R (cukup)")
        else:
            add(-1, False, f"Level S/R terdekat hanya {room_r:.1f}R (ruang sempit)")
    return score, reasons


class AnalysisService:
    def __init__(self, scanner) -> None:
        self.exchange = Exchange()
        self.scanner = scanner
        self._lock = threading.Lock()
        self._cache: dict = {}

    def get(self, symbol: str) -> Optional[dict]:
        with self._lock:
            hit = self._cache.get(symbol)
            if hit and time.time() - hit[0] < CACHE_TTL:
                return hit[1]
            frames = {}
            for tf in TFS:
                df = self.exchange.fetch_ohlcv(symbol, tf, 220)
                if df is None or len(df) < 80:
                    return hit[1] if hit else None
                frames[tf] = df
            row = next((r for r in self.scanner.snapshot()["rows"] if r["symbol"] == symbol), None)
            result = self.build(symbol, frames, row)
            self._cache[symbol] = (time.time(), result)
            return result

    @staticmethod
    def build(symbol: str, frames: Dict[str, pd.DataFrame], row: Optional[dict]) -> dict:
        views = {tf: tf_view(frames[tf]) for tf in TFS}
        price = float(row["price"]) if row and row.get("price") else views["15m"]["close"]
        funding = row.get("funding") if row else None
        levels = nearest(swing_levels(frames["1h"], views["1h"]["atr"]), price)
        plans = {"scalp": plan(price, views["15m"]["atr"]), "swing": plan(price, views["1h"]["atr"])}
        risk = plans["swing"]["risk_dist"]

        def room(side):
            lv = levels["resistance"] if side == "LONG" else levels["support"]
            return abs(lv[0]["price"] - price) / risk if lv and risk > 0 else None

        long_s, long_r = score_side("LONG", views, funding, room("LONG"))
        short_s, short_r = score_side("SHORT", views, funding, room("SHORT"))
        quality = lambda s: int(max(0, min(100, round(50 + s * 7))))
        q_long, q_short = quality(long_s), quality(short_s)
        best, best_q = ("LONG", q_long) if q_long >= q_short else ("SHORT", q_short)
        verdict = best if best_q >= 65 else "WAIT"
        aligned = sum(1 for v in views.values() if v["trend"] == "UP") - sum(1 for v in views.values() if v["trend"] == "DOWN")
        snap_stub = {"ema20": views["1h"]["ema20"], "ema50": views["1h"]["ema50"], "rsi": views["1h"]["rsi"],
                     "adx": views["1h"]["adx"], "vol_ratio": views["1h"]["vol_ratio"], "regime": "TRENDING_UP" if aligned > 0 else "TRENDING_DOWN" if aligned < 0 else "SIDEWAYS",
                     "trap_score": 0, "sm": {"total": 0}}
        edges = pick_best_edge(snap_stub, price, mtf_aligned=aligned)
        profit = edges[best] if verdict in ("LONG", "SHORT") else edges["best"]
        return {"symbol": symbol, "price": price, "funding": funding, "views": views, "levels": levels, "plans": plans,
                "mtf": aligned, "verdict": verdict, "quality": {"LONG": q_long, "SHORT": q_short},
                "reasons": {"LONG": long_r, "SHORT": short_r}, "updated": time.time(),
                "change24": row.get("chg24") if row else None, "turnover": row.get("turnover") if row else None,
                "profit_edge": profit, "profit_edges": edges}
