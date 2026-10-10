"""Backtest Lab: on-demand strategy tests with realistic costs and an out-of-sample check."""
import math
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .exchange import Exchange
from .indicators import adx, atr, ema, rsi

COST = 2 * (config.TAKER_FEE + config.SLIPPAGE)  # round-trip cost as a fraction of notional
WARMUP = 210
OOS_SPLIT = 0.7

STRATEGIES: Dict[str, dict] = {
    "ema_cross": {"name": "EMA Cross + filter RSI", "desc": "Long saat EMA cepat memotong naik EMA lambat (RSI < 70); short sebaliknya.",
                  "params": [{"key": "fast", "label": "EMA cepat", "default": 20}, {"key": "slow", "label": "EMA lambat", "default": 50}]},
    "rsi_reversion": {"name": "RSI Mean Reversion", "desc": "Long saat RSI naik kembali melewati batas bawah; short saat turun melewati batas atas.",
                      "params": [{"key": "low", "label": "RSI bawah", "default": 30}, {"key": "high", "label": "RSI atas", "default": 70}]},
    "donchian": {"name": "Donchian Breakout", "desc": "Long saat close menembus high N candle (ADX ≥ 20); short saat menembus low N candle.",
                 "params": [{"key": "n", "label": "Periode N", "default": 20}, {"key": "adx_min", "label": "ADX minimum", "default": 20}]},
    "trend_pullback": {"name": "Trend Pullback", "desc": "Searah trend (EMA20 vs EMA100): masuk saat RSI pulih dari koreksi (naik melewati level).",
                       "params": [{"key": "level", "label": "Level RSI", "default": 40}, {"key": "slow", "label": "EMA trend", "default": 100}]},
}


def resolve(strategy: str, params: dict) -> dict:
    out = {}
    for p in STRATEGIES[strategy]["params"]:
        try:
            out[p["key"]] = float(params.get(p["key"], p["default"]))
        except (TypeError, ValueError):
            out[p["key"]] = float(p["default"])
    return out


def signals(df: pd.DataFrame, strategy: str, params: dict) -> pd.Series:
    """+1 / -1 / 0 evaluated on each bar close."""
    p = resolve(strategy, params)
    close = df["close"]
    sig = pd.Series(0, index=df.index)
    if strategy == "ema_cross":
        fast, slow = ema(close, int(p["fast"])), ema(close, int(p["slow"]))
        r = rsi(close)
        up = (fast.shift(1) <= slow.shift(1)) & (fast > slow) & (r < 70)
        dn = (fast.shift(1) >= slow.shift(1)) & (fast < slow) & (r > 30)
    elif strategy == "rsi_reversion":
        r = rsi(close)
        up = (r.shift(1) < p["low"]) & (r >= p["low"])
        dn = (r.shift(1) > p["high"]) & (r <= p["high"])
    elif strategy == "donchian":
        n = int(p["n"])
        hi, lo = df["high"].rolling(n).max().shift(1), df["low"].rolling(n).min().shift(1)
        a = adx(df)
        up, dn = (close > hi) & (a >= p["adx_min"]), (close < lo) & (a >= p["adx_min"])
    else:  # trend_pullback
        r = rsi(close)
        trend = ema(close, int(p["slow"]))
        fast = ema(close, 20)
        up = (fast > trend) & (r.shift(1) < p["level"]) & (r >= p["level"])
        dn = (fast < trend) & (r.shift(1) > 100 - p["level"]) & (r <= 100 - p["level"])
    sig[up] = 1
    sig[dn] = -1
    sig.iloc[:WARMUP] = 0
    return sig


def simulate(df: pd.DataFrame, sig: pd.Series, sl_mult: float, tp_mult: float, max_hold: int, side: str) -> List[dict]:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr(df).to_numpy(float)
    ts = df["timestamp"].to_numpy()
    s = sig.to_numpy()
    n, trades, i = len(df), [], WARMUP
    while i < n - 1:
        direction = int(s[i])
        if direction == 0 or (side == "long" and direction < 0) or (side == "short" and direction > 0) or not a[i] > 0:
            i += 1
            continue
        entry = o[i + 1]
        sl_dist = sl_mult * a[i]
        sl, tp = entry - direction * sl_dist, entry + direction * tp_mult * a[i]
        size = min(config.BASE_RISK / (sl_dist / entry), config.LEVERAGE)
        exit_price, reason, j = c[-1], "END", n - 1
        for j in range(i + 1, n):
            hit_sl = l[j] <= sl if direction > 0 else h[j] >= sl
            hit_tp = h[j] >= tp if direction > 0 else l[j] <= tp
            if hit_sl or hit_tp:
                exit_price, reason = (sl, "SL") if hit_sl else (tp, "TP")
                break
            if max_hold and j - i >= max_hold:
                exit_price, reason = c[j], "TIME"
                break
        move = direction * (exit_price - entry) / entry
        trades.append({"entry_i": i + 1, "exit_i": j, "t_in": int(ts[i + 1]) // 1000, "t_out": int(ts[j]) // 1000,
                       "side": "LONG" if direction > 0 else "SHORT", "entry": float(entry), "exit": float(exit_price),
                       "ret": float(size * (move - COST)), "r": float((move - COST) / (sl_dist / entry)), "reason": reason})
        i = j + 1
    return trades


def metrics(trades: List[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "return_pct": 0.0, "max_dd_pct": 0.0,
                "sharpe": 0.0, "avg_r": 0.0, "expectancy_pct": 0.0}
    rets = [t["ret"] for t in trades]
    wins = [r for r in rets if r > 0]
    gross_loss = -sum(r for r in rets if r <= 0)
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in rets:
        eq *= 1 + r
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)
    mean = sum(rets) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rets) / n) if n > 1 else 0.0
    pf = sum(wins) / gross_loss if gross_loss > 0 else (99.0 if wins else 0.0)
    return {"trades": n, "win_rate": round(100 * len(wins) / n, 1), "profit_factor": round(min(pf, 99.0), 2),
            "return_pct": round((eq - 1) * 100, 2), "max_dd_pct": round(mdd * 100, 2),
            "sharpe": round(mean / std * math.sqrt(n), 2) if std > 0 else 0.0,
            "avg_r": round(sum(t["r"] for t in trades) / n, 2), "expectancy_pct": round(mean * 100, 3)}


def verdict(is_m: dict, oos_m: dict) -> dict:
    if oos_m["trades"] < 15:
        return {"level": "warn", "text": f"Data uji terlalu sedikit ({oos_m['trades']} trade out-of-sample). Perpanjang periode atau ganti timeframe."}
    if oos_m["profit_factor"] >= 1.2 and oos_m["return_pct"] > 0 and is_m["profit_factor"] >= 1.0:
        return {"level": "ok", "text": "Ada indikasi edge: profit factor out-of-sample ≥ 1,2. Tetap uji lagi di koin/periode lain dan di paper trading."}
    return {"level": "bad", "text": "Tidak meyakinkan: hasil out-of-sample tidak memenuhi syarat (PF ≥ 1,2 dan return positif). Jangan dipakai live."}


def run(df: pd.DataFrame, strategy: str, params: dict, sl_mult: float, tp_mult: float, max_hold: int, side: str) -> dict:
    sig = signals(df, strategy, params)
    trades = simulate(df, sig, sl_mult, tp_mult, max_hold, side)
    split = WARMUP + int((len(df) - WARMUP) * OOS_SPLIT)
    is_t = [t for t in trades if t["entry_i"] < split]
    oos_t = [t for t in trades if t["entry_i"] >= split]
    is_m, oos_m = metrics(is_t), metrics(oos_t)
    curve, eq = [[int(df["timestamp"].iloc[WARMUP]) // 1000, 100.0]], 1.0
    for t in trades:
        eq *= 1 + t["ret"]
        curve.append([t["t_out"], round(eq * 100, 3)])
    closes = df["close"].to_numpy(float)
    return {"all": metrics(trades), "in_sample": is_m, "out_of_sample": oos_m, "verdict": verdict(is_m, oos_m),
            "curve": curve, "split_t": int(df["timestamp"].iloc[split]) // 1000 if split < len(df) else None,
            "buy_hold_pct": round((closes[-1] / closes[WARMUP] - 1) * 100, 2),
            "trades": [{k: t[k] for k in ("t_in", "t_out", "side", "entry", "exit", "r", "reason")} for t in trades[-60:]][::-1],
            "candles": len(df), "from": int(df["timestamp"].iloc[0]) // 1000, "to": int(df["timestamp"].iloc[-1]) // 1000}


class LabService:
    def __init__(self) -> None:
        self.exchange = Exchange()
        self._lock = threading.Lock()
        self._hist: dict = {}

    def history(self, symbol: str, tf: str, n: int) -> Optional[pd.DataFrame]:
        key = (symbol, tf, n)
        hit = self._hist.get(key)
        if hit and time.time() - hit[0] < 600:
            return hit[1]
        df = self.exchange.fetch_history(symbol, tf, n)
        if df is not None and len(df) > WARMUP + 50:
            self._hist[key] = (time.time(), df)
            if len(self._hist) > 12:
                self._hist.pop(next(iter(self._hist)))
            return df
        return None

    def run(self, symbol: str, tf: str, strategy: str, params: dict, sl: float, tp: float, n: int, hold: int, side: str,
            compare: bool = False) -> Optional[dict]:
        with self._lock:
            df = self.history(symbol, tf, n)
            if df is None:
                return None
            if compare:
                rows = []
                for key, meta in STRATEGIES.items():
                    res = run(df, key, {}, sl, tp, hold, side)
                    rows.append({"strategy": key, "name": meta["name"], "all": res["all"], "in_sample": res["in_sample"],
                                 "out_of_sample": res["out_of_sample"], "verdict": res["verdict"]})
                rows.sort(key=lambda r: (r["out_of_sample"]["profit_factor"] if r["out_of_sample"]["trades"] >= 5 else -1), reverse=True)
                return {"compare": rows, "candles": len(df), "symbol": symbol, "tf": tf}
            result = run(df, strategy, params, sl, tp, hold, side)
            result.update(symbol=symbol, tf=tf, strategy=strategy, params=resolve(strategy, params))
            return result
