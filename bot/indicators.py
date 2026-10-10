"""Technical indicators and market-regime detection (pure pandas/numpy)."""
from typing import Optional

import numpy as np
import pandas as pd

REGIMES = ["TRENDING_UP", "TRENDING_DOWN", "SIDEWAYS", "VOLATILE", "UNKNOWN"]


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat([df["high"] - df["low"],
                        (df["high"] - prev_close).abs(),
                        (df["low"] - prev_close).abs()], axis=1)
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr_s = atr(df, period).replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["rsi"] = rsi(out["close"])
    out["tr"] = true_range(out)
    out["atr"] = atr(out)
    out["atr_mean"] = out["atr"].rolling(50, min_periods=20).mean()
    out["adx"] = adx(out)
    out["vol_ma"] = out["volume"].rolling(20, min_periods=5).mean()
    return out


def detect_regime(df: pd.DataFrame) -> str:
    """df must already contain indicators. Returns one of REGIMES."""
    if df is None or len(df) < 60:
        return "UNKNOWN"
    last = df.iloc[-1]
    if pd.isna(last["atr_mean"]) or last["atr_mean"] <= 0:
        return "UNKNOWN"
    if last["tr"] / last["atr_mean"] > 2.0:
        return "VOLATILE"
    if last["adx"] >= 25:
        return "TRENDING_UP" if last["ema20"] > last["ema50"] else "TRENDING_DOWN"
    return "SIDEWAYS"


def whale_trap_score(df: pd.DataFrame, liq_spike: bool = False, liq_bias: Optional[str] = None):
    """Heuristic stop-hunt / fake-breakout detector on the last closed candle.

    Returns (score 0-6, bias) where bias is 'bull_trap' (fake up-move, expect
    reversal down), 'bear_trap' (fake down-move, expect reversal up) or None.
    """
    if df is None or len(df) < 25:
        return 0, None
    last = df.iloc[-1]
    prior = df.iloc[-21:-1]
    rng = float(last["high"] - last["low"])
    if rng <= 0:
        return 0, None
    body = abs(float(last["close"] - last["open"]))
    upper = float(last["high"] - max(last["open"], last["close"]))
    lower = float(min(last["open"], last["close"]) - last["low"])
    vol_ma = float(last["vol_ma"]) if "vol_ma" in df.columns and not pd.isna(last["vol_ma"]) else 0.0
    vol_ratio = float(last["volume"]) / vol_ma if vol_ma > 0 else 1.0

    bull = bear = 0
    # 1) long rejection wick on elevated volume
    if vol_ratio > 1.5:
        if upper > 2 * body and upper > 0.6 * rng:
            bull += 2
        if lower > 2 * body and lower > 0.6 * rng:
            bear += 2
    # 2) fake breakout of the 20-candle range
    if last["high"] > prior["high"].max() and last["close"] < prior["high"].max():
        bull += 2
    if last["low"] < prior["low"].min() and last["close"] > prior["low"].min():
        bear += 2
    # 3) volume spike with indecisive body: attribute to the dominant wick side
    if vol_ratio > 2.5 and body < 0.3 * rng:
        if upper >= lower:
            bull += 1
        else:
            bear += 1
    # 4) liquidation spike confirms the direction of the trap
    if liq_spike and liq_bias == "bull_trap":
        bull += 1
    elif liq_spike and liq_bias == "bear_trap":
        bear += 1

    if bull == bear:
        return min(bull, 6), None
    if bull > bear:
        return min(bull, 6), "bull_trap"
    return min(bear, 6), "bear_trap"
