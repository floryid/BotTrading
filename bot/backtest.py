"""Backtest of the EMA20/50 + RSI strategy on 1H candles (startup viability gate)."""
import logging
import math
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from . import config
from .indicators import add_indicators

log = logging.getLogger("backtest")
COST = 2 * (config.TAKER_FEE + config.SLIPPAGE)  # round-trip cost as a fraction of notional


def backtest_symbol(df: pd.DataFrame) -> dict:
    """Walk forward over candles. Entry on EMA cross (RSI filtered) at the next open;
    exit on ATR SL/TP (SL wins when both touch inside one candle). Position size
    risks BASE_RISK of equity, capped at leverage."""
    d = add_indicators(df)
    equity, peak, max_dd = 1.0, 1.0, 0.0
    returns, wins_pnl, loss_pnl = [], 0.0, 0.0
    pos = None
    for i in range(55, len(d) - 1):
        row, nxt = d.iloc[i], d.iloc[i + 1]
        if pos is None:
            prev = d.iloc[i - 1]
            side = None
            if prev["ema20"] <= prev["ema50"] and row["ema20"] > row["ema50"] and row["rsi"] < 70:
                side = 1
            elif prev["ema20"] >= prev["ema50"] and row["ema20"] < row["ema50"] and row["rsi"] > 30:
                side = -1
            if side and row["atr"] > 0:
                entry = float(nxt["open"])
                sl_dist = config.SL_ATR_MULT * float(row["atr"])
                pos = {"side": side, "entry": entry, "sl": entry - side * sl_dist,
                       "tp": entry + side * config.TP_ATR_MULT * float(row["atr"]),
                       "size": min(config.BASE_RISK / (sl_dist / entry), config.LEVERAGE), "start": i + 1}
            continue
        if i < pos["start"]:
            continue
        hit_sl = (row["low"] <= pos["sl"]) if pos["side"] == 1 else (row["high"] >= pos["sl"])
        hit_tp = (row["high"] >= pos["tp"]) if pos["side"] == 1 else (row["low"] <= pos["tp"])
        if hit_sl or hit_tp:
            exit_price = pos["sl"] if hit_sl else pos["tp"]
            move = pos["side"] * (exit_price - pos["entry"]) / pos["entry"]
            ret = pos["size"] * (move - COST)
            equity *= 1 + ret
            returns.append(ret)
            wins_pnl += max(ret, 0)
            loss_pnl += max(-ret, 0)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
            pos = None

    n = len(returns)
    wins = sum(1 for r in returns if r > 0)
    mean = sum(returns) / n if n else 0.0
    std = math.sqrt(sum((r - mean) ** 2 for r in returns) / n) if n > 1 else 0.0
    return {"trades": n, "wins": wins, "win_rate": round(100 * wins / n, 2) if n else 0.0,
            "profit_factor": round(wins_pnl / loss_pnl, 2) if loss_pnl > 0 else (99.0 if wins_pnl > 0 else 0.0),
            "pnl_pct": round((equity - 1) * 100, 2),
            "sharpe": round(mean / std * math.sqrt(n), 2) if std > 0 else 0.0,
            "max_dd_pct": round(max_dd * 100, 2), "gross_win": wins_pnl, "gross_loss": loss_pnl}


def run_backtest(exchange, symbols) -> Optional[dict]:
    per_symbol = {}
    for sym in symbols:
        df = exchange.fetch_ohlcv(sym, "1h", config.BACKTEST_CANDLES)
        if df is None or len(df) < 120:
            log.warning("Backtest %s: not enough data", sym)
            continue
        per_symbol[sym] = backtest_symbol(df)
    if not per_symbol:
        return None
    trades = sum(r["trades"] for r in per_symbol.values())
    wins = sum(r["wins"] for r in per_symbol.values())
    gw = sum(r["gross_win"] for r in per_symbol.values())
    gl = sum(r["gross_loss"] for r in per_symbol.values())
    win_rate = 100 * wins / trades if trades else 0.0
    pf = gw / gl if gl > 0 else (99.0 if gw > 0 else 0.0)
    viable = (trades >= config.BACKTEST_MIN_TRADES and win_rate >= config.BACKTEST_MIN_WINRATE
              and pf >= config.BACKTEST_MIN_PF)
    for r in per_symbol.values():
        r.pop("gross_win")
        r.pop("gross_loss")
    return {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "symbols": per_symbol,
            "summary": {"trades": trades, "win_rate": round(win_rate, 2), "profit_factor": round(pf, 2),
                        "viable": viable,
                        "criteria": f"trades>={config.BACKTEST_MIN_TRADES}, win rate>={config.BACKTEST_MIN_WINRATE}%, "
                                    f"PF>={config.BACKTEST_MIN_PF}"}}
