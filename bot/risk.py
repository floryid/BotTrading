"""Confidence score, dynamic risk sizing, position sizing and entry filters."""
from typing import Dict, Optional, Tuple

import pandas as pd

from . import config
from .strategies import LONG, SHORT, WAIT


def confidence_score(action: str, snap: dict, price: float, ai_pred: Optional[float],
                     sm_score: float, rl_vote_agree: bool) -> int:
    """0-100 score for the chosen direction (bullish table mirrored for SHORT)."""
    if action not in (LONG, SHORT):
        return 0
    sign = 1 if action == LONG else -1
    score = 50.0
    score += 20 if (snap["ema20"] > snap["ema50"]) == (sign == 1) else -20

    rsi_v = snap["rsi"] if sign == 1 else 100 - snap["rsi"]  # mirror so 45-65 is "optimal" for both sides
    if 45 <= rsi_v <= 65:
        score += 15
    if snap["rsi"] < 30 or snap["rsi"] > 70:
        score -= 25

    if snap.get("vol_ratio", 1.0) > 1.3:
        score += 10
    if ai_pred and ((ai_pred > price) == (sign == 1)) and ai_pred != price:
        score += 10
    score += sign * sm_score * 20
    if rl_vote_agree:
        score += 10

    trap = snap.get("trap_score", 0)
    if trap >= 4:
        score -= 50
    elif trap >= 2:
        score -= 25
    regime = snap.get("regime")
    if regime == "VOLATILE":
        score -= 40
    elif regime == "SIDEWAYS":
        score -= 20
    return int(max(0, min(100, round(score))))


def dynamic_risk(symbol: str, price: float, atr_value: float, regime: str, sm_score: float) -> float:
    """Risk fraction of equity for this trade (0 means do not trade)."""
    mult = 1.0
    if regime == "VOLATILE":
        return 0.0
    if price > 0 and atr_value / price > 0.05:
        mult *= 0.5
    if symbol in config.SMALL_CAP_SYMBOLS:
        mult *= 0.5
    if sm_score < -0.5:
        mult *= 0.5
    if regime == "SIDEWAYS":
        mult *= 0.3
    return min(config.BASE_RISK * mult, config.MAX_RISK_PER_TRADE)


def sl_tp(side: str, price: float, atr_value: float) -> Tuple[float, float]:
    sl_dist = config.SL_ATR_MULT * atr_value
    tp_dist = config.TP_ATR_MULT * atr_value
    if side == LONG:
        return price - sl_dist, price + tp_dist
    return price + sl_dist, price - tp_dist


def calc_qty(equity: float, risk_pct: float, price: float, atr_value: float,
             used_margin: float = 0.0) -> float:
    """Size so that hitting SL loses `risk_pct` of equity, capped by free margin at leverage."""
    sl_dist = config.SL_ATR_MULT * atr_value
    if equity <= 0 or sl_dist <= 0 or price <= 0 or risk_pct <= 0:
        return 0.0
    qty = equity * risk_pct / sl_dist
    free_margin = max(0.0, equity * 0.95 - used_margin)
    max_notional = free_margin * config.LEVERAGE
    return max(0.0, min(qty, max_notional / price))


def r_multiple(pos: dict, price: float) -> Optional[float]:
    """Open profit in units of the initial risk (distance entry -> original SL)."""
    d = 1 if pos["side"] == LONG else -1
    risk = abs(pos["entry"] - pos.get("sl0", pos["sl"]))
    if risk <= 0:
        return None
    return (price - pos["entry"]) * d / risk


def next_stop(pos: dict, price: float) -> Tuple[Optional[float], str]:
    """Protective-stop management. Returns (new_sl, reason) or (None, '').
    Break-even moves the SL just past entry (covering fees) after BREAKEVEN_R;
    trailing then follows price at TRAIL_ATR_MULT x ATR. The SL only ever tightens."""
    r = r_multiple(pos, price)
    if r is None:
        return None, ""
    d = 1 if pos["side"] == LONG else -1
    best, reason = None, ""
    if config.BREAKEVEN_R > 0 and not pos.get("be") and r >= config.BREAKEVEN_R:
        buffer = pos["entry"] * (2 * config.TAKER_FEE + config.SLIPPAGE)
        best, reason = pos["entry"] + d * buffer, "BREAKEVEN"
    atr_value = pos.get("atr")
    if config.TRAIL_START_R > 0 and atr_value and r >= config.TRAIL_START_R:
        trail = price - d * config.TRAIL_ATR_MULT * atr_value
        if best is None or (trail - best) * d > 0:
            best, reason = trail, "TRAIL"
    if best is None or (best - pos["sl"]) * d <= 0:
        return None, ""
    if reason == "TRAIL" and atr_value and abs(best - pos["sl"]) < 0.25 * atr_value and pos.get("be"):
        return None, ""  # avoid chattering API calls for tiny steps
    return best, reason


def correlation_matrix(closes: Dict[str, pd.Series]) -> pd.DataFrame:
    frame = pd.DataFrame({k: v.reset_index(drop=True) for k, v in closes.items() if v is not None and len(v) > 30})
    if frame.empty:
        return pd.DataFrame()
    return frame.pct_change().dropna().corr()


def check_filters(action: str, symbol: str, conf: int, regime: str, trap_score: int, sm_score: float,
                  corr: pd.DataFrame, positions: dict, risk_pct: float,
                  profit_edge: Optional[int] = None) -> Tuple[bool, str]:
    """Returns (passed, reason_if_skipped)."""
    if action == WAIT:
        return False, "WAIT"
    if regime == "VOLATILE":
        return False, "VOLATILE"
    if trap_score >= 4:
        return False, "WHALE TRAP"
    if conf < config.MIN_CONFIDENCE:
        return False, f"CONFIDENCE {conf}"
    if profit_edge is not None and profit_edge < config.MIN_PROFIT_EDGE:
        return False, f"PROFIT EDGE {profit_edge}"
    if (action == LONG and sm_score < -0.6) or (action == SHORT and sm_score > 0.6):
        return False, "SMART MONEY"
    if risk_pct <= 0:
        return False, "RISK 0"

    if corr is not None and not corr.empty and symbol in corr.index:
        for other, pos in positions.items():
            if other == symbol or pos["side"] != action or other not in corr.columns:
                continue
            if corr.loc[symbol, other] > config.MAX_CORRELATION:
                return False, f"KORELASI {other}"

    total = sum(p.get("risk_pct", 0.0) for p in positions.values())
    if total + risk_pct > config.MAX_TOTAL_RISK + 1e-9:
        return False, "MAX TOTAL RISK"
    return True, ""
