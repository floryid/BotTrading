"""AI profit-edge scorer: menggabungkan prediksi model, teknikal, smart money, dan R:R."""
from typing import Dict, List, Optional, Tuple

from . import config
from .strategies import LONG, SHORT, WAIT

RR_RATIO = config.TP_ATR_MULT / config.SL_ATR_MULT if config.SL_ATR_MULT else 2.0


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _grade(score: float) -> str:
    if score >= 78:
        return "A"
    if score >= 68:
        return "B"
    if score >= 58:
        return "C"
    if score >= 48:
        return "D"
    return "F"


def _ai_alignment(action: str, price: float, preds: dict) -> Tuple[float, List[str]]:
    """Returns (0-1 alignment, reason snippets)."""
    if not price or action not in (LONG, SHORT):
        return 0.0, []
    sign = 1 if action == LONG else -1
    notes: List[str] = []
    parts: List[float] = []

    for key, label in (("lstm_delta_pct", "LSTM"), ("transformer_delta_pct", "Transformer")):
        dp = preds.get(key)
        if dp is None:
            continue
        aligned = (dp > 0) == (sign == 1)
        mag = min(abs(float(dp)) / 0.35, 1.0)
        parts.append(mag if aligned else -0.35 * mag)
        if aligned and mag >= 0.25:
            notes.append(f"{label} mendukung {action} ({dp:+.2f}%)")

    ens = preds.get("ensemble")
    if ens and ens != price:
        aligned = (ens > price) == (sign == 1)
        mag = min(abs(ens / price - 1) * 100 / 0.35, 1.0)
        parts.append(mag if aligned else -0.4 * mag)
        if aligned:
            notes.append(f"Ensemble harga target {ens:.4f}")

    ai_dir = preds.get("ai_direction")
    if ai_dir == action:
        parts.append(float(preds.get("ai_confidence", 0.5)))
        notes.append("Model AI searah")
    elif ai_dir in (LONG, SHORT) and ai_dir != action:
        parts.append(-0.5)

    if not parts:
        return 0.5, notes
    align = _clamp(50 + sum(parts) / len(parts) * 50) / 100
    return align, notes


def score_profit_edge(action: str, snap: dict, price: float, preds: Optional[dict] = None,
                      vote: Optional[dict] = None, conf: Optional[int] = None,
                      mtf_aligned: Optional[int] = None) -> dict:
    """Skor 0–100 seberapa layak arah `action` untuk setup profit (bukan jaminan)."""
    preds = preds or snap.get("preds") or {}
    if action not in (LONG, SHORT):
        return {"edge": 0, "win_prob": 0, "expected_r": 0, "grade": "F", "hot": False,
                "reasons": [], "side": WAIT}

    sign = 1 if action == LONG else -1
    score = 48.0
    reasons: List[str] = []

    regime = snap.get("regime", "UNKNOWN")
    if regime in ("TRENDING_UP", "TRENDING_DOWN"):
        aligned = (regime == "TRENDING_UP") == (sign == 1)
        score += 12 if aligned else -14
        reasons.append("Regime searah" if aligned else "Regime berlawanan")
    elif regime == "SIDEWAYS":
        score -= 8
        reasons.append("Pasar sideways")
    elif regime == "VOLATILE":
        score -= 22
        reasons.append("Volatilitas ekstrem")

    trend_ok = (snap["ema20"] > snap["ema50"]) == (sign == 1)
    score += 10 if trend_ok else -12

    rsi_v = snap["rsi"] if sign == 1 else 100 - snap["rsi"]
    if 42 <= rsi_v <= 68:
        score += 8
    elif rsi_v > 72 or rsi_v < 28:
        score -= 10

    adx = float(snap.get("adx") or 0)
    if adx >= 22:
        score += 6
    if snap.get("vol_ratio", 1) >= 1.25:
        score += 5

    sm = float(snap.get("sm", {}).get("total", 0) if isinstance(snap.get("sm"), dict) else snap.get("sm_total", 0))
    score += sign * sm * 18

    trap = int(snap.get("trap_score", 0))
    if trap >= 4:
        score -= 35
    elif trap >= 2:
        score -= 15

    ai_align, ai_notes = _ai_alignment(action, price, preds)
    score += (ai_align - 0.5) * 36
    reasons.extend(ai_notes[:3])

    if vote:
        vw = vote.get("long" if action == LONG else "short", 0)
        score += min(12, vw * 18)
        if vote.get("action") == action:
            reasons.append("Voting strategi searah")

    if conf is not None:
        score += (conf - 60) * 0.35

    if mtf_aligned is not None:
        score += mtf_aligned * sign * 4

    edge = int(_clamp(round(score)))
    # Win probability heuristic from edge + regime + AI agreement
    win_prob = _clamp(edge * 0.55 + ai_align * 28 + (8 if trend_ok else 0) + (6 if adx >= 20 else 0))
    win_prob = int(round(win_prob))
    exp_r = round((win_prob / 100) * RR_RATIO - (1 - win_prob / 100) * 1.0, 2)

    return {
        "edge": edge,
        "win_prob": win_prob,
        "expected_r": exp_r,
        "rr_ratio": round(RR_RATIO, 2),
        "grade": _grade(edge),
        "hot": edge >= config.MIN_PROFIT_EDGE and exp_r > 0,
        "ai_align": round(ai_align, 2),
        "side": action,
        "reasons": reasons[:6],
    }


def pick_best_edge(snap: dict, price: float, preds: Optional[dict] = None,
                   vote: Optional[dict] = None, conf_long: Optional[int] = None,
                   conf_short: Optional[int] = None, mtf_aligned: Optional[int] = None) -> dict:
    long_e = score_profit_edge(LONG, snap, price, preds, vote, conf_long, mtf_aligned)
    short_e = score_profit_edge(SHORT, snap, price, preds, vote, conf_short, mtf_aligned)
    best = long_e if long_e["edge"] >= short_e["edge"] else short_e
    return {"LONG": long_e, "SHORT": short_e, "best": best}


def scanner_row_edge(row: dict) -> dict:
    """Edge ringan untuk baris scanner (tanpa model PyTorch)."""
    if not row.get("signal") or row.get("rsi") is None:
        return {"edge": None, "side": None, "grade": "–", "hot": False}
    sig = row["signal"]
    side = LONG if sig in ("BULLISH", "OVERSOLD") else SHORT if sig in ("BEARISH", "OVERBOUGHT") else None
    if side is None:
        return {"edge": 45, "side": WAIT, "grade": "D", "hot": False}

    sign = 1 if side == LONG else -1
    score = 50.0
    trend_up = row.get("trend") == "UP"
    if (trend_up and side == LONG) or (not trend_up and side == SHORT):
        score += 14
    adx = float(row.get("adx") or 0)
    if adx >= 22:
        score += 8
    rsi = float(row["rsi"])
    if side == LONG and 30 < rsi < 62:
        score += 6
    if side == SHORT and 38 < rsi < 70:
        score += 6
    vr = row.get("vol_ratio") or 1
    if vr >= 1.5:
        score += min(10, (vr - 1) * 5)
    chg1 = row.get("chg1h") or 0
    if sign * chg1 > 0.5:
        score += 5
    fund = row.get("funding") or 0
    if sign * fund < -0.02:
        score += 4
    elif sign * fund > 0.05:
        score -= 6

    edge = int(_clamp(round(score)))
    exp_r = round((edge / 100) * RR_RATIO - (1 - edge / 100), 2)
    return {"edge": edge, "side": side, "grade": _grade(edge), "hot": edge >= config.MIN_PROFIT_EDGE and exp_r > 0,
            "win_prob": int(edge * 0.6), "expected_r": exp_r}
