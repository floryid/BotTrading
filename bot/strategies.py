"""Multi-strategy voting system (Trend, Mean Reversion, Whale, Scalper, Transformer)."""
from typing import Optional

from . import config

LONG, SHORT, WAIT = "LONG", "SHORT", "WAIT"


def _flip(signal: str) -> str:
    return {LONG: SHORT, SHORT: LONG}.get(signal, WAIT)


def vote(snap: dict, price: float, prev_price: Optional[float], tf_pred: Optional[float]) -> dict:
    """Run every strategy and aggregate.

    snap keys used: ema20, ema50, rsi, trap_score, trap_bias.
    Whale trap >= 2 flips every signal (reversal logic) and the Whale
    strategy itself votes for the reversal direction.
    """
    sig = {}
    sig["trend"] = LONG if snap["ema20"] > snap["ema50"] else SHORT
    rsi_v = snap["rsi"]
    sig["mean_reversion"] = LONG if rsi_v < 30 else SHORT if rsi_v > 70 else WAIT

    sig["scalper"] = WAIT
    if prev_price and prev_price > 0:
        if price > prev_price * (1 + config.SCALPER_THRESHOLD):
            sig["scalper"] = LONG
        elif price < prev_price * (1 - config.SCALPER_THRESHOLD):
            sig["scalper"] = SHORT

    sig["transformer"] = WAIT
    if tf_pred:
        sig["transformer"] = LONG if tf_pred > price else SHORT if tf_pred < price else WAIT

    sig["ai_ensemble"] = WAIT
    ens = snap.get("preds", {}).get("ensemble") if isinstance(snap.get("preds"), dict) else None
    if ens:
        sig["ai_ensemble"] = LONG if ens > price else SHORT if ens < price else WAIT
    ai_dir = snap.get("preds", {}).get("ai_direction") if isinstance(snap.get("preds"), dict) else None
    if ai_dir in (LONG, SHORT) and sig["ai_ensemble"] == WAIT:
        sig["ai_ensemble"] = ai_dir

    trap = int(snap.get("trap_score", 0))
    bias = snap.get("trap_bias")
    flipped = trap >= 2 and bias is not None
    sig["whale"] = WAIT
    if flipped:
        for name in ("trend", "mean_reversion", "scalper", "transformer", "ai_ensemble"):
            sig[name] = _flip(sig[name])
        sig["whale"] = SHORT if bias == "bull_trap" else LONG

    weights = {**config.WEIGHTS, "transformer": config.TRANSFORMER_WEIGHT, "ai_ensemble": config.AI_ENSEMBLE_WEIGHT}
    long_w = sum(weights[n] for n, s in sig.items() if s == LONG)
    short_w = sum(weights[n] for n, s in sig.items() if s == SHORT)

    action = WAIT
    if long_w > short_w and long_w >= config.MIN_VOTE_WEIGHT:
        action = LONG
    elif short_w > long_w and short_w >= config.MIN_VOTE_WEIGHT:
        action = SHORT
    return {"action": action, "long": round(long_w, 2), "short": round(short_w, 2),
            "signals": sig, "flipped": flipped}


def combine_rl_vote(rl_action: str, vote_action: str):
    """Returns (action, confidence_bonus) following the RL+Voting matrix."""
    if rl_action == vote_action:
        return (rl_action, 10) if rl_action != WAIT else (WAIT, 0)
    if vote_action != WAIT:
        return vote_action, 0   # voting wins any disagreement
    return rl_action, 0         # voting WAIT -> RL acts alone
