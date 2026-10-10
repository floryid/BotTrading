"""Tabular Q-learning agent (5184 states x 3 actions)."""
import random
import threading

from . import config
from .storage import load_json, save_json

ACTIONS = ["LONG", "SHORT", "WAIT"]
DIMS = [2, 4, 3, 3, 3, 3, 2, 4]  # ema, rsi, volume, atr, trend strength, smart money, liq spike, regime
N_STATES = 1
for _d in DIMS:
    N_STATES *= _d  # 5184
REGIME_IDX = {"TRENDING_UP": 0, "TRENDING_DOWN": 1, "SIDEWAYS": 2, "VOLATILE": 3, "UNKNOWN": 2}


def encode_state(snap: dict, sm_score: float, liq_spike: bool) -> int:
    price = snap["price"] or 1.0
    rsi_v = snap["rsi"]
    rsi_zone = 0 if rsi_v < 30 else 1 if rsi_v < 50 else 2 if rsi_v < 65 else 3
    vol = snap.get("vol_ratio", 1.0)
    vol_z = 0 if vol < 0.8 else 1 if vol <= 1.3 else 2
    atr_r = snap.get("atr_ratio", 1.0)
    atr_z = 0 if atr_r < 0.8 else 1 if atr_r <= 1.3 else 2
    gap = abs(snap["ema20"] - snap["ema50"]) / price
    trend_z = 0 if gap < 0.002 else 1 if gap < 0.006 else 2
    sm_z = 0 if sm_score < -0.2 else 2 if sm_score > 0.2 else 1
    values = [int(snap["ema20"] > snap["ema50"]), rsi_zone, vol_z, atr_z, trend_z, sm_z,
              int(bool(liq_spike)), REGIME_IDX.get(snap.get("regime", "UNKNOWN"), 2)]
    idx = 0
    for v, d in zip(values, DIMS):
        idx = idx * d + v
    return idx


class RLAgent:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.q: dict = {}
        self.epsilon = config.RL_EPSILON_START
        self.updates = 0
        self.load()

    def _row(self, state: int) -> list:
        return self.q.setdefault(str(state), [0.0, 0.0, 0.0])

    def act(self, state: int, explore: bool = True) -> str:
        with self._lock:
            if explore and random.random() < self.epsilon:
                return random.choice(ACTIONS)
            row = self._row(state)
            best = max(row)
            # ties (untrained state) resolve to WAIT
            candidates = [i for i, v in enumerate(row) if v == best]
            return ACTIONS[2] if 2 in candidates else ACTIONS[candidates[0]]

    def update(self, state: int, action: str, reward: float, next_state: int) -> None:
        with self._lock:
            a = ACTIONS.index(action)
            row = self._row(state)
            target = reward + config.RL_GAMMA * max(self._row(next_state))
            row[a] += config.RL_ALPHA * (target - row[a])
            self.epsilon = max(config.RL_EPSILON_MIN, self.epsilon * config.RL_EPSILON_DECAY)
            self.updates += 1

    @staticmethod
    def trade_reward(pnl: float, equity: float) -> float:
        """pnl is normalised by base risk so rewards are scale-independent
        (at $10k equity this equals the documented pnl*0.01 / pnl*0.02)."""
        base = max(equity * config.BASE_RISK, 1e-9)
        r = pnl / base
        return min(r, 2.0) if pnl >= 0 else max(r * 2.0, -3.0)

    @staticmethod
    def wait_reward(conf: int) -> float:
        return 0.1 if conf < config.MIN_CONFIDENCE else -0.1

    def stats(self) -> dict:
        with self._lock:
            return {"epsilon": round(self.epsilon, 4), "updates": self.updates,
                    "states_visited": len(self.q), "total_states": N_STATES}

    def save(self) -> None:
        with self._lock:
            payload = {"q": self.q, "epsilon": self.epsilon, "updates": self.updates}
        save_json(config.QTABLE_FILE, payload)

    def load(self) -> None:
        data = load_json(config.QTABLE_FILE)
        if data:
            self.q = data.get("q", {})
            self.epsilon = float(data.get("epsilon", config.RL_EPSILON_START))
            self.updates = int(data.get("updates", 0))
