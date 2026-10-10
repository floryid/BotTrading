"""Paper-trading engine: virtual wallet with fees and slippage."""
import threading
import time
from typing import Optional

from . import config


def direction(side: str) -> int:
    return 1 if side == "LONG" else -1


class PaperEngine:
    def __init__(self, balance: float = config.PAPER_BALANCE) -> None:
        self.balance = float(balance)
        self._lock = threading.Lock()

    def execute(self, symbol: str, side: str, qty: float, price: float, sl: float, tp: float,
                meta: Optional[dict] = None) -> dict:
        """Open a virtual position. Slippage worsens the fill; the taker fee is charged immediately."""
        d = direction(side)
        entry = price * (1 + d * config.SLIPPAGE)
        fee = entry * qty * config.TAKER_FEE
        with self._lock:
            self.balance -= fee
        pos = {"symbol": symbol, "side": side, "entry": entry, "qty": qty, "sl": sl, "tp": tp,
               "entry_fee": fee, "opened_at": time.time(), "mode": "PAPER"}
        pos.update(meta or {})
        return pos

    @staticmethod
    def check_sl_tp(pos: dict, price: float) -> Optional[str]:
        d = direction(pos["side"])
        if (price - pos["sl"]) * d <= 0:
            return "SL"
        if (price - pos["tp"]) * d >= 0:
            return "TP"
        return None

    def close(self, pos: dict, price: float, reason: str) -> dict:
        """Close at `price` (slippage applied). Returns {'pnl','exit','fees'} with net pnl."""
        d = direction(pos["side"])
        exit_price = price * (1 - d * config.SLIPPAGE)
        gross = (exit_price - pos["entry"]) * pos["qty"] * d
        exit_fee = exit_price * pos["qty"] * config.TAKER_FEE
        with self._lock:
            self.balance += gross - exit_fee
        fees = pos.get("entry_fee", 0.0) + exit_fee
        return {"pnl": gross - fees, "exit": exit_price, "fees": fees, "reason": reason}

    @staticmethod
    def unrealized(pos: dict, price: float) -> float:
        return (price - pos["entry"]) * pos["qty"] * direction(pos["side"])

    def equity(self, positions: dict, prices: dict) -> float:
        total = self.balance
        for sym, pos in positions.items():
            if pos.get("mode") == "PAPER":
                total += self.unrealized(pos, prices.get(sym, pos["entry"]))
        return total
