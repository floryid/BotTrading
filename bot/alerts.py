"""User-defined alerts evaluated against scanner data; persisted to data/alerts.json."""
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple

from . import config
from .storage import load_json, save_json

NUMERIC = {"price_above": "price", "price_below": "price", "rsi_above": "rsi", "rsi_below": "rsi",
           "change24_above": "chg24", "change24_below": "chg24", "vol_above": "vol_ratio"}
SIGNALS = {"BULLISH", "BEARISH", "OVERBOUGHT", "OVERSOLD", "NEUTRAL"}
LABELS = {"price_above": "harga ≥", "price_below": "harga ≤", "rsi_above": "RSI 1H ≥", "rsi_below": "RSI 1H ≤",
          "change24_above": "perubahan 24h ≥", "change24_below": "perubahan 24h ≤", "vol_above": "lonjakan volume ≥",
          "signal_is": "sinyal ="}
MAX_ALERTS = 50
REPEAT_COOLDOWN = 1800


class AlertManager:
    def __init__(self, notifier, on_event: Callable[[str, str, str, Optional[str]], None]) -> None:
        self.notifier = notifier
        self.on_event = on_event
        self._lock = threading.Lock()
        self.alerts: List[dict] = load_json(config.ALERTS_FILE, []) or []

    def list(self) -> List[dict]:
        with self._lock:
            return [dict(a) for a in self.alerts]

    def add(self, symbol: str, kind: str, value, repeat: bool, note: str,
            row: Optional[dict]) -> Tuple[Optional[dict], str]:
        if row is None:
            return None, "Koin tidak ada di scanner"
        if kind not in LABELS:
            return None, "Tipe alert tidak dikenal"
        if kind == "signal_is":
            value = str(value).upper()
            if value not in SIGNALS:
                return None, "Sinyal tidak valid"
        else:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None, "Nilai harus angka"
            if kind.startswith("price") and value <= 0:
                return None, "Harga harus > 0"
            if kind == "price_above" and row["price"] >= value:
                return None, f"Harga sekarang ({row['price']}) sudah di atas nilai itu"
            if kind == "price_below" and row["price"] <= value:
                return None, f"Harga sekarang ({row['price']}) sudah di bawah nilai itu"
        with self._lock:
            if len(self.alerts) >= MAX_ALERTS:
                return None, f"Maksimal {MAX_ALERTS} alert"
            alert = {"id": uuid.uuid4().hex[:8], "symbol": symbol, "type": kind, "value": value, "repeat": bool(repeat),
                     "note": str(note)[:80], "active": True, "created": time.time(), "triggered_at": None, "count": 0}
            self.alerts.append(alert)
            self._save()
            return dict(alert), ""

    def remove(self, alert_id: str) -> bool:
        with self._lock:
            before = len(self.alerts)
            self.alerts = [a for a in self.alerts if a["id"] != alert_id]
            changed = len(self.alerts) != before
            if changed:
                self._save()
            return changed

    def check(self, rows: Dict[str, dict]) -> None:
        now, fired = time.time(), []
        with self._lock:
            for a in self.alerts:
                row = rows.get(a["symbol"])
                if not a["active"] or row is None:
                    continue
                if a["repeat"] and a["triggered_at"] and now - a["triggered_at"] < REPEAT_COOLDOWN:
                    continue
                if self._hit(a, row):
                    a["triggered_at"], a["count"] = now, a["count"] + 1
                    if not a["repeat"]:
                        a["active"] = False
                    fired.append((dict(a), row))
            if fired:
                self._save()
        for a, row in fired:
            shown = a["value"] if a["type"] == "signal_is" else f"{a['value']:g}"
            current = {"price": row.get("price"), "rsi": row.get("rsi"), "chg24": row.get("chg24"),
                       "vol_ratio": row.get("vol_ratio"), "signal": row.get("signal")}.get(
                NUMERIC.get(a["type"], "signal"))
            text = f"{a['symbol']}: {LABELS[a['type']]} {shown} (sekarang {current})" + (f" — {a['note']}" if a["note"] else "")
            self.on_event("ALERT", a["symbol"], text, None)
            self.notifier.send(f"🔔 ALERT {text}")

    @staticmethod
    def _hit(a: dict, row: dict) -> bool:
        kind = a["type"]
        if kind == "signal_is":
            return row.get("signal") == a["value"]
        current = row.get(NUMERIC[kind])
        if current is None:
            return False
        return current >= a["value"] if kind.endswith("above") else current <= a["value"]

    def _save(self) -> None:
        save_json(config.ALERTS_FILE, self.alerts)
