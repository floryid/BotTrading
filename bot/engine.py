"""Trading engine: dual loop (fast 1s / slow 15m) orchestration."""
import logging
import threading
import time
from collections import deque
from typing import Optional

import pandas as pd

from . import config
from .backtest import run_backtest
from .exchange import Exchange
from .indicators import add_indicators, detect_regime, whale_trap_score
from .models import ModelManager
from .notifier import Notifier
from .paper import PaperEngine, direction
from .risk import (calc_qty, check_filters, confidence_score, correlation_matrix, dynamic_risk, next_stop,
                   r_multiple, sl_tp)
from .rl import RLAgent, encode_state
from .smart_money import SmartMoney
from .storage import load_json, save_json
from .profit_edge import pick_best_edge
from .strategies import LONG, SHORT, WAIT, combine_rl_vote, vote
from .ws_feed import PriceFeed

log = logging.getLogger("engine")
NOTIFY_SKIP_REASONS = ("VOLATILE", "WHALE TRAP", "KORELASI")


class TradingBot:
    def __init__(self) -> None:
        self.mode = "PAPER" if config.PAPER_TRADING else "LIVE"
        self.notifier = Notifier()
        self.exchange = Exchange()
        self.paper = PaperEngine()
        self.rl = RLAgent()
        self.models = ModelManager()
        self.smart = SmartMoney(self.exchange)
        self.feed = PriceFeed(config.SYMBOLS, on_giveup=self._ws_giveup)

        self.positions: dict = {}
        self.df_cache: dict = {}
        self.snapshots: dict = {}
        self.last_decision: dict = {}
        self.prev_price: dict = {}
        self.cooldown: dict = {}
        self._last_skip: dict = {}
        self.corr = pd.DataFrame()
        self.closed_trades: list = []
        self.equity_curve: deque = deque(maxlen=600)
        self.events: deque = deque(maxlen=60)
        self.backtest_result: Optional[dict] = None

        self._equity_lock = threading.Lock()
        self.equity = config.PAPER_BALANCE
        self.peak = config.PAPER_BALANCE
        self._live_equity: Optional[float] = None
        self._last_curve_ts = 0.0
        self._last_sync = 0.0

        self.dashboard_lock = threading.Lock()
        self.dashboard_data: dict = {}
        self.stop_event = threading.Event()
        self.halted = False
        self.paused = False
        self.exit_code = 0
        self.started_at = time.time()
        self.last_slow: Optional[float] = None

    # ------------------------------------------------------------------ startup
    def startup(self) -> bool:
        """9-step initialisation. Returns False if the bot must not start."""
        log.info("=== AI HEDGE FUND v2.0 | mode=%s | leverage=%dx ===", self.mode, config.LEVERAGE)
        self.exchange.load_markets()

        # 1. futures setup
        if self.mode == "LIVE":
            if not (config.API_KEY and config.SECRET):
                log.error("LIVE mode requires API_KEY and SECRET")
                return False
            self.exchange.setup_futures(config.SYMBOLS)
            self._live_equity = self.exchange.fetch_equity()
            if not self._live_equity:
                log.error("Cannot read USDT equity from Bybit (check API key permissions / balance)")
                return False
        else:
            log.info("[1/9] Paper trading: skip futures setup")

        # 2. backtest gate
        log.info("[2/9] Backtesting EMA+RSI strategy")
        self.backtest_result = run_backtest(self.exchange, config.SYMBOLS)
        if self.backtest_result:
            save_json(config.BACKTEST_FILE, self.backtest_result)
            s = self.backtest_result["summary"]
            log.info("Backtest: trades=%d winrate=%.1f%% PF=%.2f viable=%s",
                     s["trades"], s["win_rate"], s["profit_factor"], s["viable"])
        if self.mode == "LIVE" and not (self.backtest_result and self.backtest_result["summary"]["viable"]):
            msg = "Backtest tidak viable - live mode dibatalkan"
            log.error(msg)
            self.notifier.send(f"🛑 {msg}")
            self.notifier.flush()
            return False

        # 3. RL agent (loaded in constructor)
        log.info("[3/9] RL agent ready: %s", self.rl.stats())

        # 4. data + models
        log.info("[4/9] Loading market data and training models")
        for sym in config.SYMBOLS:
            df = self.exchange.fetch_ohlcv(sym, config.TIMEFRAME, config.TRAIN_CANDLES)
            if df is None or len(df) < 100:
                log.error("No market data for %s", sym)
                return False
            self.df_cache[sym] = df
            self.models.init_symbol(sym, df)
        threading.Thread(target=self._retrain_worker, daemon=True, name="retrain").start()

        # 5. positions + persistent state
        log.info("[5/9] Restoring positions")
        self._restore_state()

        # 6. paper engine
        log.info("[6/9] Paper engine balance: $%.2f", self.paper.balance)

        # 7. correlation matrix
        log.info("[7/9] Correlation matrix")
        self.refresh_correlation()
        return True

    def start_feed(self) -> None:
        """Steps 8-9 are completed by main.py (dashboard) and this call (WebSocket)."""
        self.feed.start()
        time.sleep(3)  # warm-up
        log.info("[9/9] WebSocket connected=%s", self.feed.ws_connected)

    # --------------------------------------------------------------- persistence
    def _restore_state(self) -> None:
        state = load_json(config.STATE_FILE, {}) or {}
        if "paper_balance" in state:
            self.paper.balance = float(state["paper_balance"])
        self.closed_trades = state.get("closed_trades", [])[-300:]
        self.paused = bool(state.get("paused", False))
        for point in state.get("equity_curve", [])[-600:]:
            self.equity_curve.append(point)
        saved_mode = state.get("mode")
        saved_peak = float(state.get("peak", 0) or 0)

        saved = load_json(config.POSITIONS_FILE, {}) or {}
        for sym, pos in saved.items():
            if sym not in config.SYMBOLS:
                continue
            if pos.get("mode") != self.mode:
                log.warning("Ignoring %s position saved in %s mode", sym, pos.get("mode"))
                continue
            if self.mode == "LIVE":
                try:
                    if self.exchange.fetch_position(sym) is None:
                        log.warning("%s position no longer open on exchange, dropped", sym)
                        continue
                except Exception as exc:
                    log.error("Cannot verify %s on exchange (%s); keeping it", sym, type(exc).__name__)
            self.positions[sym] = pos
            log.info("Recovered %s %s @ %.4f", sym, pos["side"], pos["entry"])

        with self._equity_lock:
            self.peak = 0.0
        eq = self.update_equity()
        if saved_mode == self.mode and saved_peak > 0:
            with self._equity_lock:
                self.peak = max(self.peak, saved_peak)
        log.info("Equity %.2f | peak %.2f", eq, self.peak)

    def save_all(self) -> None:
        save_json(config.POSITIONS_FILE, self.positions)
        save_json(config.STATE_FILE, {
            "mode": self.mode, "paper_balance": self.paper.balance, "peak": self.peak, "paused": self.paused,
            "closed_trades": self.closed_trades[-300:], "equity_curve": list(self.equity_curve)})
        self.rl.save()

    def _save_positions(self) -> None:
        save_json(config.POSITIONS_FILE, self.positions)

    # ------------------------------------------------------------------- helpers
    def price_of(self, sym: str) -> Optional[float]:
        price = self.feed.get_price(sym, config.WS_STALE_SECONDS)
        if price is not None:
            return price
        df = self.df_cache.get(sym)
        return float(df["close"].iloc[-1]) if df is not None and len(df) else None

    def _prices(self) -> dict:
        return {s: p for s in config.SYMBOLS if (p := self.price_of(s)) is not None}

    def update_equity(self) -> float:
        if self.mode == "PAPER":
            eq = self.paper.equity(self.positions, self._prices())
        else:
            eq = self._live_equity if self._live_equity is not None else self.equity
        with self._equity_lock:
            self.equity = eq
            self.peak = max(self.peak, eq)
        return eq

    def drawdown(self) -> float:
        with self._equity_lock:
            return max(0.0, (self.peak - self.equity) / self.peak) if self.peak > 0 else 0.0

    def _event(self, kind: str, text: str) -> None:
        self.events.appendleft({"ts": time.time(), "kind": kind, "text": text})

    def _ws_giveup(self) -> None:
        self.notifier.send("⚠️ WebSocket terputus > 10x retry. Bot memakai data candle terakhir (tanpa harga real-time).")
        self._event("warn", "WebSocket gave up")

    def refresh_correlation(self) -> None:
        closes = {}
        for sym in config.SYMBOLS:
            df = self.exchange.fetch_ohlcv(sym, "1h", 200)
            if df is not None:
                closes[sym] = df["close"]
        if closes:
            self.corr = correlation_matrix(closes)

    # ----------------------------------------------------------------- analysis
    def analyze(self, sym: str) -> Optional[dict]:
        raw = self.df_cache.get(sym)
        if raw is None or len(raw) < 60:
            return None
        df = add_indicators(raw)
        last = df.iloc[-1]
        sm = self.smart.score(sym)
        trap, bias = whale_trap_score(df, sm["liq_spike"], sm["liq_bias"])
        regime = detect_regime(df)
        preds = self.models.predict(sym, raw)
        atr_mean = float(last["atr_mean"]) if not pd.isna(last["atr_mean"]) else float(last["atr"])
        vol_ma = float(last["vol_ma"]) if not pd.isna(last["vol_ma"]) else float(last["volume"])
        snap = {
            "price": float(last["close"]), "ema20": float(last["ema20"]), "ema50": float(last["ema50"]),
            "rsi": float(last["rsi"]), "atr": float(last["atr"]), "adx": float(last["adx"]),
            "atr_ratio": float(last["atr"]) / atr_mean if atr_mean > 0 else 1.0,
            "vol_ratio": float(last["volume"]) / vol_ma if vol_ma > 0 else 1.0,
            "regime": regime, "trap_score": trap, "trap_bias": bias,
            "sm": sm, "preds": preds, "ts": time.time()}
        snap["rl_state"] = encode_state(snap, sm["total"], sm["liq_spike"])
        return snap

    def decide(self, sym: str, price: float, fast: bool) -> dict:
        """Voting (+ RL on the slow loop) -> confidence -> risk -> filters."""
        snap = self.snapshots[sym]
        ens = snap["preds"].get("ensemble") or snap["preds"].get("transformer")
        v = vote({**snap, "preds": snap["preds"]}, price, self.prev_price.get(sym), ens)
        action, bonus, rl_action = v["action"], 0, None
        if fast:
            # fast entries need the micro-momentum (scalper) to confirm the direction
            if action == WAIT or v["signals"]["scalper"] != action:
                action = WAIT
        else:
            rl_action = self.rl.act(snap["rl_state"], explore=(self.mode == "PAPER"))
            action, bonus = combine_rl_vote(rl_action, v["action"])
        sm_total = snap["sm"]["total"]
        conf = confidence_score(action, snap, price, snap["preds"]["ensemble"], sm_total, bonus > 0)
        edge_pack = pick_best_edge(snap, price, snap["preds"], v, conf if action == LONG else None,
                                   conf if action == SHORT else None)
        edge = edge_pack["best"] if action in (LONG, SHORT) else {"edge": 0}
        if action in (LONG, SHORT):
            side_edge = edge_pack[action]["edge"]
            bonus_edge = max(0, min(config.PROFIT_EDGE_BONUS, (side_edge - config.MIN_PROFIT_EDGE) // 3))
            conf = int(max(0, min(100, conf + bonus_edge)))
            edge = edge_pack[action]
        risk = dynamic_risk(sym, price, snap["atr"], snap["regime"], sm_total)
        ok, reason = check_filters(action, sym, conf, snap["regime"], snap["trap_score"], sm_total,
                                   self.corr, self.positions, risk, edge.get("edge"))
        return {"action": action, "conf": conf, "risk": risk, "ok": ok, "reason": reason,
                "vote": v, "rl_action": rl_action, "profit_edge": edge}

    # ----------------------------------------------------------------- execution
    def open_position(self, sym: str, side: str, price: float, d: dict, source: str) -> bool:
        snap = self.snapshots[sym]
        equity = self.update_equity()
        used_margin = sum(p["entry"] * p["qty"] / config.LEVERAGE for p in self.positions.values())
        sl, tp = sl_tp(side, price, snap["atr"])
        qty = self.exchange.round_qty(sym, calc_qty(equity, d["risk"], price, snap["atr"], used_margin))
        if qty <= 0:
            log.info("%s: quantity below exchange minimum, skip", sym)
            self._set_decision(sym, d, "QTY TOO SMALL")
            return False
        risk_pct = qty * config.SL_ATR_MULT * snap["atr"] / equity
        meta = {"conf": d["conf"], "risk_pct": risk_pct, "rl_state": snap["rl_state"], "source": source,
                "trap": snap["trap_score"], "sm": snap["sm"]["total"],
                "lstm": snap["preds"]["lstm"], "transformer": snap["preds"]["transformer"],
                "rl_action": d["rl_action"], "vote_action": d["vote"]["action"], "sl0": sl, "atr": snap["atr"]}
        try:
            if self.mode == "PAPER":
                pos = self.paper.execute(sym, side, qty, price, sl, tp, meta)
            else:
                pos = self._execute_live(sym, side, qty, price, sl, tp, meta)
                if pos is None:
                    return False
        except Exception as exc:
            log.exception("Order failed for %s", sym)
            self.notifier.send(f"❌ Order {sym} {side} gagal: {type(exc).__name__}")
            self._event("error", f"order {sym} failed")
            return False
        self.positions[sym] = pos
        self._save_positions()
        self.update_equity()
        self._set_decision(sym, d, "OPENED")
        icon = "⚡" if source == "fast" else "🚀"
        self.notifier.send(
            f"{icon} {self.mode} {sym} {side}\nEntry: {pos['entry']:.4f} | SL: {sl:.4f} | TP: {tp:.4f}\n"
            f"Qty: {qty} | Conf: {d['conf']} | Trap: {snap['trap_score']} | SM: {snap['sm']['total']:+.2f}\n"
            f"LSTM: {self._fmt(snap['preds']['lstm'])} | TF: {self._fmt(snap['preds']['transformer'])}\n"
            f"Profit edge: {d['profit_edge'].get('edge', '–')} ({d['profit_edge'].get('grade', '–')}) • E[R] {d['profit_edge'].get('expected_r', '–')}\n"
            f"RL: {d['rl_action'] or '-'} vs Vote: {d['vote']['action']}")
        self._event("open", f"{side} {sym} @ {pos['entry']:.4f} (conf {d['conf']}, {source})")
        log.info("OPEN %s %s qty=%s entry=%.4f sl=%.4f tp=%.4f conf=%d", sym, side, qty, pos["entry"], sl, tp, d["conf"])
        return True

    @staticmethod
    def _fmt(value: Optional[float]) -> str:
        return "-" if value is None else f"{value:.4f}"

    def _execute_live(self, sym: str, side: str, qty: float, price: float, sl: float, tp: float,
                      meta: dict) -> Optional[dict]:
        order = self.exchange.open_market(sym, side, qty, sl, tp)
        time.sleep(1.5)
        ex_pos = self.exchange.fetch_position(sym)
        attached = bool(ex_pos and float((ex_pos.get("info") or {}).get("stopLoss") or 0) > 0)
        if not attached:
            # never leave a leveraged position without a protective stop
            log.error("%s: SL not found on exchange after entry, closing immediately", sym)
            self.notifier.send(f"🛑 {sym}: SL tidak terpasang di Bybit, posisi ditutup otomatis")
            try:
                self.exchange.close_market(sym, side, float(ex_pos["contracts"]) if ex_pos else qty)
            except Exception:
                log.exception("Emergency close failed for %s", sym)
                self.notifier.send(f"🚨 {sym}: GAGAL menutup posisi tanpa SL - cek Bybit SEKARANG")
            return None
        entry = float(order.get("average") or (ex_pos.get("entryPrice")) or price)
        filled = float(order.get("filled") or ex_pos.get("contracts") or qty)
        pos = {"symbol": sym, "side": side, "entry": entry, "qty": filled, "sl": sl, "tp": tp,
               "entry_fee": entry * filled * config.TAKER_FEE, "opened_at": time.time(), "mode": "LIVE"}
        pos.update(meta)
        return pos

    def close_position(self, sym: str, reason: str, trigger_price: Optional[float] = None) -> Optional[dict]:
        pos = self.positions.get(sym)
        if pos is None:
            return None
        price = trigger_price or self.price_of(sym) or pos["entry"]
        try:
            if pos["mode"] == "PAPER":
                res = self.paper.close(pos, price, reason)
            else:
                res = self._close_live(pos, reason, price)
        except Exception as exc:
            log.exception("Close failed for %s", sym)
            self.notifier.send(f"❌ Gagal menutup {sym}: {type(exc).__name__}")
            return None

        del self.positions[sym]
        self.cooldown[sym] = time.time()
        equity = self.update_equity()
        trade = {"symbol": sym, "side": pos["side"], "entry": pos["entry"], "exit": res["exit"], "qty": pos["qty"],
                 "pnl": round(res["pnl"], 4), "reason": reason, "mode": pos["mode"],
                 "opened_at": pos["opened_at"], "closed_at": time.time(), "conf": pos.get("conf")}
        self.closed_trades.append(trade)
        del self.closed_trades[:-300]

        snap = self.snapshots.get(sym)
        next_state = snap["rl_state"] if snap else pos.get("rl_state", 0)
        if "rl_state" in pos:
            self.rl.update(pos["rl_state"], pos["side"], self.rl.trade_reward(res["pnl"], equity), next_state)
        self.save_all()

        icon = "✅" if reason == "TP" else "🛑" if reason == "SL" else "🔔"
        self.notifier.send(f"{icon} {reason} {sym} {pos['side']}\nPnL: {res['pnl']:+.2f} USDT | Equity: {equity:.2f}")
        self._event("close", f"{reason} {sym} {pos['side']} PnL {res['pnl']:+.2f}")
        log.info("CLOSE %s %s pnl=%+.2f", sym, reason, res["pnl"])
        return trade

    def _close_live(self, pos: dict, reason: str, price: float) -> dict:
        sym, d = pos["symbol"], direction(pos["side"])
        ex_pos = self.exchange.fetch_position(sym)
        exit_price = price
        if ex_pos is not None:
            order = self.exchange.close_market(sym, pos["side"], float(ex_pos["contracts"]))
            exit_price = float(order.get("average") or price)
        # if the exchange already closed it (SL/TP hit) the trigger price is used as estimate
        gross = (exit_price - pos["entry"]) * pos["qty"] * d
        fees = pos.get("entry_fee", 0.0) + exit_price * pos["qty"] * config.TAKER_FEE
        eq = self.exchange.fetch_equity()
        if eq is not None:
            self._live_equity = eq
        return {"pnl": gross - fees, "exit": exit_price, "fees": fees, "reason": reason}

    def _set_decision(self, sym: str, d: dict, status: str) -> None:
        self.last_decision[sym] = {"action": d["action"], "conf": d["conf"], "status": status,
                                   "rl": d.get("rl_action"), "vote": d["vote"]["action"], "ts": time.time()}

    # ---------------------------------------------------------------- fast loop
    def manage_stop(self, sym: str, pos: dict, price: float) -> None:
        """Break-even / trailing stop. In LIVE mode the exchange stop is moved first;
        if that fails the local SL is left untouched (the original stop stays active)."""
        new_sl, reason = next_stop(pos, price)
        if new_sl is None:
            return
        if pos["mode"] == "LIVE":
            if time.time() - pos.get("sl_try", 0) < 30:
                return
            pos["sl_try"] = time.time()
            try:
                self.exchange.set_stop_loss(sym, new_sl)
            except Exception as exc:
                log.error("Failed to move SL on exchange for %s: %s", sym, type(exc).__name__)
                self.notifier.send(f"⚠️ {sym}: gagal menggeser SL ke {new_sl:.4f} ({type(exc).__name__}). SL lama tetap aktif.")
                return
        old_sl = pos["sl"]
        pos["sl"], pos["be"] = new_sl, True
        self._save_positions()
        log.info("%s SL %s: %.4f -> %.4f", sym, reason, old_sl, new_sl)
        if reason == "BREAKEVEN" or not pos.get("trail_notified"):
            pos["trail_notified"] = True
            self.notifier.send(f"🔒 {sym} {pos['side']}: SL digeser ke {new_sl:.4f} ({reason}, harga {price:.4f})")
        self._event("stop", f"{reason} {sym} SL → {new_sl:.4f}")

    def set_paused(self, paused: bool) -> None:
        if paused == self.paused:
            return
        self.paused = paused
        self.save_all()
        self.notifier.send("⏸ Entry baru dijeda (posisi terbuka tetap dikelola)" if paused else "▶️ Entry baru dilanjutkan")
        self._event("warn", "Entry baru DIJEDA" if paused else "Entry baru dilanjutkan")
        self.sync_dashboard()

    def fast_tick(self) -> None:
        for sym in config.SYMBOLS:
            price = self.price_of(sym)
            if price is None:
                continue
            pos = self.positions.get(sym)
            if pos and pos["mode"] == "PAPER":
                hit = self.paper.check_sl_tp(pos, price)
                if hit:
                    self.close_position(sym, hit, pos["sl"] if hit == "SL" else pos["tp"])
                else:
                    self.manage_stop(sym, pos, price)
            elif pos:
                self.manage_stop(sym, pos, price)
            elif (not self.halted and not self.paused and sym in self.snapshots
                  and time.time() - self.cooldown.get(sym, 0) > config.TRADE_COOLDOWN):
                d = self.decide(sym, price, fast=True)
                if d["ok"]:
                    self.open_position(sym, d["action"], price, d, "fast")
            self.prev_price[sym] = price
        self.update_equity()
        if self.drawdown() >= config.MAX_DRAWDOWN:
            self.circuit_breaker()
        now = time.time()
        if now - self._last_curve_ts >= 30:
            self._last_curve_ts = now
            self.equity_curve.append([int(now), round(self.equity, 2)])
        if now - self._last_sync >= 5:
            self._last_sync = now
            self.sync_dashboard()

    # ---------------------------------------------------------------- slow loop
    def slow_tick(self) -> None:
        log.info("--- slow loop ---")
        self.last_slow = time.time()
        self.refresh_correlation()
        self.check_live_positions()
        self.refresh_ohlcv()

        for sym in config.SYMBOLS:
            if self.stop_event.is_set():
                return
            try:
                snap = self.analyze(sym)
                if snap is None:
                    continue
                self.snapshots[sym] = snap
                if sym in self.positions or self.halted or self.paused:
                    continue
                price = self.price_of(sym) or snap["price"]
                d = self.decide(sym, price, fast=False)
                if d["ok"]:
                    self.open_position(sym, d["action"], price, d, "slow")
                else:
                    self._on_skip(sym, d, snap)
            except Exception:
                log.exception("Slow loop error on %s", sym)  # one symbol must never crash the bot

        if self.mode == "LIVE":
            eq = self.exchange.fetch_equity()
            if eq is not None:
                self._live_equity = eq
        self.update_equity()
        self.equity_curve.append([int(time.time()), round(self.equity, 2)])
        if self.drawdown() >= config.MAX_DRAWDOWN:
            self.circuit_breaker()
        self.save_all()
        self.sync_dashboard()

    def _on_skip(self, sym: str, d: dict, snap: dict) -> None:
        self._set_decision(sym, d, f"SKIP: {d['reason']}")
        if d["action"] == WAIT:
            direction_hint = LONG if snap["ema20"] > snap["ema50"] else SHORT
            hypo = confidence_score(direction_hint, snap, snap["price"], snap["preds"]["ensemble"],
                                    snap["sm"]["total"], False)
            self.rl.update(snap["rl_state"], WAIT, self.rl.wait_reward(hypo), snap["rl_state"])
            return
        reason = d["reason"]
        if reason.startswith(NOTIFY_SKIP_REASONS):
            last = self._last_skip.get(sym)
            if not last or last[0] != reason or time.time() - last[1] > 3600:
                self._last_skip[sym] = (reason, time.time())
                self.notifier.send(f"⚠️ {sym} {d['action']} dilewati: {reason}")
        log.info("SKIP %s %s: %s (conf %d)", sym, d["action"], d["reason"], d["conf"])

    def refresh_ohlcv(self) -> None:
        for sym in config.SYMBOLS:
            fresh = self.exchange.fetch_ohlcv(sym, config.TIMEFRAME, config.OHLCV_LIMIT)
            if fresh is None:
                continue
            merged = pd.concat([self.df_cache[sym], fresh]).drop_duplicates("timestamp", keep="last")
            self.df_cache[sym] = merged.sort_values("timestamp").tail(config.TRAIN_CANDLES).reset_index(drop=True)

    def check_live_positions(self) -> None:
        if self.mode != "LIVE":
            return
        for sym, pos in list(self.positions.items()):
            if pos["mode"] != "LIVE":
                continue
            price = self.price_of(sym)
            try:
                still_open = self.exchange.fetch_position(sym) is not None
            except Exception as exc:
                log.warning("fetch_position %s failed: %s", sym, type(exc).__name__)
                continue
            d = direction(pos["side"])
            hit = None
            if price is not None:
                hit = "SL" if (price - pos["sl"]) * d <= 0 else "TP" if (price - pos["tp"]) * d >= 0 else None
            if not still_open:
                self.close_position(sym, hit or "EXCHANGE", pos["sl"] if hit == "SL" else pos["tp"] if hit == "TP" else price)
            elif hit:
                self.close_position(sym, hit, price)

    # ------------------------------------------------------------ safety/shutdown
    def circuit_breaker(self) -> None:
        if self.halted:
            return
        self.halted = True
        dd = self.drawdown() * 100
        log.error("CIRCUIT BREAKER: drawdown %.1f%%", dd)
        for sym in list(self.positions):
            self.close_position(sym, "CIRCUIT BREAKER")
        self.save_all()
        self.notifier.send(f"🛑 SYSTEM STOP — MAX DRAWDOWN {config.MAX_DRAWDOWN * 100:.0f}% (DD {dd:.1f}%)")
        self._event("error", "Circuit breaker triggered")
        self.exit_code = 3
        self.stop_event.set()

    def shutdown(self, reason: str = "manual") -> None:
        log.info("Shutting down (%s)", reason)
        self.feed.stop()
        self.sync_dashboard()
        self.save_all()
        if reason == "manual":
            self.notifier.send("🛑 Bot dihentikan manual (Ctrl+C)")
        self.notifier.flush()

    def _retrain_worker(self) -> None:
        while not self.stop_event.wait(300):
            for sym in config.SYMBOLS:
                if not self.models.retrain_due(sym):
                    continue
                try:
                    ok = self.models.retrain(sym, self.df_cache[sym].copy())
                    self.notifier.send(f"{'✅' if ok else '⚠️'} Retrain model {sym}: {'berhasil' if ok else 'gagal'}")
                except Exception:
                    log.exception("Retrain %s failed", sym)

    # ------------------------------------------------------------------ dashboard
    def stats(self) -> dict:
        trades = self.closed_trades
        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [-t["pnl"] for t in trades if t["pnl"] <= 0]
        by_symbol: dict = {}
        for t in trades:
            s = by_symbol.setdefault(t["symbol"], {"trades": 0, "wins": 0, "pnl": 0.0})
            s["trades"] += 1
            s["wins"] += 1 if t["pnl"] > 0 else 0
            s["pnl"] = round(s["pnl"] + t["pnl"], 2)
        return {"trades": len(trades), "wins": len(wins), "by_symbol": by_symbol,
                "best": round(max((t["pnl"] for t in trades), default=0.0), 2),
                "worst": round(min((t["pnl"] for t in trades), default=0.0), 2),
                "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
                "avg_loss": round(-sum(losses) / len(losses), 2) if losses else 0.0,
                "win_rate": round(100 * len(wins) / len(trades), 1) if trades else 0.0,
                "profit_factor": round(sum(wins) / sum(losses), 2) if losses and sum(losses) > 0 else (99.0 if wins else 0.0),
                "total_pnl": round(sum(t["pnl"] for t in trades), 2)}

    def sync_dashboard(self) -> None:
        prices = self._prices()
        positions = []
        for sym, p in self.positions.items():
            price = prices.get(sym, p["entry"])
            upnl = PaperEngine.unrealized(p, price)
            r = r_multiple(p, price)
            span = p["tp"] - p["sl"]
            positions.append({**{k: p.get(k) for k in ("symbol", "side", "entry", "qty", "sl", "tp", "conf", "mode", "source", "sl0")},
                              "price": price, "upnl": round(upnl, 2),
                              "upnl_pct": round(upnl / (p["entry"] * p["qty"] / config.LEVERAGE) * 100, 2),
                              "r": None if r is None else round(r, 2), "be": bool(p.get("be")),
                              "progress": round(min(100, max(0, (price - p["sl"]) / span * 100)), 1) if span else None,
                              "opened_at": p["opened_at"]})
        analysis = {}
        for sym, s in self.snapshots.items():
            edges = pick_best_edge(s, s["price"], s["preds"])
            analysis[sym] = {"regime": s["regime"], "rsi": round(s["rsi"], 1), "trap": s["trap_score"],
                             "trap_bias": s["trap_bias"], "sm": s["sm"]["total"],
                             "ema_trend": "UP" if s["ema20"] > s["ema50"] else "DOWN",
                             "lstm": s["preds"]["lstm"], "transformer": s["preds"]["transformer"],
                             "ai_dir": s["preds"].get("ai_direction"), "ai_conf": s["preds"].get("ai_confidence"),
                             "profit_edge": edges["best"]["edge"], "edge_grade": edges["best"]["grade"],
                             "win_prob": edges["best"]["win_prob"], "expected_r": edges["best"]["expected_r"],
                             "decision": self.last_decision.get(sym)}
        data = {
            "mode": self.mode, "leverage": config.LEVERAGE, "halted": self.halted, "paused": self.paused,
            "breakeven_r": config.BREAKEVEN_R, "trail_start_r": config.TRAIL_START_R,
            "balance": round(self.paper.balance if self.mode == "PAPER" else (self._live_equity or 0), 2),
            "equity": round(self.equity, 2), "peak": round(self.peak, 2),
            "drawdown": round(self.drawdown() * 100, 2), "max_drawdown": config.MAX_DRAWDOWN * 100,
            "total_risk": round(sum(p.get("risk_pct", 0) for p in self.positions.values()) * 100, 2),
            "max_total_risk": config.MAX_TOTAL_RISK * 100,
            "positions": positions, "analysis": analysis, "ws": self.feed.snapshot(),
            "rl": self.rl.stats(), "stats": self.stats(), "equity_curve": list(self.equity_curve),
            "trades": self.closed_trades[-150:][::-1], "events": list(self.events),
            "uptime": int(time.time() - self.started_at), "last_slow": self.last_slow,
            "next_slow": (self.last_slow + config.SLOW_INTERVAL) if self.last_slow else None,
            "updated": time.time(), "symbols": config.SYMBOLS}
        with self.dashboard_lock:
            self.dashboard_data = data

    # ------------------------------------------------------------------ main loop
    def run(self) -> int:
        self.notifier.send(
            f"🤖 AI Hedge Fund v2.0 aktif\nMode: {self.mode} | Leverage: {config.LEVERAGE}x\n"
            f"WebSocket: {'OK' if self.feed.ws_connected else 'belum terhubung'}")
        next_slow = 0.0
        while not self.stop_event.is_set():
            started = time.time()
            try:
                if started >= next_slow:
                    self.slow_tick()
                    next_slow = time.time() + config.SLOW_INTERVAL
                self.fast_tick()
            except Exception:
                log.exception("Main loop error")
                self.notifier.send("⚠️ Error di main loop (lihat log), bot lanjut")
            self.stop_event.wait(max(0.0, config.FAST_INTERVAL - (time.time() - started)))
        return self.exit_code
