"""Flask dashboard (read-only) - mobile friendly, polled by the single-page UI."""
import hmac
import logging
import threading
import time
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, make_response, request, send_from_directory

from . import config
from .alerts import LABELS as ALERT_LABELS
from .analysis import AnalysisService
from .exchange import Exchange
from .indicators import add_indicators
from .lab import STRATEGIES, LabService
from .scanner import MarketScanner
from .storage import load_json

log = logging.getLogger("dashboard")
WEB_DIR = Path(__file__).resolve().parent / "web"
TIMEFRAMES = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
CACHE_TTL = 8  # seconds; keeps the dashboard from hammering the exchange API


class ChartFeed:
    """Candles (incl. the forming one) + EMA20/EMA50 for the dashboard, with its own
    exchange client so it never competes with the trading loop."""

    def __init__(self) -> None:
        self.exchange = Exchange()
        self._lock = threading.Lock()
        self._cache: dict = {}

    def get(self, symbol: str, tf: str):
        key = (symbol, tf)
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.time() - hit[0] < CACHE_TTL:
                return hit[1]
            df = self.exchange.fetch_ohlcv(symbol, tf, 300, drop_open=False)
            if df is None or len(df) < 5:
                return hit[1] if hit else None
            d = add_indicators(df)
            rows = [{"t": int(r.timestamp) // 1000, "o": r.open, "h": r.high, "l": r.low, "c": r.close,
                     "v": r.volume, "e20": None if pd.isna(r.ema20) else r.ema20,
                     "e50": None if pd.isna(r.ema50) else r.ema50}
                    for r in d.itertuples(index=False)]
            data = {"candles": rows, "step": TIMEFRAMES[tf]}
            self._cache[key] = (time.time(), data)
            return data


def create_app(bot, scanner, analysis, lab) -> Flask:
    app = Flask(__name__, static_folder=None)
    chart_feed = ChartFeed()

    def authorized() -> bool:
        if not config.DASHBOARD_TOKEN:
            return True
        supplied = request.args.get("token") or request.cookies.get("token") or request.headers.get("X-Token", "")
        return hmac.compare_digest(supplied.encode(), config.DASHBOARD_TOKEN.encode())

    @app.before_request
    def guard():
        if not authorized():
            return jsonify(error="unauthorized"), 401
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            # state-changing calls must come from the dashboard page itself (blocks cross-site requests)
            origin = request.headers.get("Origin")
            if request.headers.get("X-Requested-With") != "dashboard" or (origin and origin.split("://", 1)[-1] != request.host):
                return jsonify(error="forbidden"), 403
        return None

    @app.after_request
    def harden(resp):
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    @app.get("/")
    def index():
        resp = make_response(send_from_directory(WEB_DIR, "index.html"))
        if config.DASHBOARD_TOKEN and request.args.get("token"):
            resp.set_cookie("token", request.args["token"], httponly=True, samesite="Strict")
        return resp

    @app.get("/api")
    def api():
        with bot.dashboard_lock:
            return jsonify(bot.dashboard_data)

    @app.get("/backtest")
    def backtest():
        return jsonify(bot.backtest_result or load_json(config.BACKTEST_FILE, {}))

    @app.get("/price-stream")
    def price_stream():
        return jsonify(bot.feed.snapshot())

    @app.get("/smart-money")
    def smart_money():
        return jsonify({"market": bot.smart.snapshot(),
                        "symbols": {s: snap["sm"] for s, snap in list(bot.snapshots.items())}})

    @app.get("/strategies")
    def strategies():
        return jsonify({"weights": {**config.WEIGHTS, "transformer": config.TRANSFORMER_WEIGHT,
                                    "ai_ensemble": config.AI_ENSEMBLE_WEIGHT},
                        "min_vote_weight": config.MIN_VOTE_WEIGHT, "min_confidence": config.MIN_CONFIDENCE,
                        "min_profit_edge": config.MIN_PROFIT_EDGE})

    @app.get("/scanner")
    def scanner_data():
        return jsonify(scanner.snapshot())

    @app.get("/analysis")
    def analysis_data():
        symbol = request.args.get("symbol", "")
        if symbol not in config.SYMBOLS and not scanner.has(symbol):
            return jsonify(error="invalid symbol"), 400
        data = analysis.get(symbol)
        if data is None:
            return jsonify(error="market data unavailable"), 503
        snap = bot.snapshots.get(symbol)
        data = {**data, "bot": {"regime": snap["regime"], "trap": snap["trap_score"], "sm": snap["sm"]["total"],
                                "decision": bot.last_decision.get(symbol)} if snap else None,
                "equity": bot.equity, "leverage": config.LEVERAGE, "base_risk": config.BASE_RISK * 100}
        return jsonify(data)

    @app.get("/alerts")
    def alerts_list():
        return jsonify({"alerts": scanner.alerts.list(), "types": ALERT_LABELS})

    @app.post("/alerts")
    def alerts_add():
        body = request.get_json(silent=True) or {}
        symbol = str(body.get("symbol", ""))
        alert, err = scanner.alerts.add(symbol, str(body.get("type", "")), body.get("value"), bool(body.get("repeat")),
                                        str(body.get("note", "")), scanner.row(symbol))
        return (jsonify(alert), 201) if alert else (jsonify(error=err), 400)

    @app.delete("/alerts/<alert_id>")
    def alerts_delete(alert_id):
        return (jsonify(ok=True), 200) if scanner.alerts.remove(alert_id) else (jsonify(error="not found"), 404)

    @app.post("/control/pause")
    def control_pause():
        body = request.get_json(silent=True) or {}
        bot.set_paused(bool(body.get("paused")))
        return jsonify(paused=bot.paused)

    @app.get("/lab/meta")
    def lab_meta():
        return jsonify({"strategies": [{"key": k, **m} for k, m in STRATEGIES.items()],
                        "timeframes": list(TIMEFRAMES), "symbols": sorted(r["symbol"] for r in scanner.snapshot()["rows"])})

    @app.get("/lab")
    def lab_run():
        a = request.args
        symbol, tf, strategy = a.get("symbol", ""), a.get("tf", "1h"), a.get("strategy", "ema_cross")
        compare = a.get("compare") == "1"
        if symbol not in config.SYMBOLS and not scanner.has(symbol):
            return jsonify(error="invalid symbol"), 400
        if tf not in ("15m", "1h", "4h") or (not compare and strategy not in STRATEGIES):
            return jsonify(error="invalid timeframe or strategy"), 400
        try:
            sl, tp = float(a.get("sl", 1.5)), float(a.get("tp", 3.0))
            n, hold = int(a.get("n", 2000)), int(a.get("hold", 48))
        except ValueError:
            return jsonify(error="invalid number"), 400
        side = a.get("side", "both")
        if not (0.3 <= sl <= 6 and 0.3 <= tp <= 12 and 500 <= n <= 5000 and 0 <= hold <= 500 and side in ("both", "long", "short")):
            return jsonify(error="parameter out of range"), 400
        params = {k[2:]: v for k, v in a.items() if k.startswith("p_")}
        result = lab.run(symbol, tf, strategy, params, sl, tp, n, hold, side, compare)
        return (jsonify(result), 200) if result else (jsonify(error="not enough market data"), 503)

    @app.get("/vendor/<path:name>")
    def vendor(name):
        return send_from_directory(WEB_DIR / "vendor", name)

    @app.get("/candles")
    def candles():
        symbol = request.args.get("symbol", config.SYMBOLS[0])
        tf = request.args.get("tf", config.TIMEFRAME)
        if (symbol not in config.SYMBOLS and not scanner.has(symbol)) or tf not in TIMEFRAMES:
            return jsonify(error="invalid symbol or timeframe"), 400
        data = chart_feed.get(symbol, tf)
        if data is None:
            return jsonify(error="market data unavailable"), 503
        return jsonify({**data, "symbol": symbol, "tf": tf,
                        "positions": [p for s, p in list(bot.positions.items()) if s == symbol],
                        "trades": [t for t in list(bot.closed_trades) if t["symbol"] == symbol][-40:]})

    return app


def start_dashboard(bot) -> MarketScanner:
    scanner = MarketScanner(bot.notifier)
    scanner.start()
    app = create_app(bot, scanner, AnalysisService(scanner), LabService())

    def serve():
        try:
            app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, threaded=True, use_reloader=False)
        except OSError as exc:
            log.error("Dashboard failed to start: %s", exc)

    threading.Thread(target=serve, daemon=True, name="dashboard").start()
    log.info("[8/9] Dashboard on http://%s:%d | scanner monitoring top %d pairs", config.DASHBOARD_HOST,
             config.DASHBOARD_PORT, config.SCANNER_TOP_N)
    return scanner
