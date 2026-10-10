import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import config  # noqa: E402
from bot.backtest import backtest_symbol  # noqa: E402
from bot.indicators import add_indicators, detect_regime, whale_trap_score  # noqa: E402
from bot.paper import PaperEngine  # noqa: E402
from bot.risk import calc_qty, check_filters, confidence_score, dynamic_risk, sl_tp  # noqa: E402
from bot.rl import DIMS, N_STATES, RLAgent, encode_state  # noqa: E402
from bot.strategies import LONG, SHORT, WAIT, combine_rl_vote, vote  # noqa: E402


def make_df(n=300, drift=0.0005, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.004, n))
    open_ = np.concatenate([[100], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    return pd.DataFrame({"timestamp": np.arange(n), "open": open_, "high": high, "low": low,
                         "close": close, "volume": rng.uniform(100, 200, n)})


def snap(**kw):
    base = {"price": 100.0, "ema20": 101.0, "ema50": 100.0, "rsi": 55.0, "atr": 1.0, "atr_ratio": 1.0,
            "vol_ratio": 1.0, "trap_score": 0, "trap_bias": None, "regime": "TRENDING_UP"}
    base.update(kw)
    return base


def test_state_space_is_5184():
    assert N_STATES == 5184 and len(DIMS) == 8
    s = snap()
    idx = encode_state(s, 0.0, False)
    assert 0 <= idx < N_STATES
    assert 0 <= encode_state(snap(rsi=80, regime="VOLATILE", ema20=90), 0.9, True) < N_STATES


def test_vote_trend_and_transformer_long():
    v = vote(snap(), 100.0, None, 101.0)
    assert v["action"] == LONG and not v["flipped"]


def test_vote_whale_trap_flips_signals():
    v = vote(snap(trap_score=3, trap_bias="bull_trap"), 100.0, None, 101.0)
    assert v["flipped"] and v["signals"]["whale"] == SHORT and v["action"] == SHORT


def test_vote_weighted_majority():
    v = vote(snap(rsi=50), 100.0, None, None)  # only trend votes: 0.30 meets the minimum
    assert v["action"] == LONG
    v = vote(snap(ema20=99.0), 100.0, 99.0, 101.0)  # trend SHORT .3 vs scalper LONG .2 + tf LONG .2
    assert v["action"] == LONG


def test_combine_rl_vote_matrix():
    assert combine_rl_vote(LONG, LONG) == (LONG, 10)
    assert combine_rl_vote(LONG, SHORT) == (SHORT, 0)
    assert combine_rl_vote(WAIT, LONG) == (LONG, 0)
    assert combine_rl_vote(SHORT, WAIT) == (SHORT, 0)
    assert combine_rl_vote(WAIT, WAIT) == (WAIT, 0)


def test_confidence_bounds_and_penalties():
    good = confidence_score(LONG, snap(vol_ratio=1.5), 100.0, 101.0, 0.5, True)
    assert 60 <= good <= 100
    assert confidence_score(WAIT, snap(), 100.0, None, 0, False) == 0
    trapped = confidence_score(LONG, snap(trap_score=4, regime="VOLATILE"), 100.0, 101.0, 0.5, True)
    assert trapped < 30
    short = confidence_score(SHORT, snap(ema20=99.0, rsi=45), 100.0, 99.0, -0.5, False)
    assert short >= 60


def test_dynamic_risk_multipliers():
    assert dynamic_risk("BTC/USDT", 100, 1, "TRENDING_UP", 0.0) == pytest.approx(0.01)
    assert dynamic_risk("BTC/USDT", 100, 1, "VOLATILE", 0.0) == 0.0
    assert dynamic_risk("BTC/USDT", 100, 1, "SIDEWAYS", 0.0) == pytest.approx(0.003)
    assert dynamic_risk("HYPE/USDT", 100, 1, "TRENDING_UP", 0.0) == pytest.approx(0.005)
    assert dynamic_risk("BTC/USDT", 100, 6, "TRENDING_UP", -0.6) == pytest.approx(0.0025)


def test_sl_tp_and_qty():
    sl, tp = sl_tp(LONG, 100.0, 2.0)
    assert (sl, tp) == (97.0, 106.0)
    sl, tp = sl_tp(SHORT, 100.0, 2.0)
    assert (sl, tp) == (103.0, 94.0)
    qty = calc_qty(10_000, 0.01, 100.0, 2.0)
    assert qty == pytest.approx(100 / 3.0)  # risk $100 over a $3 stop
    assert calc_qty(100, 0.01, 100.0, 0.001) * 100 <= 100 * 0.95 * config.LEVERAGE + 1e-9  # leverage cap


def test_profit_edge_scoring():
    from bot.profit_edge import pick_best_edge, scanner_row_edge

    s = snap(vol_ratio=1.5, regime="TRENDING_UP")
    preds = {"ensemble": 101.0, "lstm_delta_pct": 0.2, "transformer_delta_pct": 0.15,
             "ai_direction": "LONG", "ai_confidence": 0.8}
    pack = pick_best_edge(s, 100.0, preds)
    assert pack["LONG"]["edge"] > pack["SHORT"]["edge"]
    assert pack["LONG"]["hot"] or pack["LONG"]["edge"] >= 50
    row = {"signal": "BULLISH", "rsi": 52, "adx": 28, "trend": "UP", "vol_ratio": 2.1, "chg1h": 1.2, "funding": -0.01}
    se = scanner_row_edge(row)
    assert se["side"] == "LONG" and se["edge"] >= 55


def test_filters():
    empty = pd.DataFrame()
    ok, _ = check_filters(LONG, "BTC/USDT", 70, "TRENDING_UP", 0, 0.0, empty, {}, 0.01)
    assert ok
    assert check_filters(LONG, "BTC/USDT", 59, "TRENDING_UP", 0, 0.0, empty, {}, 0.01) == (False, "CONFIDENCE 59")
    assert not check_filters(LONG, "BTC/USDT", 90, "VOLATILE", 0, 0.0, empty, {}, 0.01)[0]
    assert not check_filters(LONG, "BTC/USDT", 90, "TRENDING_UP", 4, 0.0, empty, {}, 0.01)[0]
    assert not check_filters(LONG, "BTC/USDT", 90, "TRENDING_UP", 0, -0.7, empty, {}, 0.01)[0]
    positions = {s: {"side": LONG, "risk_pct": 0.01} for s in ("A", "B", "C", "D", "E")}
    assert check_filters(LONG, "BTC/USDT", 90, "TRENDING_UP", 0, 0.0, empty, positions, 0.01)[1] == "MAX TOTAL RISK"
    corr = pd.DataFrame([[1, 0.9], [0.9, 1]], index=["BTC/USDT", "ETH/USDT"], columns=["BTC/USDT", "ETH/USDT"])
    pos = {"ETH/USDT": {"side": LONG, "risk_pct": 0.01}}
    assert check_filters(LONG, "BTC/USDT", 90, "TRENDING_UP", 0, 0.0, corr, pos, 0.01)[1].startswith("KORELASI")
    assert check_filters(SHORT, "BTC/USDT", 90, "TRENDING_UP", 0, 0.0, corr, pos, 0.01)[0]  # opposite side hedges
    assert check_filters(LONG, "BTC/USDT", 90, "TRENDING_UP", 0, 0.0, empty, {}, 0.01, 40)[1] == "PROFIT EDGE 40"


def test_paper_engine_pnl_and_sltp():
    p = PaperEngine(10_000)
    pos = p.execute("BTC/USDT", LONG, 1.0, 100.0, 97.0, 106.0)
    assert pos["entry"] == pytest.approx(100.1)
    assert p.check_sl_tp(pos, 98.0) is None
    assert p.check_sl_tp(pos, 96.9) == "SL"
    assert p.check_sl_tp(pos, 106.0) == "TP"
    res = p.close(pos, 106.0, "TP")
    assert res["pnl"] > 0
    assert p.balance == pytest.approx(10_000 + res["pnl"])
    short = p.execute("ETH/USDT", SHORT, 2.0, 50.0, 53.0, 44.0)
    assert p.check_sl_tp(short, 53.5) == "SL" and p.check_sl_tp(short, 43.0) == "TP"


def test_rl_learning_and_rewards(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "QTABLE_FILE", tmp_path / "q.json")
    agent = RLAgent()
    for _ in range(50):
        agent.update(10, LONG, 1.0, 10)
    assert agent.act(10, explore=False) == LONG
    assert agent.act(11, explore=False) == WAIT  # untrained state -> WAIT
    assert agent.epsilon >= config.RL_EPSILON_MIN
    assert RLAgent.trade_reward(1000, 10_000) == 2.0  # capped
    assert RLAgent.trade_reward(-1000, 10_000) == -3.0
    assert RLAgent.trade_reward(50, 10_000) == pytest.approx(0.5)
    assert RLAgent.wait_reward(50) == 0.1 and RLAgent.wait_reward(70) == -0.1
    agent.save()
    assert RLAgent().act(10, explore=False) == LONG


def test_indicators_regime_and_trap():
    df = add_indicators(make_df())
    assert not df[["ema20", "ema50", "rsi", "atr", "adx"]].iloc[-1].isna().any()
    assert detect_regime(df) in ("TRENDING_UP", "TRENDING_DOWN", "SIDEWAYS", "VOLATILE")
    assert detect_regime(df.iloc[:20]) == "UNKNOWN"
    # fake breakout above the 20-candle high with a long upper wick on high volume
    fake = df.copy()
    prior_high = fake["high"].iloc[-21:-1].max()
    i = fake.index[-1]
    fake.loc[i, ["open", "close"]] = [prior_high * 0.995, prior_high * 0.994]
    fake.loc[i, "high"] = prior_high * 1.02
    fake.loc[i, "low"] = prior_high * 0.993
    fake.loc[i, "volume"] = fake["vol_ma"].iloc[-1] * 3
    score, bias = whale_trap_score(fake)
    assert score >= 4 and bias == "bull_trap"


def test_backtest_runs():
    res = backtest_symbol(make_df(500, seed=3))
    assert {"trades", "win_rate", "profit_factor", "sharpe", "max_dd_pct"} <= set(res)
    assert 0 <= res["win_rate"] <= 100


def test_scanner_classify_and_refresh(monkeypatch):
    from bot.scanner import MarketScanner, classify

    assert classify(True, 75, 30) == "OVERBOUGHT"
    assert classify(False, 25, 30) == "OVERSOLD"
    assert classify(True, 55, 25) == "BULLISH"
    assert classify(False, 45, 25) == "BEARISH"
    assert classify(True, 55, 10) == "NEUTRAL"

    class FakeEx:
        def fetch_tickers(self, params=None):
            def t(sym, vol, chg):
                return {"symbol": sym, "last": 10.0, "percentage": chg, "quoteVolume": vol, "high": 12.0, "low": 8.0,
                        "info": {"fundingRate": "0.0001", "openInterestValue": "1000"}}
            return {"AAA/USDT:USDT": t("AAA/USDT:USDT", 5e6, 3.0), "BTC/USDT:USDT": t("BTC/USDT:USDT", 9e9, -1.0),
                    "AAA/USDT:USDT-261225": t("AAA/USDT:USDT-261225", 1e9, 0.0), "AAA/USDC:USDC": t("AAA/USDC:USDC", 1e9, 0.0)}

    scanner = MarketScanner.__new__(MarketScanner)
    import threading
    from collections import deque
    scanner._lock = threading.Lock()
    scanner.rows, scanner.fng, scanner.updated, scanner.analysis_updated = {}, None, 0.0, 0.0
    scanner.events, scanner._event_id, scanner._cool, scanner._primed, scanner.notifier = deque(maxlen=10), 0, {}, False, None
    scanner.alerts = type("A", (), {"check": lambda self, rows: None})()
    scanner.exchange = type("E", (), {"ex": FakeEx()})()
    monkeypatch.setattr(config, "SCANNER_TOP_N", 5)
    scanner._refresh_tickers()
    snap = scanner.snapshot()
    assert sorted(r["symbol"] for r in snap["rows"]) == ["AAA/USDT", "BTC/USDT"]  # dated futures / USDC excluded
    btc = next(r for r in snap["rows"] if r["symbol"] == "BTC/USDT")
    assert btc["in_bot"] and btc["range_pos"] == 50.0 and btc["funding"] == pytest.approx(0.01)
    assert scanner.has("AAA/USDT") and not scanner.has("ZZZ/USDT")


def test_next_stop_breakeven_and_trailing(monkeypatch):
    from bot.risk import next_stop, r_multiple

    monkeypatch.setattr(config, "BREAKEVEN_R", 1.0)
    monkeypatch.setattr(config, "TRAIL_START_R", 0.0)
    pos = {"side": LONG, "entry": 100.0, "sl": 97.0, "sl0": 97.0, "atr": 2.0}
    assert r_multiple(pos, 103.0) == pytest.approx(1.0)
    assert next_stop(pos, 102.0) == (None, "")                      # < 1R: nothing
    sl, why = next_stop(pos, 103.0)
    assert why == "BREAKEVEN" and sl > 100.0                         # covers fees, never below entry
    pos.update(sl=sl, be=True)
    assert next_stop(pos, 103.5) == (None, "")                      # already moved
    short = {"side": SHORT, "entry": 100.0, "sl": 103.0, "sl0": 103.0, "atr": 2.0}
    sl, why = next_stop(short, 97.0)
    assert why == "BREAKEVEN" and sl < 100.0

    monkeypatch.setattr(config, "TRAIL_START_R", 1.5)
    monkeypatch.setattr(config, "TRAIL_ATR_MULT", 1.0)
    pos = {"side": LONG, "entry": 100.0, "sl": 97.0, "sl0": 97.0, "atr": 2.0}
    sl, why = next_stop(pos, 110.0)
    assert why == "TRAIL" and sl == pytest.approx(108.0)
    pos.update(sl=sl, be=True)
    assert next_stop(pos, 108.2) == (None, "")                      # step < 0.25 ATR: no API chatter
    assert next_stop(pos, 107.0) == (None, "")                      # price pullback never loosens the stop


def test_alerts_lifecycle(tmp_path, monkeypatch):
    from bot.alerts import AlertManager

    monkeypatch.setattr(config, "ALERTS_FILE", tmp_path / "alerts.json")
    sent, events = [], []
    notifier = type("N", (), {"send": lambda self, t: sent.append(t)})()
    mgr = AlertManager(notifier, lambda *a: events.append(a))
    row = {"symbol": "BTC/USDT", "price": 100.0, "rsi": 50.0, "chg24": 1.0, "vol_ratio": 1.0, "signal": "NEUTRAL"}
    assert mgr.add("BTC/USDT", "price_above", 90, False, "", row)[1]            # already above -> rejected
    assert mgr.add("BTC/USDT", "bogus", 1, False, "", row)[1]
    assert mgr.add("ZZZ/USDT", "price_above", 1, False, "", None)[1]
    a, err = mgr.add("BTC/USDT", "price_above", 105, False, "tes", row)
    assert a and not err
    mgr.check({"BTC/USDT": row})
    assert not sent
    mgr.check({"BTC/USDT": {**row, "price": 106.0}})
    assert len(sent) == 1 and events[0][0] == "ALERT"
    mgr.check({"BTC/USDT": {**row, "price": 107.0}})
    assert len(sent) == 1                                                       # one-shot
    assert AlertManager(notifier, lambda *a: None).list()[0]["active"] is False  # persisted
    assert mgr.remove(a["id"]) and not mgr.list()


def test_analysis_build_and_levels():
    from bot.analysis import AnalysisService, nearest, swing_levels

    frames = {tf: make_df(220, drift=0.0015, seed=i) for i, tf in enumerate(["15m", "1h", "4h"])}
    out = AnalysisService.build("TEST/USDT", frames, {"price": float(frames["15m"]["close"].iloc[-1]), "funding": 0.01,
                                                      "chg24": 2.0, "turnover": 1e8})
    assert out["verdict"] in ("LONG", "SHORT", "WAIT") and set(out["quality"]) == {"LONG", "SHORT"}
    assert 0 <= out["quality"]["LONG"] <= 100 and -3 <= out["mtf"] <= 3
    p = out["plans"]["swing"]
    assert p["long"]["sl"] < p["long"]["entry"] < p["long"]["tp1"] < p["long"]["tp2"]
    assert p["short"]["tp2"] < p["short"]["tp1"] < p["short"]["entry"] < p["short"]["sl"]
    lv = nearest(swing_levels(frames["1h"], out["views"]["1h"]["atr"]), out["price"])
    assert all(l["price"] > out["price"] for l in lv["resistance"]) and all(l["price"] < out["price"] for l in lv["support"])


def test_lab_simulation_exact_exits():
    from bot import lab

    n = 260
    close = np.full(n, 100.0)
    df = pd.DataFrame({"timestamp": np.arange(n) * 3_600_000, "open": close, "high": close + 1.0, "low": close - 1.0,
                       "close": close, "volume": 1.0})
    sig = pd.Series(0, index=df.index)
    sig.iloc[230] = 1
    df.loc[235, "high"] = 110.0                                    # TP touched at bar 235
    trades = lab.simulate(df, sig, 1.5, 3.0, 0, "both")
    assert len(trades) == 1 and trades[0]["reason"] == "TP" and trades[0]["exit_i"] == 235
    assert trades[0]["ret"] > 0
    df2 = df.copy()
    df2.loc[233, "low"] = 90.0                                     # SL touched first, and SL wins ties inside one bar
    df2.loc[233, "high"] = 120.0
    t2 = lab.simulate(df2, sig, 1.5, 3.0, 0, "both")
    assert t2[0]["reason"] == "SL" and t2[0]["ret"] < 0
    assert lab.simulate(df, sig, 1.5, 3.0, 0, "short") == []       # side filter
    m = lab.metrics(trades + t2)
    assert m["trades"] == 2 and 0 <= m["win_rate"] <= 100


def test_lab_strategies_run_on_random_data():
    from bot import lab

    df = make_df(900, seed=5)
    for key in lab.STRATEGIES:
        res = lab.run(df, key, {}, 1.5, 3.0, 48, "both")
        assert {"all", "in_sample", "out_of_sample", "verdict", "curve"} <= set(res)
        assert res["all"]["trades"] == res["in_sample"]["trades"] + res["out_of_sample"]["trades"]


def test_scanner_events_detection():
    from collections import deque
    import threading
    from bot.scanner import MarketScanner

    s = MarketScanner.__new__(MarketScanner)
    s._lock, s.events, s._event_id, s._cool, s.notifier = threading.Lock(), deque(maxlen=10), 0, {}, None
    row = {"turnover": 5e7, "base": "AAA"}
    old = {"signal": "NEUTRAL", "rsi": 35.0}
    s._detect("AAA/USDT", old, {"signal": "BULLISH", "rsi": 28.0, "adx": 30, "vol_ratio": 3.5, "chg1h": 2.0}, row)
    kinds = sorted(e["type"] for e in s.events)
    assert kinds == ["RSI", "SIGNAL", "VOLUME"]
    s._detect("AAA/USDT", old, {"signal": "BULLISH", "rsi": 28.0, "adx": 30, "vol_ratio": 3.5, "chg1h": 2.0}, row)
    assert len(s.events) == 3                                              # cooldown blocks duplicates
    s._detect("BBB/USDT", {"signal": None, "rsi": None}, {"signal": "BULLISH", "rsi": 50, "adx": 30, "vol_ratio": 1, "chg1h": 0}, row)
    assert len(s.events) == 3                                              # first analysis never alerts
    s._detect("CCC/USDT", old, {"signal": "BEARISH", "rsi": 50, "adx": 30, "vol_ratio": 1, "chg1h": 0}, {"turnover": 1e6, "base": "CCC"})
    assert len(s.events) == 3                                              # illiquid coins ignored
