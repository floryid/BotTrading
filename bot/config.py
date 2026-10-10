"""Central configuration. Values can be overridden through the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# --- Mode & credentials -----------------------------------------------------
PAPER_TRADING = _bool("PAPER_TRADING", True)  # safe default: never live by accident
API_KEY = os.getenv("API_KEY", "")
SECRET = os.getenv("SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
CRYPTOQUANT_KEY = os.getenv("CRYPTOQUANT_KEY", "")
COINGLASS_KEY = os.getenv("COINGLASS_KEY", "")

# --- Market -----------------------------------------------------------------
SYMBOLS = [s.strip() for s in os.getenv(
    "SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,HYPE/USDT").split(",") if s.strip()]
TIMEFRAME = "15m"
LEVERAGE = _int("LEVERAGE", 5)
SMALL_CAP_SYMBOLS = {"HYPE/USDT"}
OHLCV_LIMIT = 200          # candles refreshed in every slow loop
TRAIN_CANDLES = 1000       # candles used for model training (Bybit max per call)

# --- Loops ------------------------------------------------------------------
FAST_INTERVAL = 1          # seconds
SLOW_INTERVAL = 900        # seconds (15 minutes)
TRADE_COOLDOWN = 900       # seconds before re-entering a symbol after a close
WS_STALE_SECONDS = 15

# --- Risk -------------------------------------------------------------------
BASE_RISK = 0.01           # 1% equity per trade
MAX_RISK_PER_TRADE = 0.02
MAX_TOTAL_RISK = 0.05
MAX_DRAWDOWN = 0.20
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
TAKER_FEE = 0.0006
SLIPPAGE = 0.001
BREAKEVEN_R = _float("BREAKEVEN_R", 1.0)        # move SL to entry once profit reaches this many R (0 = off)
TRAIL_START_R = _float("TRAIL_START_R", 0.0)    # start trailing SL at this many R (0 = off)
TRAIL_ATR_MULT = _float("TRAIL_ATR_MULT", 1.0)  # trailing distance in ATR
MIN_CONFIDENCE = 60
MIN_PROFIT_EDGE = _int("MIN_PROFIT_EDGE", 55)   # skor AI profit-edge minimum untuk entry
PROFIT_EDGE_BONUS = _int("PROFIT_EDGE_BONUS", 12)  # bonus confidence maks jika edge sangat tinggi
MAX_CORRELATION = 0.80
PAPER_BALANCE = _float("PAPER_BALANCE", 10000.0)

# --- Backtest viability gate (live mode only) --------------------------------
BACKTEST_CANDLES = 500
BACKTEST_MIN_WINRATE = 50.0
BACKTEST_MIN_PF = 1.2
BACKTEST_MIN_TRADES = 10

# --- Strategy weights -------------------------------------------------------
WEIGHTS = {"trend": 0.30, "mean_reversion": 0.20, "whale": 0.30, "scalper": 0.20}
TRANSFORMER_WEIGHT = 0.20
AI_ENSEMBLE_WEIGHT = _float("AI_ENSEMBLE_WEIGHT", 0.15)
MIN_VOTE_WEIGHT = 0.30
SCALPER_THRESHOLD = 0.001  # 0.1%

# --- RL ---------------------------------------------------------------------
RL_ALPHA = 0.1
RL_GAMMA = 0.9
RL_EPSILON_START = 0.20
RL_EPSILON_MIN = 0.05
RL_EPSILON_DECAY = 0.995

# --- AI models --------------------------------------------------------------
LSTM_SEQ = 20
TRANSFORMER_SEQ = 30
LSTM_EPOCHS = 30
TRANSFORMER_EPOCHS = 20
RETRAIN_HOURS = 24

# --- Smart money ------------------------------------------------------------
SMART_MONEY_TTL = 300
SM_WEIGHTS = {"whale": 0.40, "liquidation": 0.35, "orderbook": 0.25}
LIQ_SPIKE_USD = 1_000_000

# --- WebSocket --------------------------------------------------------------
BYBIT_DOMAIN = os.getenv("BYBIT_DOMAIN", "bybit.com")  # e.g. bybit.nl when bybit.com is blocked by your ISP
WS_URL = f"wss://stream.{BYBIT_DOMAIN}/v5/public/linear"
WS_MAX_RETRY = 10

# --- Dashboard --------------------------------------------------------------
SCANNER_TOP_N = _int("SCANNER_TOP_N", 80)                    # most liquid USDT perpetuals to monitor
SCANNER_ANALYZE_N = _int("SCANNER_ANALYZE_N", 50)            # of those, how many get 1H technical analysis
SCANNER_TICKER_SECONDS = _int("SCANNER_TICKER_SECONDS", 15)
SCANNER_ANALYSIS_SECONDS = _int("SCANNER_ANALYSIS_SECONDS", 300)
SCANNER_TELEGRAM = _bool("SCANNER_TELEGRAM", False)          # push auto-detected opportunities to Telegram
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = _int("DASHBOARD_PORT", 5000)
DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN", "")  # optional access token

# --- Files ------------------------------------------------------------------
POSITIONS_FILE = DATA_DIR / "positions.json"
QTABLE_FILE = DATA_DIR / "rl_qtable.json"
BACKTEST_FILE = DATA_DIR / "backtest_result.json"
STATE_FILE = DATA_DIR / "state.json"
ALERTS_FILE = DATA_DIR / "alerts.json"
LOG_FILE = DATA_DIR / "bot.log"


def swap_symbol(sym: str) -> str:
    """BTC/USDT -> ccxt linear perpetual symbol BTC/USDT:USDT."""
    return f"{sym}:USDT"


def ws_symbol(sym: str) -> str:
    """BTC/USDT -> BTCUSDT."""
    return sym.replace("/", "")


def file_key(sym: str) -> str:
    return sym.replace("/", "")
