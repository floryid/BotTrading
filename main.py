"""Entry point: python main.py [--reset-paper]"""
import argparse
import logging
import signal
import sys

from bot import config
from bot.dashboard import start_dashboard
from bot.engine import TradingBot
from bot.logger import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Hedge Fund v2.0 - Bybit futures bot")
    parser.add_argument("--reset-paper", action="store_true",
                        help="delete saved paper state/positions (use after a circuit breaker)")
    args = parser.parse_args()
    setup_logging()
    log = logging.getLogger("main")

    if args.reset_paper:
        for f in (config.POSITIONS_FILE, config.STATE_FILE):
            f.unlink(missing_ok=True)
        log.info("Paper state reset")

    bot = TradingBot()
    signal.signal(signal.SIGINT, lambda *_: bot.stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: bot.stop_event.set())

    if not bot.startup():
        bot.shutdown("startup failed")
        return 1
    bot.sync_dashboard()
    start_dashboard(bot)
    bot.start_feed()

    code = bot.run()
    bot.shutdown("circuit breaker" if code == 3 else "manual")
    return code


if __name__ == "__main__":
    sys.exit(main())
