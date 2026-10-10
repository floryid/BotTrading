import logging
from logging.handlers import RotatingFileHandler

from . import config


def setup_logging() -> logging.Logger:
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
        file_handler = RotatingFileHandler(config.LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        for noisy in ("urllib3", "websocket", "werkzeug"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("bot")
