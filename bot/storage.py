import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("storage")
_io_lock = threading.Lock()


def save_json(path: Path, data: Any) -> None:
    """Atomic write: write to a temp file then replace."""
    tmp = Path(f"{path}.tmp")
    with _io_lock:
        try:
            tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            log.error("Failed to save %s: %s", path, exc)


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        log.error("Failed to load %s: %s", path, exc)
        return default
