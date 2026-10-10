"""LSTM and Transformer next-close predictors (PyTorch), per symbol."""
import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config

log = logging.getLogger("models")
torch.set_num_threads(2)


N_FEATURES = 5


class LSTMNet(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class TransformerNet(nn.Module):
    def __init__(self, seq_len: int, n_features: int = N_FEATURES, d_model: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.encoder(self.proj(x) + self.pos)
        return self.head(h.mean(dim=1)).squeeze(-1)


def _window_features(close: np.ndarray, volume: np.ndarray, rsi_tail: Optional[np.ndarray] = None,
                     ema_spread: Optional[np.ndarray] = None) -> np.ndarray:
    """Scale-free features for sequence models (price, volume, momentum, RSI, trend spread)."""
    c = (close / close[-1] - 1.0) * 100.0
    v = volume / (volume.mean() or 1.0) - 1.0
    mom = np.zeros_like(c)
    if len(close) >= 6:
        mom[-5:] = (close[-5:] / close[-6:-1] - 1.0) * 100.0
    if rsi_tail is not None and len(rsi_tail) == len(close):
        rsi_n = (rsi_tail - 50.0) / 50.0
    else:
        rsi_n = np.zeros_like(c)
    if ema_spread is not None and len(ema_spread) == len(close):
        spread = ema_spread
    else:
        spread = np.zeros_like(c)
    return np.stack([c, v, mom, rsi_n, spread], axis=1).astype(np.float32)


def _series_features(df: pd.DataFrame, i: int, seq: int):
    w = df.iloc[i - seq:i]
    close = w["close"].to_numpy(dtype=np.float64)
    volume = w["volume"].to_numpy(dtype=np.float64)
    ema20 = w["ema20"].to_numpy(dtype=np.float64) if "ema20" in w else close
    ema50 = w["ema50"].to_numpy(dtype=np.float64) if "ema50" in w else close
    spread = np.where(ema50 != 0, (ema20 / ema50 - 1.0) * 100.0, 0.0)
    rsi = w["rsi"].to_numpy(dtype=np.float64) if "rsi" in w else np.full(seq, 50.0)
    return _window_features(close, volume, rsi, spread)


def make_dataset(df: pd.DataFrame, seq: int):
    from .indicators import add_indicators

    d = add_indicators(df) if "ema20" not in df.columns else df
    close = d["close"].to_numpy(dtype=np.float64)
    xs, ys = [], []
    for i in range(seq, len(d)):
        xs.append(_series_features(d, i, seq))
        ys.append((close[i] / close[i - 1] - 1.0) * 100.0)
    if not xs:
        return None, None
    return torch.tensor(np.array(xs)), torch.tensor(np.array(ys, dtype=np.float32))


def _fit(model: nn.Module, x, y, epochs: int) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    n = len(x)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, 64):
            idx = perm[start:start + 64]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()


class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()  # _model_lock
        self.lstm: dict = {}
        self.transformer: dict = {}
        self.last_trained: dict = {}

    @staticmethod
    def _lstm_path(sym: str):
        return config.DATA_DIR / f"lstm_{config.file_key(sym)}.pt"

    def init_symbol(self, sym: str, df: pd.DataFrame) -> None:
        """Startup: load LSTM from disk (or train it) and always train the Transformer."""
        path = self._lstm_path(sym)
        net = LSTMNet()
        loaded = False
        if path.exists():
            try:
                net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                net.eval()
                loaded = True
                log.info("LSTM %s loaded from disk", sym)
            except Exception as exc:  # corrupt / incompatible file -> retrain
                log.warning("LSTM %s load failed (%s), retraining", sym, exc)
        if not loaded:
            self._train_lstm(sym, df, net)
        self._train_transformer(sym, df)
        with self._lock:
            self.last_trained[sym] = time.time() if not loaded else path.stat().st_mtime

    def _train_lstm(self, sym: str, df: pd.DataFrame, net: Optional[LSTMNet] = None) -> bool:
        x, y = make_dataset(df, config.LSTM_SEQ)
        if x is None or len(x) < 50:
            log.warning("Not enough data to train LSTM %s", sym)
            return False
        net = net or LSTMNet()
        _fit(net, x, y, config.LSTM_EPOCHS)
        torch.save(net.state_dict(), self._lstm_path(sym))
        with self._lock:
            self.lstm[sym] = net
        log.info("LSTM %s trained on %d samples", sym, len(x))
        return True

    def _train_transformer(self, sym: str, df: pd.DataFrame) -> bool:
        x, y = make_dataset(df, config.TRANSFORMER_SEQ)
        if x is None or len(x) < 50:
            log.warning("Not enough data to train Transformer %s", sym)
            return False
        net = TransformerNet(config.TRANSFORMER_SEQ)
        _fit(net, x, y, config.TRANSFORMER_EPOCHS)
        with self._lock:
            self.transformer[sym] = net
        log.info("Transformer %s trained on %d samples", sym, len(x))
        return True

    def retrain_due(self, sym: str) -> bool:
        with self._lock:
            last = self.last_trained.get(sym, 0)
        return time.time() - last > config.RETRAIN_HOURS * 3600

    def retrain(self, sym: str, df: pd.DataFrame) -> bool:
        ok = self._train_lstm(sym, df) and self._train_transformer(sym, df)
        if ok:
            with self._lock:
                self.last_trained[sym] = time.time()
        return ok

    def _predict_one(self, net: Optional[nn.Module], df: pd.DataFrame, seq: int) -> Tuple[Optional[float], Optional[float]]:
        if net is None or len(df) < seq:
            return None, None
        from .indicators import add_indicators

        d = add_indicators(df) if "ema20" not in df.columns else df
        if len(d) < seq:
            return None, None
        close = d["close"].to_numpy(dtype=np.float64)
        x = torch.tensor(_series_features(d, len(d), seq)).unsqueeze(0)
        with torch.no_grad():
            pct = float(net(x).item())
        pct = max(-5.0, min(5.0, pct))
        return float(close[-1] * (1 + pct / 100.0)), pct

    def predict(self, sym: str, df: pd.DataFrame) -> dict:
        """Returns prices, delta %, ensemble, and consolidated AI direction."""
        with self._lock:
            lstm, tf = self.lstm.get(sym), self.transformer.get(sym)
            p_l, dp_l = self._predict_one(lstm, df, config.LSTM_SEQ)
            p_t, dp_t = self._predict_one(tf, df, config.TRANSFORMER_SEQ)
        parts = [p for p in (p_l, p_t) if p is not None]
        dps = [d for d in (dp_l, dp_t) if d is not None]
        ai_dir = WAIT
        ai_conf = 0.0
        if dps:
            avg_dp = float(np.mean(dps))
            if abs(avg_dp) >= 0.04:
                ai_dir = "LONG" if avg_dp > 0 else "SHORT"
            agree = dp_l is not None and dp_t is not None and (dp_l > 0) == (dp_t > 0)
            ai_conf = round(min(1.0, abs(avg_dp) / 0.5) * (1.0 if agree or len(dps) == 1 else 0.65), 2)
        return {
            "lstm": p_l, "transformer": p_t, "ensemble": float(np.mean(parts)) if parts else None,
            "lstm_delta_pct": dp_l, "transformer_delta_pct": dp_t,
            "ai_direction": ai_dir, "ai_confidence": ai_conf,
        }
