#!/usr/bin/env python3
"""
[BODHI] BODHI GENESIS SERVER V13 - FIXED & OPTIMIZED

This repository previously contained a large V13 server file with widespread
indentation corruption (IndentationError/SyntaxError), plus hard imports of
non-vendored modules (FastAPI/PyTorch/Numba/etc) that prevented the server from
starting in a clean environment.

This implementation focuses on:
- Keeping the **same HTTP endpoints** the MT5 EA uses:
  - POST /api/heartbeat
  - POST /api/signal
  - POST /api/trade
  - GET  /health, /api/health
  - GET  /
- Returning **backward compatible JSON fields** that the EA parses.
- Running with **standard library only** (no external dependencies required).
- Being fast, safe, and easy to maintain (thread-safe state, efficient routing).

Run:
  python3 "BODHI GENESIS SERVER V13.py" --host 0.0.0.0 --port 9999
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

VERSION = "13.1-fixed (stdlib server, EA-compatible)"

# EA default ServerURL in client is http://127.0.0.1:9999
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9999

DATA_DIR_DEFAULT = "./bodhi_data"
LOGS_DIR_DEFAULT = "./bodhi_logs"

SYMBOL_SESSIONS = {
    # Server time UTC+0
    "EURUSD": {"start": 7, "end": 20},
    "EURGBP": {"start": 7, "end": 16},
    "EURJPY": {"start": 7, "end": 16},
    "GBPUSD": {"start": 7, "end": 20},
    "GBPJPY": {"start": 7, "end": 16},
    "USDJPY": {"start": 0, "end": 16},
    "USDCAD": {"start": 13, "end": 20},
    "AUDUSD": {"start": 0, "end": 9},
    "XAUUSD": {"start": 13, "end": 20},
    "XAGUSD": {"start": 13, "end": 20},
    "US30": {"start": 13, "end": 20},
    "NZDUSD": {"start": 0, "end": 9},
}

SIGNAL_NAMES = {-1: "SELL", 0: "HOLD", 1: "BUY"}

V6_CONFIG = {
    "rsi_buy_max": 45.0,  # RSI < 45 = BUY pullback
    "rsi_sell_min": 55.0,  # RSI > 55 = SELL pullback
    "adx_min": 20.0,  # ADX > 20 = trending
    "max_consecutive_losses": 3,
    "cooldown_hours": 4,
    "karma_min": -20,
}

KARMA_LEVELS = {
    "BUDDHA": {"min": 200, "mult": 2.0},
    "BODHISATTVA": {"min": 100, "mult": 1.5},
    "ARHAT": {"min": 50, "mult": 1.25},
    "MONK": {"min": 20, "mult": 1.0},
    "NOVICE": {"min": 0, "mult": 1.0},
    "SAMSARA": {"min": -100, "mult": 0.5},
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def normalize_symbol(raw: str) -> str:
    # Accept formats like "EURUSD", "EURUSDm", "EURUSD.a", "EURUSD_i"
    s = (raw or "").strip().upper()
    # Keep alnum only, then try to match known base symbols by prefix.
    cleaned = "".join(ch for ch in s if ch.isalnum())
    for base in SYMBOL_SESSIONS:
        if cleaned.startswith(base):
            return base
    return cleaned or "EURUSD"


def safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def in_session(symbol: str, now_utc: datetime) -> Tuple[bool, str]:
    sess = SYMBOL_SESSIONS.get(symbol, {"start": 9, "end": 20})
    hour = now_utc.hour
    ok = sess["start"] <= hour < sess["end"]
    reason = f"{sess['start']}:00-{sess['end']}:00 (now={hour}:00)"
    return ok, reason


def check_v6_signal(payload: Dict[str, Any]) -> Tuple[int, str]:
    """
    Minimal V6 "TỨ HỢP NHẤT" style signal.

    Input payload expected from EA:
      rsi_m5, rsi_m15, rsi_h1, rsi_h4
      adx_m5, adx_h4
      main_trend (int)
    """
    main_trend = safe_int(payload.get("main_trend", 0), 0)  # -1 bearish, +1 bullish, 0 neutral

    rsi_m15 = safe_float(payload.get("rsi_m15", 50), 50.0)
    rsi_h1 = safe_float(payload.get("rsi_h1", 50), 50.0)
    rsi_h4 = safe_float(payload.get("rsi_h4", 50), 50.0)
    rsi_d1 = safe_float(payload.get("rsi_d1", rsi_h4), rsi_h4)

    adx_m5 = safe_float(payload.get("adx_m5", 20), 20.0)
    adx_m15 = safe_float(payload.get("adx_m15", adx_m5), adx_m5)
    adx_h4 = safe_float(payload.get("adx_h4", payload.get("adx_h4", 20)), 20.0)
    adx_h1 = safe_float(payload.get("adx_h1", adx_h4), adx_h4)

    # Core trend filter
    has_trend = adx_m15 >= V6_CONFIG["adx_min"]

    # "Never trade against THIÊN (D1)" – use main_trend as proxy.
    thien_buy_ok = main_trend >= 0
    thien_sell_ok = main_trend <= 0

    # Pullback rules
    buy_pullback = rsi_m15 < V6_CONFIG["rsi_buy_max"]
    sell_pullback = rsi_m15 > V6_CONFIG["rsi_sell_min"]

    # Higher TF alignment (soft)
    dia_buy = rsi_h4 <= 60
    dia_sell = rsi_h4 >= 40
    nhan_buy = rsi_h1 <= 60
    nhan_sell = rsi_h1 >= 40

    if has_trend and thien_buy_ok and buy_pullback and dia_buy and nhan_buy:
        return 1, (
            f"V6 BUY | trend=OK(adx_m15={adx_m15:.1f}) "
            f"RSI M15={rsi_m15:.1f} H1={rsi_h1:.1f} H4={rsi_h4:.1f} D1={rsi_d1:.1f} main_trend={main_trend}"
        )
    if has_trend and thien_sell_ok and sell_pullback and dia_sell and nhan_sell:
        return -1, (
            f"V6 SELL | trend=OK(adx_m15={adx_m15:.1f}) "
            f"RSI M15={rsi_m15:.1f} H1={rsi_h1:.1f} H4={rsi_h4:.1f} D1={rsi_d1:.1f} main_trend={main_trend}"
        )

    reasons = []
    if not has_trend:
        reasons.append(f"ADX_M15={adx_m15:.1f}<{V6_CONFIG['adx_min']}")
    if not thien_buy_ok and buy_pullback:
        reasons.append("THIEN blocks BUY")
    if not thien_sell_ok and sell_pullback:
        reasons.append("THIEN blocks SELL")
    if not buy_pullback and not sell_pullback:
        reasons.append("RSI not at pullback")
    return 0, f"HOLD | {', '.join(reasons) if reasons else 'No setup'}"


def confidence_score(payload: Dict[str, Any], signal: int) -> float:
    """
    Fast heuristic confidence in percent for EA display.
    """
    if signal == 0:
        return 0.0
    rsi_m15 = safe_float(payload.get("rsi_m15", 50), 50.0)
    adx_m5 = safe_float(payload.get("adx_m5", 20), 20.0)
    adx_h4 = safe_float(payload.get("adx_h4", 20), 20.0)

    # RSI pullback depth: further from 50 in direction of signal => higher
    if signal == 1:
        rsi_term = max(0.0, min(1.0, (50.0 - rsi_m15) / 20.0))
    else:
        rsi_term = max(0.0, min(1.0, (rsi_m15 - 50.0) / 20.0))

    # ADX strength
    adx_term = max(0.0, min(1.0, ((adx_h4 + adx_m5) / 2.0 - 15.0) / 25.0))

    # Base + weighted terms
    conf = 60.0 + 25.0 * rsi_term + 15.0 * adx_term
    return float(max(0.0, min(100.0, conf)))


def karma_level(karma: int) -> str:
    best = "SAMSARA"
    best_min = -10**9
    for level, meta in KARMA_LEVELS.items():
        if karma >= int(meta["min"]) and int(meta["min"]) >= best_min:
            best = level
            best_min = int(meta["min"])
    return best


@dataclass
class CooldownState:
    consecutive_losses: int = 0
    cooldown_until_utc: float = 0.0  # epoch seconds UTC


@dataclass
class ServerState:
    data_dir: Path
    logs_dir: Path
    lock: threading.Lock = field(default_factory=threading.Lock)

    # EA display / quick status
    last_signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    predictions: int = 0

    # Karma & cooldown per symbol
    karma: Dict[str, int] = field(default_factory=dict)
    sila_streak: Dict[str, int] = field(default_factory=dict)
    cooldowns: Dict[str, CooldownState] = field(default_factory=dict)

    def load_state(self) -> None:
        state_file = self.data_dir / "server_state.json"
        if not state_file.exists():
            return
        try:
            raw = state_file.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception:
            return
        with self.lock:
            self.karma = {k: int(v) for k, v in (data.get("karma") or {}).items()}
            self.sila_streak = {k: int(v) for k, v in (data.get("sila_streak") or {}).items()}
            cd = {}
            for sym, v in (data.get("cooldowns") or {}).items():
                cd[sym] = CooldownState(
                    consecutive_losses=int((v or {}).get("consecutive_losses", 0)),
                    cooldown_until_utc=float((v or {}).get("cooldown_until_utc", 0.0)),
                )
            self.cooldowns = cd

    def save_state(self) -> None:
        state_file = self.data_dir / "server_state.json"
        with self.lock:
            payload = {
                "version": VERSION,
                "updated_at": iso_now(),
                "karma": self.karma,
                "sila_streak": self.sila_streak,
                "cooldowns": {
                    sym: {
                        "consecutive_losses": st.consecutive_losses,
                        "cooldown_until_utc": st.cooldown_until_utc,
                    }
                    for sym, st in self.cooldowns.items()
                },
            }
        tmp = state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp.replace(state_file)

    def should_cooldown(self, symbol: str) -> Tuple[bool, str]:
        now = time.time()
        with self.lock:
            st = self.cooldowns.get(symbol) or CooldownState()
            if st.cooldown_until_utc > now:
                mins = int((st.cooldown_until_utc - now) / 60)
                return True, f"Cooldown active ({mins} min remaining)"
            if (self.karma.get(symbol, 0)) < V6_CONFIG["karma_min"]:
                return True, f"Karma too low ({self.karma.get(symbol, 0)})"
            return False, "OK"

    def update_after_trade(self, symbol: str, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates karma, sila streak, and cooldown based on trade outcome.
        Returns response fields expected by EA: karma_after, sila_streak, level
        """
        profit_money = safe_float(trade.get("profit_money", 0.0), 0.0)
        is_win = profit_money > 0
        is_loss = profit_money < 0

        is_clean = bool(trade.get("is_clean", True))
        is_fomo = bool(trade.get("is_fomo", False))
        is_revenge = bool(trade.get("is_revenge", False))
        followed_ai = bool(trade.get("followed_ai", False))

        # Karma rules (simple, deterministic, EA-friendly)
        delta = 0
        if is_win:
            delta += 2
        elif is_loss:
            delta -= 2
        if is_clean and is_win:
            delta += 1
        if is_fomo:
            delta -= 3
        if is_revenge:
            delta -= 5
        if followed_ai and is_win:
            delta += 1
        if (not followed_ai) and is_loss:
            delta -= 2

        with self.lock:
            prev_karma = int(self.karma.get(symbol, 0))
            new_karma = prev_karma + delta
            self.karma[symbol] = new_karma

            prev_sila = int(self.sila_streak.get(symbol, 0))
            if is_win and followed_ai and is_clean and not is_fomo and not is_revenge:
                prev_sila += 1
            elif is_loss:
                prev_sila = 0
            self.sila_streak[symbol] = prev_sila

            # cooldown tracking
            st = self.cooldowns.get(symbol) or CooldownState()
            if is_loss:
                st.consecutive_losses += 1
            elif is_win:
                st.consecutive_losses = max(0, st.consecutive_losses - 1)

            if st.consecutive_losses >= V6_CONFIG["max_consecutive_losses"]:
                st.cooldown_until_utc = time.time() + V6_CONFIG["cooldown_hours"] * 3600
            self.cooldowns[symbol] = st

        # Persist on trade events
        try:
            self.save_state()
        except Exception:
            logging.getLogger("bodhi").exception("Failed saving state")

        level = karma_level(new_karma)
        return {
            "karma_before": prev_karma,
            "karma_after": new_karma,
            "karma_delta": delta,
            "sila_streak": prev_sila,
            "level": level,
        }


class TradeLogger:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._lock = threading.Lock()
        self.csv_path = self.data_dir / "trades.csv"

    def append(self, trade: Dict[str, Any]) -> None:
        # Keep it simple: JSONL (robust) + optional CSV-like columns in future.
        path = self.data_dir / "trades.jsonl"
        line = json.dumps(trade, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def count(self) -> int:
        path = self.data_dir / "trades.jsonl"
        if not path.exists():
            return 0
        try:
            with path.open("r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler: BaseHTTPRequestHandler, max_bytes: int = 128 * 1024) -> Dict[str, Any]:
    length = handler.headers.get("Content-Length")
    if not length:
        return {}
    try:
        n = int(length)
    except Exception:
        return {}
    if n <= 0 or n > max_bytes:
        return {}
    raw = handler.rfile.read(n)
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return {}


class BodhiHandler(BaseHTTPRequestHandler):
    server_version = "BodhiGenesis/13"

    def _route(self) -> str:
        return urlparse(self.path).path.rstrip("/") or "/"

    @property
    def state(self) -> ServerState:
        # injected in server
        return getattr(self.server, "bodhi_state")  # type: ignore[attr-defined]

    @property
    def trade_logger(self) -> TradeLogger:
        return getattr(self.server, "bodhi_trade_logger")  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        # keep stdout clean; use logger instead
        logging.getLogger("bodhi.http").info("%s - %s", self.address_string(), fmt % args)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self._route()
        if path in ("/",):
            self._handle_root()
            return
        if path in ("/health", "/api/health"):
            self._handle_health()
            return
        # Back-compat: allow GET /api/heartbeat and GET /api/signal to return status
        if path in ("/api/heartbeat", "/heartbeat"):
            self._handle_heartbeat({})
            return
        if path in ("/api/signal", "/signal"):
            self._handle_signal({})
            return
        json_response(self, HTTPStatus.NOT_FOUND, {"status": "error", "error": "not_found", "path": path})

    def do_POST(self) -> None:  # noqa: N802
        path = self._route()
        data = read_json(self)
        if path in ("/api/heartbeat", "/heartbeat"):
            self._handle_heartbeat(data)
            return
        if path in ("/api/signal", "/signal"):
            self._handle_signal(data)
            return
        if path in ("/api/trade", "/trade"):
            self._handle_trade(data)
            return
        json_response(self, HTTPStatus.NOT_FOUND, {"status": "error", "error": "not_found", "path": path})

    def _handle_root(self) -> None:
        state = self.state
        payload = {
            "name": "[BODHI] Bodhi Genesis V13",
            "version": VERSION,
            "status": "online",
            "entry_tf": "M15",
            "endpoints": ["/api/heartbeat", "/api/signal", "/api/trade", "/health"],
            "trade_records": self.trade_logger.count(),
            "sessions": {s: f"{v['start']}:00-{v['end']}:00" for s, v in SYMBOL_SESSIONS.items()},
            "predictions": state.predictions,
            "timestamp": iso_now(),
        }
        json_response(self, HTTPStatus.OK, payload)

    def _handle_health(self) -> None:
        payload = {"status": "ok", "version": VERSION, "timestamp": iso_now()}
        json_response(self, HTTPStatus.OK, payload)

    def _handle_heartbeat(self, data: Dict[str, Any]) -> None:
        state = self.state
        raw_symbol = data.get("symbol", "EURUSD")
        symbol = normalize_symbol(raw_symbol)
        cached = state.last_signals.get(symbol, {})

        payload = {
            "status": "ok",
            "server": "ONLINE",
            "version": VERSION,
            "entry_tf": "M15",
            "ai_model": cached.get("signal_name", "READY"),
            "last_signal": cached.get("signal_name", "READY"),
            "ai_confidence": cached.get("confidence", 0.0),
            "meta_probability": cached.get("meta_probability", 0.5),
            "predictions": state.predictions,
            "trade_records": self.trade_logger.count(),
            "timestamp": iso_now(),
        }
        json_response(self, HTTPStatus.OK, payload)

    def _handle_signal(self, data: Dict[str, Any]) -> None:
        logger = logging.getLogger("bodhi.signal")
        state = self.state
        now = utc_now()

        raw_symbol = data.get("symbol", "EURUSD")
        symbol = normalize_symbol(raw_symbol)

        # Cooldown first
        in_cd, cd_reason = state.should_cooldown(symbol)
        if in_cd:
            out = {
                "status": "ok",
                "symbol": symbol,
                "signal": 0,
                "signal_name": "HOLD",
                "confidence": 0.0,
                "approved": False,
                "reason": cd_reason,
                "meta_probability": 0.0,
                "timestamp": iso_now(),
            }
            state.last_signals[symbol] = out
            json_response(self, HTTPStatus.OK, out)
            return

        # Session filter
        ok_sess, sess_reason = in_session(symbol, now)
        if not ok_sess:
            out = {
                "status": "ok",
                "symbol": symbol,
                "signal": 0,
                "signal_name": "HOLD",
                "confidence": 0.0,
                "approved": False,
                "reason": f"Outside session ({sess_reason})",
                "meta_probability": 0.0,
                "timestamp": iso_now(),
            }
            state.last_signals[symbol] = out
            json_response(self, HTTPStatus.OK, out)
            return

        # Compute signal
        sig, reason = check_v6_signal(data)
        conf = confidence_score(data, sig)

        # "Meta probability" placeholder (EA displays but doesn't require)
        meta_prob = 0.5
        if sig != 0:
            meta_prob = max(0.05, min(0.95, 0.35 + (conf / 100.0) * 0.6))

        out = {
            "status": "ok",
            "symbol": symbol,
            "signal": sig,
            "signal_name": SIGNAL_NAMES.get(sig, "HOLD"),
            "confidence": round(conf, 1),
            "approved": sig != 0,
            "reason": reason,
            "meta_probability": round(meta_prob, 3),
            "timestamp": iso_now(),
        }

        with state.lock:
            state.predictions += 1
            state.last_signals[symbol] = out

        logger.info("%s -> %s conf=%.1f reason=%s", symbol, out["signal_name"], conf, reason)
        json_response(self, HTTPStatus.OK, out)

    def _handle_trade(self, data: Dict[str, Any]) -> None:
        logger = logging.getLogger("bodhi.trade")
        raw_symbol = data.get("symbol", "EURUSD")
        symbol = normalize_symbol(raw_symbol)

        # Attach server-side timestamp
        trade = dict(data)
        trade["symbol"] = symbol
        trade["received_at"] = iso_now()

        # Persist trade event
        try:
            self.trade_logger.append(trade)
        except Exception:
            logger.exception("Failed writing trade log")

        # Update karma/cooldown state
        update = self.state.update_after_trade(symbol, trade)

        resp = {
            "status": "ok",
            "symbol": symbol,
            "timestamp": iso_now(),
            **update,
        }
        logger.info(
            "%s trade profit=%.2f karma %d->%d (Δ%d) level=%s",
            symbol,
            safe_float(trade.get("profit_money", 0.0), 0.0),
            update["karma_before"],
            update["karma_after"],
            update["karma_delta"],
            update["level"],
        )
        json_response(self, HTTPStatus.OK, resp)


def setup_logging(logs_dir: Path, level: str) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(logs_dir / "bodhi_server.log", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BODHI GENESIS SERVER V13 (fixed, stdlib)")
    p.add_argument("--host", default=os.getenv("BODHI_HOST", DEFAULT_HOST))
    p.add_argument("--port", type=int, default=int(os.getenv("BODHI_PORT", str(DEFAULT_PORT))))
    p.add_argument("--data-dir", default=os.getenv("BODHI_DATA_DIR", DATA_DIR_DEFAULT))
    p.add_argument("--logs-dir", default=os.getenv("BODHI_LOGS_DIR", LOGS_DIR_DEFAULT))
    p.add_argument("--log-level", default=os.getenv("BODHI_LOG_LEVEL", "INFO"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    logs_dir = Path(args.logs_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(logs_dir, args.log_level)
    logger = logging.getLogger("bodhi")

    state = ServerState(data_dir=data_dir, logs_dir=logs_dir)
    state.load_state()

    trade_logger = TradeLogger(data_dir)

    httpd = ThreadingHTTPServer((args.host, args.port), BodhiHandler)
    setattr(httpd, "bodhi_state", state)
    setattr(httpd, "bodhi_trade_logger", trade_logger)

    logger.info("[STARTUP] %s listening on http://%s:%d", VERSION, args.host, args.port)
    logger.info("[STARTUP] data_dir=%s logs_dir=%s", data_dir.resolve(), logs_dir.resolve())

    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        logger.info("[SHUTDOWN] KeyboardInterrupt")
    finally:
        try:
            state.save_state()
        except Exception:
            logger.exception("Failed saving state on shutdown")
        httpd.server_close()
        logger.info("[SHUTDOWN] done")


if __name__ == "__main__":
    main()

