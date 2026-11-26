"""Persistence shared by the ingest service (writer) and the dashboard (reader).

SQLite  -> history of every decision (one row per second)
.npz    -> the latest CSI window, for the live heatmap (overwritten atomically)
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    ts            REAL PRIMARY KEY,   -- unix time (s)
    raw_label     TEXT,               -- classifier output for this window
    label         TEXT,               -- after probability smoothing
    confidence    REAL,
    p_low REAL, p_medium REAL, p_high REAL,
    power_level   TEXT,
    power_share   REAL,
    reason        TEXT,
    motion        REAL,               -- sc_std_mean, a model-independent activity indicator
    rssi          REAL,
    pkt_rate      REAL,
    source        TEXT                -- serial port or replay file
);
"""


def connect(path: Path = config.DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=5, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")          # dashboard can read while ingest writes
    con.executescript(SCHEMA)
    return con


def insert(con: sqlite3.Connection, row: dict) -> None:
    cols = ", ".join(row)
    con.execute(f"INSERT OR REPLACE INTO predictions ({cols}) VALUES ({', '.join('?' * len(row))})",
                list(row.values()))
    con.commit()


def read_history(since_ts: float, path: Path = config.DB_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    with sqlite3.connect(path, timeout=5) as con:
        return pd.read_sql_query("SELECT * FROM predictions WHERE ts >= ? ORDER BY ts", con,
                                 params=(since_ts,))


def write_snapshot(amp: np.ndarray, mean_series: np.ndarray, ts: float,
                   path: Path = config.LIVE_SNAPSHOT) -> None:
    tmp = Path(path).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, amp=amp.astype(np.float32), mean=mean_series.astype(np.float32), ts=ts)
    os.replace(tmp, path)


def read_snapshot(path: Path = config.LIVE_SNAPSHOT):
    try:
        with np.load(path) as d:
            return d["amp"], d["mean"], float(d["ts"])
    except (OSError, KeyError, ValueError):
        return None
