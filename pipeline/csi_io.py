"""Parsing and loading of WiSense raw CSI data.

Receiver serial line format (one per received frame):

    CSI,seq,ts_us,rx_seq,rssi,sig_mode,cwb,first_invalid,len,v0 v1 v2 ...

`v*` are raw signed 8-bit values from the ESP32 driver, in pairs
(imaginary, real) per subcarrier, so `len` bytes = `len // 2` subcarriers.

The logger stores each frame as a CSV row with the same fields, plus the
laptop's receive time, in data/raw/<session_id>.csv.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CSI_FIELDS = ["seq", "ts_us", "rx_seq", "rssi", "sig_mode", "cwb", "first_invalid", "len"]
RAW_COLUMNS = ["host_time_s", *CSI_FIELDS, "csi"]
RX_SEQ_MOD = 4096  # 802.11 sequence numbers are 12-bit


@dataclass
class CsiFrame:
    seq: int
    ts_us: int
    rx_seq: int
    rssi: int
    sig_mode: int
    cwb: int
    first_invalid: int
    len: int
    values: list[int]


def parse_csi_line(line: str) -> CsiFrame | None:
    """Parse one `CSI,...` serial line. Returns None if it is malformed."""
    if not line.startswith("CSI,"):
        return None
    parts = line.strip().split(",")
    if len(parts) != 2 + len(CSI_FIELDS):
        return None
    try:
        header = [int(p) for p in parts[1:-1]]
        values = [int(v) for v in parts[-1].split()]
    except ValueError:
        return None
    frame = CsiFrame(*header, values=values)
    if frame.len != len(values) or frame.len % 2 or frame.len < 4:
        return None  # truncated line (e.g. serial overrun)
    if any(v < -128 or v > 127 for v in values):
        return None
    return frame


def rx_seq_gap(prev: int, cur: int) -> int:
    """Frames missing between two 802.11 sequence numbers (handles wrap).

    0 means consecutive. A duplicate (retransmission) also returns 0.
    """
    diff = (cur - prev) % RX_SEQ_MOD
    return max(diff - 1, 0)


class GapCounter:
    """Counts lost frames from 802.11 sequence numbers.

    Counters are tracked per PHY mode (sig_mode): management/association
    frames (non-HT) and data frames (HT) use different sequence counters, and
    mixing them makes the count meaningless. A jump larger than `reset_jump`
    is treated as a counter reset (e.g. re-association), not as loss.
    """

    def __init__(self, reset_jump: int = 500):
        self.reset_jump = reset_jump
        self.prev: dict[int, int] = {}
        self.gaps = 0
        self.resets = 0

    def update(self, sig_mode: int, rx_seq: int) -> None:
        prev = self.prev.get(sig_mode)
        self.prev[sig_mode] = rx_seq
        if prev is None:
            return
        gap = rx_seq_gap(prev, rx_seq)
        if gap > self.reset_jump:
            self.resets += 1
        else:
            self.gaps += gap


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a raw session CSV written by logger.py."""
    return pd.read_csv(path, dtype={"csi": str})


def csi_complex(df: pd.DataFrame) -> np.ndarray:
    """Convert raw rows to a complex matrix of shape (frames, subcarriers).

    Only frames with the session's most common PHY mode (sig_mode) and CSI
    length are kept: other modes have a different gain/layout and would
    show up as fake amplitude jumps. Subcarriers flagged by `first_invalid`
    are set to NaN.
    """
    sub = df[df["sig_mode"] == df["sig_mode"].mode().iloc[0]]
    sub = sub[sub["len"] == sub["len"].mode().iloc[0]]
    raw = np.array([s.split() for s in sub["csi"]], dtype=np.int16).astype(np.float32)
    imag, real = raw[:, 0::2], raw[:, 1::2]
    h = (real + 1j * imag).astype(np.complex64)
    h[sub["first_invalid"].to_numpy() == 1, :2] = np.nan
    return h


def amplitude(df: pd.DataFrame, drop_null: bool = True) -> np.ndarray:
    """Per-subcarrier amplitude |H|, shape (frames, subcarriers).

    With drop_null, subcarriers that are zero in almost every frame
    (guard bands / DC) are removed.
    """
    amp = np.abs(csi_complex(df))
    if drop_null:
        zero_frac = np.mean((amp == 0) | np.isnan(amp), axis=0)
        amp = amp[:, zero_frac < 0.95]
    return amp
