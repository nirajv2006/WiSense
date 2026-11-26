"""Data acquisition: parse CSI lines, read from serial or replay files, save recordings.

Supported line formats
----------------------
1. WiSense receiver firmware (this repo):
       CSI,<millis>,<rssi>,<n_values>,<imag0> <real0> <imag1> <real1> ...
2. Espressif ESP-CSI examples (csi_recv / csi_recv_router):
       CSI_DATA,<seq>,<mac>,<rssi>,...,<local_timestamp_us>,...,"[imag0,real0,imag1,real1,...]"

Both are converted into the same `CSIFrame`, so everything downstream is
independent of the firmware used.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

N_IQ = 2 * config.N_RAW_SUBCARRIERS      # 128 raw int8 values
IQ_COLS = [f"iq{i}" for i in range(N_IQ)]
_BRACKET = re.compile(r"\[([^\]]*)\]")


@dataclass
class CSIFrame:
    ts_ms: float          # device timestamp in milliseconds
    rssi: int
    iq: np.ndarray        # (128,) int16 raw values, (imag, real) pairs


def parse_line(line: str) -> CSIFrame | None:
    """Parse one serial line. Returns None for anything that is not CSI."""
    line = line.strip()
    try:
        if line.startswith("CSI,"):
            _, ts, rssi, _n, data = line.split(",", 4)
            values = np.array(data.split(), dtype=np.int16)
            ts_ms = float(ts)
        elif line.startswith("CSI_DATA"):
            m = _BRACKET.search(line)
            if not m:
                return None
            fields = line[: m.start()].split(",")
            values = np.array([v for v in m.group(1).replace(",", " ").split()], dtype=np.int16)
            rssi = fields[3]
            # local_timestamp (µs) is column 18 in the classic ESP-CSI format
            try:
                ts_ms = float(fields[18]) / 1000.0
            except (IndexError, ValueError):
                ts_ms = time.time() * 1000.0
        else:
            return None
        if values.size < N_IQ:
            return None
        return CSIFrame(ts_ms=ts_ms, rssi=int(rssi), iq=values[:N_IQ])
    except (ValueError, IndexError):
        return None


# ── Sources ──────────────────────────────────────────────────────────────────
class SerialSource:
    """Reads CSI frames from the receiver over USB serial."""

    def __init__(self, port: str, baud: int = config.BAUD):
        import serial  # pyserial

        self.ser = serial.Serial(port, baud, timeout=1)
        self.ser.reset_input_buffer()

    def __iter__(self):
        while True:
            raw = self.ser.readline()
            if not raw:
                continue
            frame = parse_line(raw.decode(errors="ignore"))
            if frame is not None:
                yield frame

    def send(self, text: str) -> None:
        try:
            self.ser.write((text + "\n").encode())
        except Exception:
            pass


class ReplaySource:
    """Plays back recorded CSV files as if they came from the board (for demos/tests)."""

    def __init__(self, paths, speed: float = 1.0, loop: bool = False):
        self.paths = [Path(p) for p in paths]
        self.speed, self.loop = speed, loop

    def __iter__(self):
        offset, last_out = 0.0, None
        while True:
            for p in self.paths:
                df = pd.read_csv(p)
                iq = df[IQ_COLS].to_numpy(np.int16)
                ts = df["ts_ms"].to_numpy(float)
                rssi = df["rssi"].to_numpy(int)
                # make consecutive recordings one continuous timeline
                if last_out is not None:
                    offset = last_out + 1000.0 / config.EXPECTED_RATE_HZ - ts[0]
                ts = ts + offset
                last_out = ts[-1]
                t_start, wall_start = ts[0], time.time()
                for i in range(len(df)):
                    if self.speed > 0:
                        due = wall_start + (ts[i] - t_start) / 1000.0 / self.speed
                        delay = due - time.time()
                        if delay > 0:
                            time.sleep(delay)
                    yield CSIFrame(ts_ms=ts[i], rssi=int(rssi[i]), iq=iq[i])
            if not self.loop:
                return

    def send(self, text: str) -> None:
        pass


# ── Recordings on disk ───────────────────────────────────────────────────────
def save_recording(frames: list[CSIFrame], host_times: list[float], path: Path, meta: dict) -> None:
    """One recording = CSV of raw frames + JSON sidecar with its label and conditions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(np.vstack([f.iq for f in frames]), columns=IQ_COLS)
    df.insert(0, "host_time", np.round(host_times, 3))
    df.insert(0, "rssi", [f.rssi for f in frames])
    df.insert(0, "ts_ms", [f.ts_ms for f in frames])
    df.to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_recording(path: Path):
    """Returns (ts_ms, rssi, iq[N,128], meta dict)."""
    path = Path(path)
    df = pd.read_csv(path)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return (df["ts_ms"].to_numpy(float), df["rssi"].to_numpy(float),
            df[IQ_COLS].to_numpy(np.int16), meta)
