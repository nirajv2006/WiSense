"""Generate SYNTHETIC CSI recordings for testing the pipeline without hardware.

    python simulate.py                     # 4 sessions x 3 classes x 2 recordings
    python simulate.py --sessions 2 --seconds 30

Output goes to data/raw/SYN-sXX/ with "synthetic": true in the metadata.
build_dataset.py ignores it unless you pass --include-synthetic.

!! Synthetic data is only for testing code and the dashboard. Never report
!! results obtained on it as WiSense performance.

Model: each subcarrier's channel = static multipath (different per session/room)
+ one reflected path per person (path length changes with walking / breathing)
+ partial line-of-sight blocking, receiver noise, random AGC gain, outlier
packets, packet jitter and drops, then quantised to int8 like the ESP32.
"""
import argparse
import time

import numpy as np

from wisense import config
from wisense.csi_io import CSIFrame, save_recording

C = 3e8
FC = 2.437e9                      # channel 6
DF = 312.5e3                      # subcarrier spacing
K = np.r_[np.arange(0, 32), np.arange(-32, 0)]          # ESP32 LLTF index order
GUARD = np.array([i == 0 or 27 <= i <= 37 for i in range(64)])
PEOPLE = {"low": (0, 1), "medium": (2, 3), "high": (4, 6)}


def make_session(rng):
    n_paths = rng.integers(4, 9)
    delays = rng.uniform(5e-9, 80e-9, n_paths)
    gains = rng.rayleigh(1.0, n_paths) * np.exp(1j * rng.uniform(0, 2 * np.pi, n_paths))
    f = FC + K * DF
    static = (gains[None, :] * np.exp(-2j * np.pi * f[:, None] * delays[None, :])).sum(1)
    return static / np.abs(static).mean()


def simulate(rng, static, label, seconds, rssi_base, rate=config.EXPECTED_RATE_HZ):
    lo, hi = PEOPLE[label]
    n_people = int(rng.integers(lo, hi + 1))
    f = FC + K * DF
    t = np.cumsum(rng.normal(1 / rate, 0.003, int(seconds * rate * 1.05)).clip(0.004))
    t = t[t < seconds]
    t = t[rng.random(len(t)) > 0.03]                      # 3% packet loss

    h = np.repeat(static[None, :], len(t), axis=0).astype(complex)
    block = np.ones(len(t))
    for _ in range(n_people):
        walking = rng.random() < 0.45
        d0 = rng.uniform(3, 9)
        amp = rng.uniform(0.08, 0.3)
        if walking:
            speed = rng.uniform(0.3, 1.0)
            period = rng.uniform(4, 10)
            d = d0 + speed * period / (2 * np.pi) * np.sin(2 * np.pi * t / period + rng.uniform(0, 6))
        else:
            fidget = rng.uniform(0.002, 0.02) * np.cumsum(rng.normal(0, 1, len(t))) / np.sqrt(rate)
            d = d0 + fidget
        d = d + 0.005 * np.sin(2 * np.pi * rng.uniform(0.2, 0.4) * t)   # breathing
        h += amp * np.exp(-2j * np.pi * f[None, :] * d[:, None] / C) * np.exp(1j * rng.uniform(0, 6))
        if rng.random() < 0.5:                                          # sometimes crosses LOS
            block *= 1 - rng.uniform(0.03, 0.12) * (0.5 + 0.5 * np.sin(2 * np.pi * t / rng.uniform(5, 15)))
    h *= block[:, None]
    h += (rng.normal(0, 0.03, h.shape) + 1j * rng.normal(0, 0.03, h.shape))
    agc = rng.normal(1, 0.04, len(t)) * np.where(rng.random(len(t)) < 0.002, 1.6, 1.0)
    h *= (rng.uniform(18, 32) * agc)[:, None]      # recording-level gain varies too
    spikes = rng.random(len(t)) < 0.005
    h[spikes] *= rng.uniform(0.2, 3.0, (spikes.sum(), 1))
    h[:, GUARD] = 0

    iq = np.empty((len(t), 128), np.int16)
    iq[:, 0::2] = np.clip(np.round(h.imag), -128, 127)
    iq[:, 1::2] = np.clip(np.round(h.real), -128, 127)
    # RSSI: dominated by room/distance (per session), drifts between recordings,
    # and only weakly affected by people blocking the line of sight.
    drift = rng.normal(0, 2.0)
    rssi = np.round(rssi_base + drift + 8 * np.log10(block) + rng.normal(0, 1.2, len(t))).astype(int)
    return t, iq, rssi, n_people


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    conditions = ["door_closed", "door_open", "chairs_moved", "evening"]
    for s in range(1, args.sessions + 1):
        session = f"SYN-s{s:02d}"
        static = make_session(rng)
        rssi_base = rng.uniform(-62, -42)
        for label in config.CLASSES:
            for r in range(args.per_class):
                t, iq, rssi, n = simulate(rng, static, label, args.seconds, rssi_base)
                ts_ms = t * 1000.0
                frames = [CSIFrame(ts_ms[i], int(rssi[i]), iq[i]) for i in range(len(t))]
                path = config.RAW_DIR / session / f"{session}__{label}__r{r + 1}.csv"
                meta = {"session": session, "label": label, "people": n, "room": "simulated",
                        "condition": conditions[(s - 1) % len(conditions)], "activity": "mixed",
                        "notes": "synthetic data - for software testing only",
                        "start": time.strftime("%Y-%m-%dT%H:%M:%S"), "duration_s": args.seconds,
                        "packets": len(t), "rate_hz": round(len(t) / args.seconds, 1),
                        "synthetic": True}
                save_recording(frames, list(time.time() + t), path, meta)
                print(f"{path.relative_to(config.ROOT)}  people={n}")


if __name__ == "__main__":
    main()
