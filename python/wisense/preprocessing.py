"""Signal processing: raw I/Q -> clean, uniformly sampled amplitude matrix.

The same function (`preprocess_window`) is used for training and for live
inference, so the model always sees identically processed data.

Steps
-----
1. Amplitude      |H| = sqrt(I^2 + Q^2) on the 52 data subcarriers.
2. Resampling     packets arrive with jitter; interpolate onto a uniform
                  50 Hz grid so time/frequency features are meaningful.
3. Hampel filter  replace impulsive outliers (bad packets) with the local median.
4. Low-pass       4th-order Butterworth, 10 Hz, zero-phase - removes noise
                  faster than any human movement.
5. Normalisation  divide each packet by its mean amplitude. The ESP32's
                  automatic gain control rescales whole packets at random;
                  this removes that effect but keeps the *shape* changes
                  caused by people. (The raw level is kept as a separate feature.)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from . import config

_VALID = np.array(config.VALID_SUBCARRIERS)


def iq_to_amplitude(iq: np.ndarray) -> np.ndarray:
    """(N,128) int (imag, real) pairs -> (N,52) float amplitude."""
    iq = iq.astype(np.float32)
    imag, real = iq[:, 0::2], iq[:, 1::2]
    return np.sqrt(real ** 2 + imag ** 2)[:, _VALID]


def iq_to_phase(iq: np.ndarray) -> np.ndarray:
    """(N,52) raw phase in radians (not used by the default model; available for extensions)."""
    iq = iq.astype(np.float32)
    return np.arctan2(iq[:, 0::2], iq[:, 1::2])[:, _VALID]


def resample_uniform(ts_ms: np.ndarray, x: np.ndarray, fs: float = config.EXPECTED_RATE_HZ):
    """Linear interpolation of x (N, ...) onto a uniform time grid. Returns (t_s, x_uniform)."""
    t = (ts_ms - ts_ms[0]) / 1000.0
    t, idx = np.unique(t, return_index=True)          # drop duplicate timestamps
    x = x[idx]
    grid = np.arange(0, t[-1] + 1e-9, 1.0 / fs)
    if x.ndim == 1:
        return grid, np.interp(grid, t, x)
    out = np.empty((len(grid), x.shape[1]), dtype=np.float32)
    for j in range(x.shape[1]):
        out[:, j] = np.interp(grid, t, x[:, j])
    return grid, out


def hampel(x: np.ndarray, half_window: int = config.HAMPEL_HALF_WINDOW,
           n_sigma: float = config.HAMPEL_N_SIGMA) -> np.ndarray:
    """Hampel outlier filter along time (axis 0), applied to every subcarrier."""
    size = (2 * half_window + 1, 1) if x.ndim == 2 else 2 * half_window + 1
    med = median_filter(x, size=size, mode="nearest")
    mad = 1.4826 * median_filter(np.abs(x - med), size=size, mode="nearest")
    out = x.copy()
    mask = np.abs(x - med) > n_sigma * np.maximum(mad, 1e-6)
    out[mask] = med[mask]
    return out


_SOS = butter(config.LOWPASS_ORDER, config.LOWPASS_CUTOFF_HZ,
              btype="low", fs=config.EXPECTED_RATE_HZ, output="sos")


def lowpass(x: np.ndarray) -> np.ndarray:
    if len(x) <= 27:                                   # too short for zero-phase filtering
        return x
    return sosfiltfilt(_SOS, x, axis=0).astype(np.float32)


def normalise_packets(amp: np.ndarray) -> np.ndarray:
    m = amp.mean(axis=1, keepdims=True)
    return amp / np.where(m == 0, 1, m)


def preprocess_window(ts_ms: np.ndarray, iq: np.ndarray, rssi: np.ndarray) -> dict:
    """Full chain for one window. Returns a dict with everything features need."""
    amp_raw = iq_to_amplitude(iq)
    _, amp_u = resample_uniform(ts_ms, amp_raw)
    _, rssi_u = resample_uniform(ts_ms, rssi.astype(np.float32))
    amp_f = lowpass(hampel(amp_u))
    amp_f = np.clip(amp_f, 0, None)
    return {
        "amp": amp_f,                                  # filtered amplitude (T,52)
        "amp_norm": normalise_packets(amp_f),          # AGC-compensated (T,52)
        "rssi": rssi_u,
        "n_packets": len(ts_ms),
        "duration_s": (ts_ms[-1] - ts_ms[0]) / 1000.0,
    }
