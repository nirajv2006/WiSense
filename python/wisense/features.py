"""Feature engineering: one fixed-length feature vector per 4 s CSI window.

Features are grouped so they can be compared in an ablation study:

LEVEL       absolute signal level. Depends on room, distance and antenna
            orientation, so it can help in a known room but generalises poorly.
VARIATION   how much the channel changes over time. People move and breathe,
            the empty room does not - this is the core occupancy signal.
SPECTRAL    *how fast* the channel changes (breathing vs walking vs noise).
STRUCTURE   how the disturbance is spread across subcarriers.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from . import config
from .preprocessing import preprocess_window

FEATURE_GROUPS = {
    "level": ["amp_mean", "rssi_mean"],
    "variation": [
        "meanamp_var",          # variance of per-packet mean amplitude (= current firmware feature)
        "sc_std_mean",          # mean temporal std per subcarrier (normalised)
        "sc_var_mean",          # mean temporal variance per subcarrier (normalised)
        "sc_std_p90",           # 90th percentile of per-subcarrier std
        "sc_iqr_mean",          # robust spread over time
        "sc_range_mean",        # max - min over time
        "cv_mean",              # coefficient of variation (raw amplitude)
        "diff_abs_mean",        # mean |a[t] - a[t-1]| = rate of change
        "rssi_std",
    ],
    "spectral": ["pc1_std", "band_breath", "band_motion", "band_fast", "spec_centroid"],
    "structure": ["pc1_ratio", "sc_std_dispersion", "adj_corr_mean", "skew_mean", "kurt_mean"],
}
FEATURE_NAMES = [f for group in FEATURE_GROUPS.values() for f in group]


def _band_power(sig: np.ndarray, fs: float) -> tuple[dict, float]:
    sig = sig - sig.mean()
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
    freqs = np.fft.rfftfreq(len(sig), 1 / fs)
    total = spec[freqs >= 0.1].sum() + 1e-12
    bands = {name: float(spec[(freqs >= lo) & (freqs < hi)].sum() / total)
             for name, (lo, hi) in config.BANDS.items()}
    sel = freqs >= 0.1
    centroid = float((freqs[sel] * spec[sel]).sum() / total)
    return bands, centroid


def compute_features(p: dict) -> dict:
    """p = output of preprocess_window(). Returns {feature_name: value}."""
    amp, an, rssi = p["amp"], p["amp_norm"], p["rssi"]
    fs = config.EXPECTED_RATE_HZ

    sc_std = an.std(axis=0)
    q75, q25 = np.percentile(an, [75, 25], axis=0)
    centered = an - an.mean(axis=0)

    # Principal component analysis over subcarriers: the first component
    # captures the dominant, correlated change caused by a moving body.
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = centered @ vt[0]
        pc1_ratio = float(s[0] ** 2 / (np.sum(s ** 2) + 1e-12))
    except np.linalg.LinAlgError:
        pc1, pc1_ratio = centered.mean(axis=1), 0.0
    bands, centroid = _band_power(pc1, fs)

    # Correlation over time between neighbouring subcarriers
    a, b = centered[:, :-1], centered[:, 1:]
    denom = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-12
    adj_corr = float(np.mean((a * b).sum(0) / denom))

    with np.errstate(all="ignore"):
        sk = np.nan_to_num(skew(an, axis=0)).mean()
        ku = np.nan_to_num(kurtosis(an, axis=0)).mean()

    mean_amp_series = amp.mean(axis=1)
    raw_mean = amp.mean(axis=0)

    f = {
        "amp_mean": float(amp.mean()),
        "rssi_mean": float(rssi.mean()),
        "meanamp_var": float(mean_amp_series.var()),
        "sc_std_mean": float(sc_std.mean()),
        "sc_var_mean": float((sc_std ** 2).mean()),
        "sc_std_p90": float(np.percentile(sc_std, 90)),
        "sc_iqr_mean": float((q75 - q25).mean()),
        "sc_range_mean": float((an.max(0) - an.min(0)).mean()),
        "cv_mean": float((amp.std(axis=0) / np.maximum(raw_mean, 1e-6)).mean()),
        "diff_abs_mean": float(np.abs(np.diff(an, axis=0)).mean()),
        "rssi_std": float(rssi.std()),
        "pc1_std": float(pc1.std()),
        "band_breath": bands["breath"],
        "band_motion": bands["motion"],
        "band_fast": bands["fast"],
        "spec_centroid": centroid,
        "pc1_ratio": pc1_ratio,
        "sc_std_dispersion": float(sc_std.std() / (sc_std.mean() + 1e-12)),
        "adj_corr_mean": adj_corr,
        "skew_mean": float(sk),
        "kurt_mean": float(ku),
    }
    return {k: f[k] for k in FEATURE_NAMES}


def iter_windows(ts_ms: np.ndarray, n: int):
    """Yield (start, end) index pairs of sliding windows defined in *time*, not packet count."""
    win_ms, step_ms = config.WINDOW_S * 1000, config.STEP_S * 1000
    min_packets = int(config.MIN_FILL * config.WINDOW_S * config.EXPECTED_RATE_HZ)
    t = ts_ms[0]
    while t + win_ms <= ts_ms[-1]:
        s = np.searchsorted(ts_ms, t)
        e = np.searchsorted(ts_ms, t + win_ms)
        if e - s >= min_packets:
            yield s, e
        t += step_ms


def window_features(ts_ms, iq, rssi) -> dict:
    p = preprocess_window(ts_ms, iq, rssi)
    f = compute_features(p)
    f["pkt_rate"] = p["n_packets"] / max(p["duration_s"], 1e-3)   # quality check, not a model input
    return f
