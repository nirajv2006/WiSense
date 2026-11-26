"""Empty-room baseline features.

Most WiSense features measure *movement* (variance, rate of change, motion
bands). Objects that stand still, such as water bottles in the cardboard demo
room, barely move, but they do change the *shape* of the signal across the 52
subcarriers: water absorbs some frequencies more than others. These features
compare each window's subcarrier profile with the profile of the EMPTY room.

    base_dev        mean relative difference from the empty profile
                    (0 = looks exactly like the empty room)
    base_corr_dist  1 - correlation with the empty profile
                    (0 = same shape, larger = more reshaped)

Both use the AGC-normalised amplitude, so random gain changes of the ESP32 do
not count as a difference.

The baseline is the average profile of recordings labelled with 0 people /
bottles. During cross-validation it is computed from the training folds only,
so a test recording never sees its own data through the baseline.
"""
from __future__ import annotations

import numpy as np

BASELINE_FEATURES = ["base_dev", "base_corr_dist"]


def window_profile(p: dict) -> np.ndarray:
    """Mean AGC-normalised amplitude per subcarrier for one preprocessed window (52,)."""
    return p["amp_norm"].mean(axis=0)


def make_baseline(profiles) -> np.ndarray | None:
    """Average several empty-room profiles into one baseline. None if there are none."""
    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim != 2 or len(profiles) == 0:
        return None
    return profiles.mean(axis=0)


def baseline_features(profile: np.ndarray, baseline: np.ndarray | None) -> list[float]:
    """[base_dev, base_corr_dist] for one window. Zeros when there is no baseline."""
    if baseline is None:
        return [0.0, 0.0]
    profile = np.asarray(profile, dtype=float)
    dev = float(np.mean(np.abs(profile - baseline)) / (np.mean(np.abs(baseline)) + 1e-12))
    if profile.std() < 1e-12 or baseline.std() < 1e-12:
        corr = 1.0
    else:
        corr = float(np.corrcoef(profile, baseline)[0, 1])
    return [dev, 1.0 - corr]


def add_baseline_features(X: np.ndarray, profiles: np.ndarray, baseline: np.ndarray | None) -> np.ndarray:
    """Append the baseline columns to a feature matrix (one row per window)."""
    extra = np.array([baseline_features(p, baseline) for p in profiles]).reshape(len(X), -1)
    return np.hstack([X, extra])
