"""Unit tests for the WiSense pipeline.   Run:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wisense import config
from wisense.csi_io import parse_line
from wisense.features import FEATURE_NAMES, window_features
from wisense.models import VarianceThreshold
from wisense.power import PowerAdvisor
from wisense.preprocessing import hampel, iq_to_amplitude

RNG = np.random.default_rng(0)


def fake_iq(n=200):
    iq = RNG.integers(-30, 30, (n, 128)).astype(np.int16)
    ts = np.arange(n) * 20.0
    return ts, iq, np.full(n, -50.0)


def test_parse_wisense_line():
    vals = " ".join(str(v) for v in range(128))
    f = parse_line(f"CSI,1234,-47,128,{vals}\r\n")
    assert f.ts_ms == 1234 and f.rssi == -47 and f.iq.shape == (128,) and f.iq[5] == 5


def test_parse_esp_csi_line():
    vals = ",".join(str(v % 20) for v in range(128))
    head = "CSI_DATA,7,aa:bb:cc:dd:ee:ff,-55" + ",0" * 14 + ",5000000" + ",0" * 5
    f = parse_line(f'{head},"[{vals}]"')
    assert f.rssi == -55 and f.iq.shape == (128,) and f.ts_ms == 5000.0


def test_non_csi_lines_ignored():
    for line in ["# rate 50", "DATA,1,2,3,4,5,Occupied", "", "CSI,garbage"]:
        assert parse_line(line) is None


def test_amplitude_uses_52_subcarriers():
    iq = np.zeros((1, 128), np.int16)
    iq[0, 0::2], iq[0, 1::2] = 3, 4            # imag=3, real=4 -> |H|=5
    amp = iq_to_amplitude(iq)
    assert amp.shape == (1, len(config.VALID_SUBCARRIERS)) == (1, 52)
    assert np.allclose(amp, 5)


def test_hampel_removes_spike():
    x = np.ones((50, 2))
    x[25, 0] = 100
    assert hampel(x)[25, 0] == 1


def test_features_finite_and_complete():
    ts, iq, rssi = fake_iq()
    f = window_features(ts, iq, rssi)
    assert all(k in f for k in FEATURE_NAMES)
    assert all(np.isfinite(f[k]) for k in FEATURE_NAMES)


def test_advisor_steps_up_fast_and_down_slow():
    adv = PowerAdvisor(level="STANDARD")
    for t in range(0, 4):
        assert adv.update("high", 0.9, t).level == "STANDARD"
    assert adv.update("high", 0.9, config.STEP_UP_AFTER_S).level == "FULL"
    for t in range(10, 10 + config.STEP_DOWN_AFTER_S):
        assert adv.update("low", 0.9, t).level == "FULL"
    assert adv.update("low", 0.9, 10 + config.STEP_DOWN_AFTER_S).level == "ECO"


def test_advisor_holds_when_unsure():
    adv = PowerAdvisor(level="ECO")
    for t in range(100):
        assert adv.update("high", config.MIN_CONFIDENCE - 0.1, t).level == "ECO"


def test_variance_threshold_learns_ordering():
    x = np.concatenate([RNG.normal(1, 0.1, 50), RNG.normal(5, 0.1, 50), RNG.normal(9, 0.1, 50)])
    X = np.zeros((150, len(FEATURE_NAMES)))
    X[:, FEATURE_NAMES.index("meanamp_var")] = x
    y = np.repeat(config.CLASSES, 50)
    assert (VarianceThreshold().fit(X, y).predict(X) == y).mean() > 0.95
