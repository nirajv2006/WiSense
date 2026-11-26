import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi_io import RAW_COLUMNS, amplitude, parse_csi_line, rx_seq_gap  # noqa: E402


def make_line(values, seq=1, rx_seq=10, first_invalid=0):
    return f"CSI,{seq},123456,{rx_seq},-40,1,0,{first_invalid},{len(values)}," + " ".join(map(str, values))


def test_parse_valid_line():
    f = parse_csi_line(make_line([0, 0, 3, 4, -6, 8]))
    assert f is not None
    assert (f.seq, f.rx_seq, f.rssi, f.len) == (1, 10, -40, 6)
    assert f.values == [0, 0, 3, 4, -6, 8]


def test_parse_rejects_truncated_and_garbage():
    good = make_line([1, 2, 3, 4])
    assert parse_csi_line(good[:-2]) is None          # len mismatch
    assert parse_csi_line("DATA,1,2,3") is None
    assert parse_csi_line("CSI,a,b") is None
    assert parse_csi_line(make_line([1, 2, 300, 4])) is None  # out of int8 range


def test_rx_seq_gap_wraps():
    assert rx_seq_gap(10, 11) == 0
    assert rx_seq_gap(10, 13) == 2
    assert rx_seq_gap(4095, 0) == 0
    assert rx_seq_gap(4094, 1) == 2
    assert rx_seq_gap(7, 7) == 0  # retransmission


def test_amplitude_matrix():
    # 3 subcarriers per frame: null, (imag=3, real=4) -> 5, (imag=-6, real=8) -> 10
    vals = [0, 0, 3, 4, -6, 8]
    rows = [[0.0, i, 0, i, -40, 1, 0, 0, 6, " ".join(map(str, vals))] for i in range(4)]
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    amp = amplitude(df)
    assert amp.shape == (4, 2)  # null subcarrier dropped
    np.testing.assert_allclose(amp[0], [5.0, 10.0])


def test_amplitude_drops_invalid_first_word():
    # first_invalid=1 -> first 2 subcarriers are hardware garbage and must vanish
    vals = [9, 9, 9, 9, 3, 4, -6, 8]
    rows = [[0.0, i, 0, i, -40, 1, 0, 1, 8, " ".join(map(str, vals))] for i in range(4)]
    amp = amplitude(pd.DataFrame(rows, columns=RAW_COLUMNS))
    assert amp.shape == (4, 2)
    assert not np.isnan(amp).any()


def test_gap_counter_separates_phy_modes_and_ignores_resets():
    from csi_io import GapCounter
    g = GapCounter()
    # interleaved non-HT (mode 0, seq ~2108) and HT (mode 1, seq ~2) frames
    for mode, seq in [(0, 2), (0, 3), (0, 2108), (0, 4), (1, 0), (1, 1), (1, 3), (1, 4)]:
        g.update(mode, seq)
    # mode 0: 2->3 ok, 3->2108 reset, 2108->4 reset; mode 1: one frame lost (2)
    assert g.gaps == 1
    assert g.resets == 2
