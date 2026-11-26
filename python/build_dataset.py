"""Turn every raw recording in data/raw into one feature table (data/features.csv).

    python build_dataset.py
    python build_dataset.py --include-synthetic      # also use simulate.py data

One row = one 4 s window. Columns: metadata (session, label, ...) + features.
"""
import argparse

import pandas as pd

from wisense import config
from wisense.csi_io import load_recording
from wisense.features import FEATURE_NAMES, iter_windows, window_features

META_COLS = ["recording", "session", "room", "condition", "activity", "label", "people",
             "synthetic", "window_start_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(config.RAW_DIR))
    ap.add_argument("--out", default=str(config.FEATURES_CSV))
    ap.add_argument("--include-synthetic", action="store_true")
    args = ap.parse_args()

    files = sorted(p for p in __import__("pathlib").Path(args.raw).rglob("*.csv"))
    if not files:
        raise SystemExit(f"No recordings in {args.raw}. Run collect.py (or simulate.py) first.")

    rows, summary = [], []
    for path in files:
        ts, rssi, iq, meta = load_recording(path)
        if meta.get("synthetic") and not args.include_synthetic:
            continue
        if meta.get("label") not in config.CLASSES:
            print(f"skip {path.name}: label {meta.get('label')!r} not in {config.CLASSES}")
            continue
        n = 0
        for s, e in iter_windows(ts, len(ts)):
            f = window_features(ts[s:e], iq[s:e], rssi[s:e])
            rows.append({
                "recording": path.stem, "session": meta.get("session", path.parent.name),
                "room": meta.get("room", "unknown"), "condition": meta.get("condition", "default"),
                "activity": meta.get("activity", "unknown"), "label": meta["label"],
                "people": meta.get("people"), "synthetic": bool(meta.get("synthetic", False)),
                "window_start_s": round((ts[s] - ts[0]) / 1000, 2), **f,
            })
            n += 1
        summary.append((path.stem, meta["label"], len(ts), n))

    if not rows:
        raise SystemExit("No usable windows. (Synthetic data needs --include-synthetic.)")
    df = pd.DataFrame(rows)[META_COLS + FEATURE_NAMES + ["pkt_rate"]]
    df.to_csv(args.out, index=False)

    print(f"\n{len(summary)} recordings -> {len(df)} windows -> {args.out}\n")
    print(df.groupby(["session", "label"]).size().unstack(fill_value=0).reindex(
        columns=config.CLASSES, fill_value=0).to_string())
    low_rate = df[df.pkt_rate < config.MIN_FILL * config.EXPECTED_RATE_HZ]
    if len(low_rate):
        print(f"\nNote: {len(low_rate)} windows have a low packet rate.")


if __name__ == "__main__":
    main()
