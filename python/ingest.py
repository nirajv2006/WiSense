"""Live inference service: CSI stream -> features -> model -> power advice -> SQLite.

Run this in one terminal and the dashboard in another:

    python ingest.py --port /dev/cu.usbserial-0001          # real receiver
    python ingest.py --replay "data/raw/s01/*.csv" --loop   # demo from a recording
    python ingest.py --port ... --no-model                  # signal only (before training)

Every STEP_S seconds it classifies the last WINDOW_S seconds of CSI, smooths
the class probabilities, updates the power recommendation, stores one row
in data/wisense.db, and sends "S,<label>,<level>" back to the receiver's OLED.
"""
import argparse
import collections
import glob
import time

import joblib
import numpy as np

from wisense import config, storage
from wisense.csi_io import ReplaySource, SerialSource
from wisense.features import FEATURE_NAMES, compute_features
from wisense.power import PowerAdvisor
from wisense.preprocessing import preprocess_window


def load_model(path):
    try:
        bundle = joblib.load(path)
    except FileNotFoundError:
        return None, None
    meta = bundle["meta"]
    if meta["feature_names"] != FEATURE_NAMES:
        raise SystemExit("Model was trained with a different feature set. Re-run train.py.")
    if meta.get("synthetic"):
        print("NOTE: this model was trained on SYNTHETIC data - predictions on real CSI are meaningless.")
    return bundle["model"], meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="receiver serial port")
    src.add_argument("--replay", help="glob of recorded CSV files to play back")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = real time)")
    ap.add_argument("--loop", action="store_true", help="loop the replay forever")
    ap.add_argument("--model", default=str(config.MODEL_PATH))
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--reset-db", action="store_true", help="clear previous history first")
    args = ap.parse_args()

    model, meta = (None, None) if args.no_model else load_model(args.model)
    if model is None and not args.no_model:
        print(f"No model at {args.model} - running in signal-only mode. Train one with train.py.")
    if args.port:
        source, source_name = SerialSource(args.port), args.port
    else:
        files = sorted(glob.glob(args.replay))
        if not files:
            raise SystemExit(f"No files match {args.replay}")
        source, source_name = ReplaySource(files, args.speed, args.loop), f"replay:{args.replay}"

    con = storage.connect()
    if args.reset_db:
        con.execute("DELETE FROM predictions")
        con.commit()

    win_ms, step_ms = config.WINDOW_S * 1000, config.STEP_S * 1000
    min_packets = int(config.MIN_FILL * config.WINDOW_S * config.EXPECTED_RATE_HZ)
    buf = collections.deque(maxlen=int(config.WINDOW_S * config.EXPECTED_RATE_HZ * 3))
    probas = collections.deque(maxlen=config.PROBA_SMOOTHING)
    advisor = PowerAdvisor()
    next_eval, last_ts = None, None
    print(f"Listening on {source_name}. Dashboard: streamlit run dashboard.py")

    try:
        for fr in source:
            if last_ts is not None and fr.ts_ms + 5000 < last_ts:     # device reboot / replay restart
                print("Timestamp jumped back (receiver restarted?) - resetting buffers")
                buf.clear()
                probas.clear()
                advisor.candidate = None
                next_eval = None
            last_ts = fr.ts_ms
            buf.append(fr)
            if next_eval is None:
                next_eval = fr.ts_ms + win_ms
            if fr.ts_ms < next_eval:
                continue
            next_eval += step_ms
            if next_eval < fr.ts_ms:                 # fell behind: skip ahead
                next_eval = fr.ts_ms + step_ms

            frames = [f for f in buf if f.ts_ms > fr.ts_ms - win_ms]
            if len(frames) < min_packets:
                print(f"waiting for packets ({len(frames)}/{min_packets} in window)")
                continue
            ts = np.array([f.ts_ms for f in frames])
            iq = np.vstack([f.iq for f in frames])
            rssi = np.array([f.rssi for f in frames], float)
            p = preprocess_window(ts, iq, rssi)
            feats = compute_features(p)
            rate = len(frames) / max((ts[-1] - ts[0]) / 1000, 1e-3)
            now = time.time()
            storage.write_snapshot(p["amp"], p["amp"].mean(axis=1), now)

            row = {"ts": now, "motion": feats["sc_std_mean"], "rssi": float(rssi.mean()),
                   "pkt_rate": rate, "source": source_name}
            if model is not None:
                x = np.array([[feats[k] for k in FEATURE_NAMES]])
                pr = model.predict_proba(x)[0]
                pr = np.array([pr[list(model.classes_).index(c)] if c in model.classes_ else 0.0
                               for c in config.CLASSES])
                raw_label = config.CLASSES[int(pr.argmax())]
                probas.append(pr)
                sm = np.mean(probas, axis=0)
                label, conf = config.CLASSES[int(sm.argmax())], float(sm.max())
                rec = advisor.update(label, conf, fr.ts_ms / 1000.0)
                row.update(raw_label=raw_label, label=label, confidence=conf,
                           p_low=float(sm[0]), p_medium=float(sm[1]), p_high=float(sm[2]),
                           power_level=rec.level, power_share=rec.share, reason=rec.reason)
                source.send(f"S,{label},{rec.level}")
                print(f"{time.strftime('%H:%M:%S')}  {label.upper():6s} ({conf:4.0%})  "
                      f"power {rec.level:8s}  {rate:4.0f} pkt/s  {rec.reason}")
            else:
                print(f"{time.strftime('%H:%M:%S')}  motion {feats['sc_std_mean']:.4f}  "
                      f"RSSI {rssi.mean():.0f}  {rate:.0f} pkt/s")
            storage.insert(con, row)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
