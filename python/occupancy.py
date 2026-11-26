"""WiSense occupancy tool: record -> train -> live, in one program.

STEP 1  Record labelled 15-second CSI clips (at least 10 each of low / medium / high)

    python occupancy.py record --port /dev/cu.usbserial-0001

    Press Enter, keep the room in one state for 15 s, then type what it was
    (l = low, m = medium, h = high). The tool keeps count and tells you when
    every class has the minimum 10 recordings.

STEP 2  Train the model

    python occupancy.py train

STEP 3  Live, near-instant occupancy + power control

    python occupancy.py live --port /dev/cu.usbserial-0001

    Every 0.5 s it classifies the last 2 s of CSI as LOW / MEDIUM / HIGH and
    sends the matching power mode to the receiver (LOW -> ECO, MEDIUM ->
    STANDARD, HIGH -> FULL). The receiver firmware then lowers its outputs.

Other commands
    python occupancy.py status                      # how many recordings per class
    python occupancy.py live --replay "data/raw/rec-*/*.csv"   # test without boards

Recordings go to data/raw/<session>/ in the same format as collect.py, so the
full research pipeline (build_dataset.py, train.py) can use them as well.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from wisense import config, storage
from wisense.baseline import BASELINE_FEATURES, add_baseline_features, make_baseline, window_profile
from wisense.csi_io import CSIFrame, ReplaySource, SerialSource, load_recording, parse_line, save_recording
from wisense.features import FEATURE_NAMES, compute_features
from wisense.preprocessing import preprocess_window

# ── Settings for this tool ───────────────────────────────────────────────────
RECORD_SECONDS = 15            # length of one labelled clip
MIN_PER_CLASS = 10             # recordings needed per class before training
LIVE_WINDOW_S = 2.0            # CSI used for one live decision (short = fast reaction)
LIVE_STEP_S = 0.5              # a new decision every 0.5 s
LIVE_SMOOTH = 2                # average probabilities over the last N decisions (~1 s)
MIN_FILL = 0.6                 # need >= 60 % of the expected packets in a window
MODEL_PATH = config.MODELS_DIR / "instant_model.joblib"
ALL_FEATURES = FEATURE_NAMES + BASELINE_FEATURES   # model inputs, in column order

LEVEL_FOR = {c: config.POWER_LEVELS[c]["level"] for c in config.CLASSES}   # low->ECO ...
SHARE_FOR = {config.POWER_LEVELS[c]["level"]: config.POWER_LEVELS[c]["share"] for c in config.CLASSES}
KEYS = {"l": "low", "low": "low", "m": "medium", "medium": "medium", "h": "high", "high": "high"}


# ═════════════════════════════════════════════════════════════════════════════
# Recording
# ═════════════════════════════════════════════════════════════════════════════
def list_recordings(include_synthetic: bool = False):
    """[(path, meta)] for every labelled recording under data/raw."""
    out = []
    for p in sorted(config.RAW_DIR.rglob("*.csv")):
        mp = p.with_suffix(".json")
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        if meta.get("label") not in config.CLASSES:
            continue
        if meta.get("synthetic") and not include_synthetic:
            continue
        out.append((p, meta))
    return out


def tally(include_synthetic: bool = False) -> dict:
    counts = {c: 0 for c in config.CLASSES}
    for _, meta in list_recordings(include_synthetic):
        counts[meta["label"]] += 1
    return counts


def print_tally(counts: dict, target: int = MIN_PER_CLASS) -> None:
    print("\n  Recordings so far")
    for c in config.CLASSES:
        n = counts[c]
        bar = "#" * min(n, target) + "." * max(target - n, 0)
        extra = f" +{n - target}" if n > target else ""
        mark = "OK" if n >= target else f"need {target - n} more"
        print(f"    {c:<7s} [{bar}]{extra}  {n:3d}   {mark}   ({config.CLASS_DEFINITIONS[c]})")
    print()


def capture(src, seconds: float):
    """Read CSI frames from the board for `seconds`. Shows a progress bar."""
    if hasattr(src, "ser"):
        src.ser.reset_input_buffer()           # drop data that piled up while waiting
    frames, host_times = [], []
    t0 = time.time()
    last_draw = 0.0
    for fr in src:
        now = time.time()
        frames.append(fr)
        host_times.append(now)
        el = now - t0
        if now - last_draw > 0.25:
            last_draw = now
            done = int(30 * min(el / seconds, 1))
            rate = len(frames) / max(el, 1e-3)
            sys.stdout.write(f"\r  REC [{'=' * done}{' ' * (30 - done)}] {el:4.1f}/{seconds:.0f}s "
                             f"{len(frames):5d} pkts {rate:4.0f}/s RSSI {fr.rssi:4d}")
            sys.stdout.flush()
        if el >= seconds:
            break
    print()
    return frames, host_times


class SimulatedBoard:
    """Stand-in for the receiver (synthetic CSI) so the recording flow can be tried
    without hardware. Data from it is marked synthetic and never used by default."""

    def __init__(self, seed: int = 1):
        import simulate                                   # the repo's CSI simulator
        self.sim = simulate
        self.rng = np.random.default_rng(seed)
        self.static = simulate.make_session(self.rng)
        self.next_label = "low"

    def __iter__(self):
        t, iq, rssi, _ = self.sim.simulate(self.rng, self.static, self.next_label, 120, -50)
        start = time.time()
        for i in range(len(t)):
            delay = start + t[i] - time.time()
            if delay > 0:
                time.sleep(delay)
            yield CSIFrame(t[i] * 1000.0, int(rssi[i]), iq[i])


def suggest_next(counts: dict, recent: list[str]) -> str:
    """Class to record next: the one with the fewest clips, avoiding the class just recorded.

    Recording 10 low, then 10 medium, then 10 high ties each class to a stretch of
    time, and the boards slowly drift (warm-up, nearby Wi-Fi). The model can then
    learn 'early vs late' instead of occupancy. Mixing the order prevents that."""
    order = sorted(config.CLASSES, key=lambda c: (counts[c], recent[-1:] == [c]))
    if recent and order[0] == recent[-1] and len(order) > 1:
        order = order[1:] + order[:1]
    return order[0]


def wait_for_stream(src, min_rate: float = 40, timeout: float = 90) -> bool:
    """Block until CSI arrives at >= min_rate packets/s. Opening the port resets the
    receiver, and the transmitter then needs a few seconds to rejoin."""
    ser, t0 = src.ser, time.time()
    bucket, n = t0, 0
    while time.time() - t0 < timeout:
        raw = ser.readline()
        if raw and parse_line(raw.decode(errors="ignore")) is not None:
            n += 1
        now = time.time()
        if now - bucket >= 1.0:
            rate = n / (now - bucket)
            sys.stdout.write(f"\r  checking CSI stream: {rate:3.0f} pkt/s (need {min_rate:.0f}) "
                             f"{now - t0:3.0f}s ")
            sys.stdout.flush()
            if rate >= min_rate:
                print(" - ok")
                return True
            bucket, n = now, 0
    print(f"\n  No steady CSI after {timeout:.0f} s. Is the transmitter powered? "
          "Try resetting it. Is a serial monitor holding the port?")
    return False


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


def cmd_record(args):
    session = args.session or f"rec-{datetime.now():%Y%m%d}"
    synthetic = args.simulate
    if synthetic:
        src, port_name = SimulatedBoard(), "simulated"
        print("SIMULATION MODE - synthetic CSI, saved as synthetic (ignored by train unless --include-synthetic).")
    else:
        if not args.port:
            raise SystemExit("Give the receiver port, e.g. --port /dev/cu.usbserial-0001 (or --simulate).")
        src, port_name = SerialSource(args.port), args.port

    print(f"Session '{session}'. Each recording lasts {args.seconds} s.")
    print("Before each one: put the room in ONE occupancy state and keep it that way until it ends.")
    print("Mix the classes (l, m, h, m, l, ...) rather than recording all of one class in a row.")
    recent: list[str] = []                                  # labels saved in this sitting, in order
    while True:
        counts = tally(include_synthetic=synthetic)
        print_tally(counts, args.target)
        nxt = suggest_next(counts, recent)
        print(f"  Suggested next: '{nxt[0]}' ({nxt}, {config.CLASS_DEFINITIONS[nxt]})")
        if all(n >= args.target for n in counts.values()):
            print(f"  Every class has at least {args.target} recordings - you can train now "
                  f"('t'). More recordings (different times, positions, activities) still help.")
        hint = "/".join(f"{c[0]}" for c in config.CLASSES)
        cmd = ask(f"[Enter] record   [{hint}] record with label set in advance   "
                  "[t] train   [q] quit > ")
        if cmd == "q":
            break
        if cmd == "t":
            cmd_train(argparse.Namespace(include_synthetic=synthetic, force=False, out=str(MODEL_PATH),
                                         window=LIVE_WINDOW_S))
            continue
        preset = KEYS.get(cmd)
        if cmd and preset is None:
            print("  ? unknown key")
            continue
        if synthetic:                                      # the simulator needs to know what to fake
            src.next_label = preset or ["low", "medium", "high"][int(np.random.randint(3))]

        if len(recent) >= 2 and preset and recent[-2:] == [preset, preset]:
            print(f"  NOTE: that would be 3 '{preset}' clips in a row. Mixing classes avoids "
                  "the model learning time drift instead of occupancy.")
        if not synthetic and not wait_for_stream(src):
            continue
        for i in (3, 2, 1):
            sys.stdout.write(f"\r  starting in {i} ...")
            sys.stdout.flush()
            time.sleep(1)
        print("\r  recording - hold the occupancy steady")
        frames, host_times = capture(src, args.seconds)
        if len(frames) < 10:
            print("  No CSI received. Is the transmitter on? Is the PlatformIO serial monitor closed?")
            continue
        rate = len(frames) / max(host_times[-1] - host_times[0], 1e-3)
        if rate < MIN_FILL * config.EXPECTED_RATE_HZ:
            print(f"  WARNING: only {rate:.0f} packets/s (expected ~{config.EXPECTED_RATE_HZ}). "
                  "Move the boards closer or check interference.")

        label = preset
        while label is None:
            a = ask("  What occupancy was it?  [l]ow  [m]edium  [h]igh   [d]iscard > ")
            if a in ("d", "discard"):
                label = "discard"
            else:
                label = KEYS.get(a)
        if label == "discard":
            print("  discarded.")
            continue
        people = ask("  How many people / bottles? (0 = empty, Enter to skip) > ")
        people = int(people) if people.isdigit() else None
        if people is not None:
            lo, hi = config.CLASS_PEOPLE_RANGE[label]
            if people < lo or (hi is not None and people > hi):
                print(f"  NOTE: {people} people is outside the usual '{label}' range "
                      f"({config.CLASS_DEFINITIONS[label]}). Saved anyway - check it is right.")

        stamp = datetime.now()
        path = config.RAW_DIR / session / f"{session}__{label}__{stamp:%Y%m%d_%H%M%S}.csv"
        meta = {"session": session, "label": label, "people": people, "room": args.room,
                "condition": args.condition, "activity": "mixed", "notes": "occupancy.py record",
                "start": stamp.isoformat(timespec="seconds"),
                "duration_s": round(host_times[-1] - host_times[0], 1),
                "packets": len(frames), "rate_hz": round(rate, 1),
                "synthetic": synthetic, "source": port_name}
        save_recording(frames, host_times, path, meta)
        recent.append(label)
        print(f"  saved {label.upper()} ({len(frames)} packets) -> {path.relative_to(config.ROOT)}")


def cmd_status(args):
    counts = tally(args.include_synthetic)
    print_tally(counts)
    print(f"  Model: {MODEL_PATH.relative_to(config.ROOT)} "
          f"{'(trained ' + time.ctime(MODEL_PATH.stat().st_mtime) + ')' if MODEL_PATH.exists() else '(not trained yet)'}")


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════
def windows_of(ts, window_s=LIVE_WINDOW_S, step_s=LIVE_STEP_S):
    min_packets = int(MIN_FILL * window_s * config.EXPECTED_RATE_HZ)
    t = ts[0]
    while t + window_s * 1000 <= ts[-1] + 1:
        s, e = np.searchsorted(ts, t), np.searchsorted(ts, t + window_s * 1000)
        if e - s >= min_packets:
            yield s, e
        t += step_s * 1000


def featurize(ts, iq, rssi) -> tuple[list[float], np.ndarray]:
    """Base features of one window, plus its subcarrier profile (for the baseline features)."""
    p = preprocess_window(ts, iq, rssi)
    f = compute_features(p)
    return [f[k] for k in FEATURE_NAMES], window_profile(p)


def pick_baseline(P, empty, low, idx):
    """Empty-room baseline from the windows in `idx` (training data only).

    Uses recordings typed as 0 people/bottles; if there are none, falls back to
    all 'low' recordings (less precise). Returns (baseline, source_description)."""
    sel = idx[empty[idx]]
    if len(sel):
        return make_baseline(P[sel]), f"{len(sel)} windows from empty (0) recordings"
    sel = idx[low[idx]]
    if len(sel):
        return make_baseline(P[sel]), f"{len(sel)} windows from 'low' recordings (no empty ones typed as 0)"
    return None, "none"


def cmd_train(args):
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    recs = list_recordings(args.include_synthetic)
    counts = {c: sum(m["label"] == c for _, m in recs) for c in config.CLASSES}
    print_tally(counts)
    short = [c for c, n in counts.items() if n < MIN_PER_CLASS]
    if short and not args.force:
        print(f"  Not enough data yet: {', '.join(short)} need {MIN_PER_CLASS} recordings each. "
              "Keep recording (or use --force to train anyway).")
        return
    if any(n == 0 for n in counts.values()):
        print("  Every class needs at least one recording.")
        return

    win_s = args.window
    print(f"  Extracting features ({win_s:g} s windows every {LIVE_STEP_S} s) ...")
    X, P, y, g, empty = [], [], [], [], []
    for i, (p, meta) in enumerate(recs):
        ts, rssi, iq, _ = load_recording(p)
        for s, e in windows_of(ts, win_s):
            feats, prof = featurize(ts[s:e], iq[s:e], rssi[s:e])
            X.append(feats)
            P.append(prof)
            y.append(meta["label"])
            g.append(i)
            empty.append(meta.get("people") == 0)
    X, P, y, g, empty = np.array(X), np.array(P), np.array(y), np.array(g), np.array(empty)
    X, P = np.nan_to_num(X), np.nan_to_num(P)
    low = y == config.CLASSES[0]
    n_empty = len({int(i) for i in g[empty]})
    print(f"  {len(recs)} recordings -> {len(X)} windows "
          f"({n_empty} empty recordings typed as 0, used for the empty-room baseline)")
    if n_empty == 0:
        print("  TIP: record a few clips of the EMPTY box and type 0 for the count. "
              "That gives a clean empty-room baseline for the static-bottle features.")

    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                                   class_weight="balanced", n_jobs=-1, random_state=0)

    # Honest check: whole recordings are held out, so the model is tested on clips it never saw.
    n_splits = min(5, min(counts.values()))
    cv = None
    if n_splits >= 2:
        pred = np.empty(len(y), dtype=object)
        for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=0).split(X, y, g):
            base, _ = pick_baseline(P, empty, low, tr)          # training folds only: no leakage
            Xf = add_baseline_features(X, P, base)
            pred[te] = clone(model).fit(Xf[tr], y[tr]).predict(Xf[te])
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro", labels=config.CLASSES)
        cm = confusion_matrix(y, pred, labels=config.CLASSES)
        # per-recording vote = what you get by watching the live output for a few seconds
        votes = [collections.Counter(pred[g == i]).most_common(1)[0][0] == y[g == i][0]
                 for i in np.unique(g)]
        cv = {"window_accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4),
              "recording_accuracy": round(float(np.mean(votes)), 4), "folds": n_splits,
              "confusion": cm.tolist()}
        print(f"\n  Test on unseen recordings ({n_splits}-fold):")
        print(f"    accuracy per {win_s:g}-s decision   : {acc:6.1%}")
        print(f"    macro F1                    : {f1:6.1%}")
        print(f"    whole-recording vote        : {np.mean(votes):6.1%}")
        print("\n    confusion (rows = true, cols = predicted)")
        print("             " + "".join(f"{c:>8s}" for c in config.CLASSES))
        for c, row in zip(config.CLASSES, cm):
            print(f"    {c:>8s} " + "".join(f"{v:8d}" for v in row))
        if acc < 0.7:
            print("\n    Accuracy is low. Record more varied clips, keep boards fixed in place, "
                  "and make sure each clip really matches its label.")

    baseline, base_src = pick_baseline(P, empty, low, np.arange(len(X)))
    model.fit(add_baseline_features(X, P, baseline), y)
    print(f"\n  Empty-room baseline: {base_src}")
    importances = sorted(zip(ALL_FEATURES, model.feature_importances_), key=lambda t: -t[1])
    print("  Most useful features: " + ", ".join(f"{n} ({v:.0%})" for n, v in importances[:5]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"feature_names": ALL_FEATURES, "baseline_source": base_src, "classes": config.CLASSES, "window_s": win_s,
            "step_s": LIVE_STEP_S, "counts": counts, "n_windows": int(len(X)), "cv": cv,
            "synthetic": bool(args.include_synthetic and any(m.get("synthetic") for _, m in recs)),
            "trained": datetime.now().isoformat(timespec="seconds")}
    joblib.dump({"model": model, "meta": meta, "baseline": baseline}, out)
    print(f"\n  Model saved -> {out}")
    print("  Next: python occupancy.py live --port <your port>")


# ═════════════════════════════════════════════════════════════════════════════
# Live
# ═════════════════════════════════════════════════════════════════════════════
class PowerControl:
    """Occupancy -> power mode, tuned for fast reaction.

    * Going UP (more people) happens on the first confident decision - nobody waits.
    * Going DOWN waits `down_hold_s` of consistent lower occupancy, so one
      noisy half-second cannot switch things off. Use 0 for fully instant.
    * Below `min_conf` the current mode is kept.
    """

    ORDER = config.POWER_ORDER                           # ECO, STANDARD, FULL

    def __init__(self, down_hold_s=3.0, min_conf=0.5, start="STANDARD"):
        self.level, self.down_hold_s, self.min_conf = start, down_hold_s, min_conf
        self.lower_since = None

    def update(self, label, conf, now):
        target = LEVEL_FOR[label]
        if conf < self.min_conf:
            return self.level, f"unsure ({conf:.0%}), holding"
        ti, ci = self.ORDER.index(target), self.ORDER.index(self.level)
        if ti > ci:
            self.level, self.lower_since = target, None
            return self.level, "stepped up"
        if ti == ci:
            self.lower_since = None
            return self.level, ""
        self.lower_since = self.lower_since or now
        if now - self.lower_since >= self.down_hold_s:
            self.level, self.lower_since = target, None
            return self.level, "stepped down"
        return self.level, f"-> {target} in {self.down_hold_s - (now - self.lower_since):.1f}s"


COLOR = {"low": "\033[32m", "medium": "\033[33m", "high": "\033[31m"}


def cmd_live(args):
    try:
        bundle = joblib.load(args.model)
    except FileNotFoundError:
        raise SystemExit(f"No model at {args.model}. Record data and run 'python occupancy.py train' first.")
    model, meta = bundle["model"], bundle["meta"]
    if meta["feature_names"] != ALL_FEATURES:
        raise SystemExit("Model uses a different feature set - retrain with 'python occupancy.py train'.")
    if meta.get("synthetic"):
        print("NOTE: model trained on SYNTHETIC data - only for testing the software.")
    win_s, step_s = meta.get("window_s", LIVE_WINDOW_S), args.step or meta.get("step_s", LIVE_STEP_S)
    baseline = bundle.get("baseline")
    classes = list(model.classes_)

    if args.port:
        src, name = SerialSource(args.port), args.port
    else:
        files = sorted(glob.glob(args.replay))
        if not files:
            raise SystemExit(f"No files match {args.replay}")
        src, name = ReplaySource(files, args.speed, args.loop), "replay"

    con = None if args.no_db else storage.connect()
    power = PowerControl(args.down_hold, args.min_conf)
    probs = collections.deque(maxlen=args.smooth)
    buf = collections.deque(maxlen=int(win_s * config.EXPECTED_RATE_HZ * 3))
    min_packets = int(MIN_FILL * win_s * config.EXPECTED_RATE_HZ)
    energy = {"used": 0.0, "full": 0.0}                 # watt-seconds, for the savings estimate
    next_eval = last_ts = last_wall = None
    color = sys.stdout.isatty() and not args.no_color
    print(f"Live on {name}: {win_s:.0f}-s window, decision every {step_s} s. Ctrl+C to stop.")
    print(f"Power: LOW->ECO ({SHARE_FOR['ECO']:.0%})  MEDIUM->STANDARD ({SHARE_FOR['STANDARD']:.0%})  "
          f"HIGH->FULL (100%), load {args.load_watts:.0f} W\n")

    calib = [] if args.calibrate else None
    if calib is not None:
        print(f"Calibrating: keep the room/box EMPTY for {args.calibrate:.0f} s ...")
    elif baseline is None:
        print("NOTE: model has no empty-room baseline; baseline features are 0 (use --calibrate).")

    try:
        for fr in src:
            if calib is not None:                          # collect the empty-room baseline first
                calib.append(fr)
                if fr.ts_ms - calib[0].ts_ms < args.calibrate * 1000:
                    continue
                cts = np.array([f.ts_ms for f in calib])
                ciq = np.vstack([f.iq for f in calib])
                crs = np.array([f.rssi for f in calib], float)
                profs = [featurize(cts[s:e], ciq[s:e], crs[s:e])[1] for s, e in windows_of(cts, win_s)]
                if profs:
                    baseline = make_baseline(profs)
                    print(f"Baseline captured from {len(profs)} windows. Live from now on.\n")
                else:
                    print("Calibration got too little CSI; keeping the model's stored baseline.\n")
                calib = None
                continue
            if last_ts is not None and fr.ts_ms + 5000 < last_ts:     # board restarted / replay looped
                buf.clear(); probs.clear(); next_eval = None
            last_ts = fr.ts_ms
            buf.append(fr)
            if next_eval is None:
                next_eval = fr.ts_ms + win_s * 1000
            if fr.ts_ms < next_eval:
                continue
            next_eval = max(next_eval + step_s * 1000, fr.ts_ms)

            frames = [f for f in buf if f.ts_ms > fr.ts_ms - win_s * 1000]
            if len(frames) < min_packets:
                print(f"  waiting for CSI ({len(frames)}/{min_packets} packets in window)")
                continue
            ts = np.array([f.ts_ms for f in frames])
            iq = np.vstack([f.iq for f in frames])
            rssi = np.array([f.rssi for f in frames], float)
            t_calc = time.perf_counter()
            feats, prof = featurize(ts, iq, rssi)
            x = add_baseline_features(np.nan_to_num(np.array([feats])), np.nan_to_num([prof]), baseline)
            p = model.predict_proba(x)[0]
            p = np.array([p[classes.index(c)] if c in classes else 0.0 for c in config.CLASSES])
            calc_ms = (time.perf_counter() - t_calc) * 1000
            probs.append(p)
            sm = np.mean(probs, axis=0)
            label, conf = config.CLASSES[int(sm.argmax())], float(sm.max())
            level, why = power.update(label, conf, fr.ts_ms / 1000)
            src.send(f"S,{label},{level}")                # receiver OLED + power outputs

            wall = time.time()
            if last_wall is not None:                     # CSI clock, so replay is correct too
                dt = min(max(fr.ts_ms / 1000 - last_wall, 0.0), 5.0)
                energy["used"] += args.load_watts * SHARE_FOR[level] * dt
                energy["full"] += args.load_watts * dt
            last_wall = fr.ts_ms / 1000
            saved = 1 - energy["used"] / energy["full"] if energy["full"] else 0.0
            rate = len(frames) / max((ts[-1] - ts[0]) / 1000, 1e-3)
            tag = f"{label.upper():6s}"
            if color:
                tag = f"{COLOR[label]}{tag}\033[0m"
            bars = " ".join(f"{c[0].upper()}:{v:3.0%}" for c, v in zip(config.CLASSES, sm))
            print(f"{time.strftime('%H:%M:%S')}  {tag} {conf:4.0%}  [{bars}]  "
                  f"power {level:8s} {why:22s} saved {saved:4.0%}  {rate:3.0f} pkt/s  {calc_ms:3.0f} ms")

            if con is not None:
                storage.insert(con, {
                    "ts": wall, "raw_label": config.CLASSES[int(p.argmax())], "label": label,
                    "confidence": conf, "p_low": float(sm[0]), "p_medium": float(sm[1]),
                    "p_high": float(sm[2]), "power_level": level, "power_share": SHARE_FOR[level],
                    "reason": why or "steady", "motion": float(x[0][FEATURE_NAMES.index("sc_std_mean")]),
                    "rssi": float(rssi.mean()), "pkt_rate": rate, "source": f"occupancy.py:{name}"})
    except KeyboardInterrupt:
        pass
    if energy["full"]:
        kwh_used, kwh_full = energy["used"] / 3.6e6, energy["full"] / 3.6e6
        print(f"\nEnergy (nominal {args.load_watts:.0f} W load): used {kwh_used:.4f} kWh vs "
              f"{kwh_full:.4f} kWh always-full -> saved {1 - kwh_used / kwh_full:.0%}")


# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="record labelled 15 s CSI clips")
    r.add_argument("--port", help="receiver serial port, e.g. /dev/cu.usbserial-0001")
    r.add_argument("--session", help="folder name for this sitting (default rec-<date>)")
    r.add_argument("--seconds", type=float, default=RECORD_SECONDS)
    r.add_argument("--target", type=int, default=MIN_PER_CLASS, help="recordings wanted per class")
    r.add_argument("--room", default="room1")
    r.add_argument("--condition", default="default")
    r.add_argument("--simulate", action="store_true", help="no hardware: synthetic CSI for trying the flow")
    r.set_defaults(func=cmd_record)

    s = sub.add_parser("status", help="show recordings per class")
    s.add_argument("--include-synthetic", action="store_true")
    s.set_defaults(func=cmd_status)

    t = sub.add_parser("train", help="train the live model")
    t.add_argument("--include-synthetic", action="store_true")
    t.add_argument("--force", action="store_true", help=f"train with fewer than {MIN_PER_CLASS} per class")
    t.add_argument("--out", default=str(MODEL_PATH))
    t.add_argument("--window", type=float, default=LIVE_WINDOW_S,
                   help="seconds of CSI per decision (shorter = faster reaction, longer = steadier)")
    t.set_defaults(func=cmd_train)

    lv = sub.add_parser("live", help="live occupancy + power control")
    g = lv.add_mutually_exclusive_group(required=True)
    g.add_argument("--port")
    g.add_argument("--replay", help="glob of recordings to play back instead of the board")
    lv.add_argument("--model", default=str(MODEL_PATH))
    lv.add_argument("--step", type=float, help=f"seconds between decisions (default {LIVE_STEP_S})")
    lv.add_argument("--smooth", type=int, default=LIVE_SMOOTH, help="decisions to average (1 = none)")
    lv.add_argument("--down-hold", type=float, default=3.0,
                    help="seconds of lower occupancy before power steps down (0 = instant)")
    lv.add_argument("--min-conf", type=float, default=0.5)
    lv.add_argument("--load-watts", type=float, default=1000, help="room load at FULL, for savings estimate")
    lv.add_argument("--speed", type=float, default=1.0)
    lv.add_argument("--loop", action="store_true")
    lv.add_argument("--no-db", action="store_true", help="don't log to data/wisense.db (dashboard)")
    lv.add_argument("--no-color", action="store_true")
    lv.add_argument("--calibrate", type=float, default=0, metavar="SECONDS",
                    help="first record SECONDS of the EMPTY room as a fresh baseline (e.g. 10)")
    lv.set_defaults(func=cmd_live)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
