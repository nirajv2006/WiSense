"""Record one labelled CSI recording from the receiver.

Example:
    python collect.py --port /dev/cu.usbserial-0001 --session s01 --label medium \
        --people 3 --room lab_A --condition door_closed --seconds 60

Each run writes   data/raw/<session>/<session>__<label>__<time>.csv   (raw I/Q frames)
and a matching .json file with the label and recording conditions.
"""
import argparse
import time
from datetime import datetime

from wisense import config
from wisense.csi_io import SerialSource, save_recording


def check_people(label, people):
    if people is None:
        return
    lo, hi = config.CLASS_PEOPLE_RANGE[label]
    if people < lo or (hi is not None and people > hi):
        print(f"WARNING: {people} people does not match '{label}' "
              f"({config.CLASS_DEFINITIONS[label]}). Check the label!")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="receiver serial port, e.g. /dev/cu.usbserial-0001")
    ap.add_argument("--label", required=True, choices=config.CLASSES)
    ap.add_argument("--session", required=True,
                    help="ID for one collection sitting, e.g. s01. Models are evaluated on unseen sessions.")
    ap.add_argument("--people", type=int, help="number of people present (for documentation only)")
    ap.add_argument("--room", default="room1")
    ap.add_argument("--condition", default="default",
                    help="free text for what differs between sessions, e.g. door_open, furniture_moved")
    ap.add_argument("--activity", default="mixed", help="e.g. sitting, walking, mixed")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    check_people(args.label, args.people)
    src = SerialSource(args.port)
    stamp = datetime.now()
    path = config.RAW_DIR / args.session / f"{args.session}__{args.label}__{stamp:%Y%m%d_%H%M%S}.csv"

    print(f"Recording '{args.label}' for {args.seconds} s  ->  {path}")
    print("Keep the situation consistent with the label until it finishes (Ctrl+C stops early).")
    frames, host_times = [], []
    t0 = last = time.time()
    try:
        for frame in src:
            frames.append(frame)
            host_times.append(time.time())
            now = time.time()
            if now - last >= 2:
                print(f"  {now - t0:5.0f}s  {len(frames):6d} packets  "
                      f"{len(frames) / (now - t0):5.1f} pkt/s  RSSI {frame.rssi} dBm")
                last = now
            if now - t0 >= args.seconds:
                break
    except KeyboardInterrupt:
        print("Stopped early.")

    if not frames:
        raise SystemExit("No CSI frames received. Is the transmitter connected? "
                         "Is the serial monitor closed? Baud 921600?")
    duration = host_times[-1] - host_times[0]
    rate = len(frames) / max(duration, 1e-3)
    meta = {
        "session": args.session, "label": args.label, "people": args.people,
        "room": args.room, "condition": args.condition, "activity": args.activity,
        "notes": args.notes, "start": stamp.isoformat(timespec="seconds"),
        "duration_s": round(duration, 1), "packets": len(frames), "rate_hz": round(rate, 1),
        "synthetic": False,
    }
    save_recording(frames, host_times, path, meta)
    print(f"Saved {len(frames)} packets ({rate:.1f} pkt/s).")
    if rate < 0.6 * config.EXPECTED_RATE_HZ:
        print(f"WARNING: packet rate is low (expected ~{config.EXPECTED_RATE_HZ}/s). "
              "Move the boards closer or check Wi-Fi interference.")


if __name__ == "__main__":
    main()
