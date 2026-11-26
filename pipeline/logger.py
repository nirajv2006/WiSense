"""WiSense CSI session logger.

Records raw CSI from the ESP32 receiver over USB serial into one labelled
session file, and appends that session's metadata to data/sessions.csv.

Example (5-minute "medium" session with 3 people in room lab1, setup A):

    python pipeline/logger.py --label medium --people 3 --room lab1 --setup A \
        --duration 300 --notes "2 seated, 1 walking; door closed; fan off"

Press Ctrl-C to stop early; the partial session is still saved and marked.
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from datetime import datetime
from pathlib import Path

import serial  # pyserial

from csi_io import RAW_COLUMNS, GapCounter, parse_csi_line

LABELS = ("low", "medium", "high")
SESSION_COLUMNS = [
    "session_id", "file", "label", "people", "room", "setup", "start_time",
    "duration_s", "frames", "rate_hz", "rx_seq_gaps", "malformed",
    "queue_drops", "dropouts", "max_silence_s", "completed", "stop_reason", "notes",
]
DROPOUT_SECONDS = 1.0  # a pause in CSI longer than this counts as a dropout
PORT_PATTERNS = ("/dev/cu.usbserial*", "/dev/cu.SLAB_USBtoUART*",
                 "/dev/cu.wchusbserial*", "/dev/ttyUSB*", "/dev/ttyACM*")


RECEIVER_PREFIXES = ("CSI,", "STAT,", "DATA,", "# WiSense")
PROBE_SECONDS = 7  # receiver prints DATA every 0.5 s; allows for a reset + boot


def candidate_ports() -> list[str]:
    return sorted({p for pat in PORT_PATTERNS for p in glob.glob(pat)})


def find_receiver(baud: int, ports: list[str] | None = None,
                  probe_seconds: float = PROBE_SECONDS) -> tuple[str, serial.Serial]:
    """Find the receiver by what it prints, not by its port name.

    macOS names USB serial ports after the USB socket, so the receiver's name
    changes when boards are replugged. Every candidate port is opened and
    read; the first that sends receiver lines (CSI/STAT/DATA or the boot
    banner) wins. Its port stays open and is returned, so the board is not
    reset a second time.
    """
    ports = candidate_ports() if ports is None else ports
    if not ports:
        sys.exit("No ESP32 serial port found. Plug in the receiver (the board with the OLED).")
    print(f"Looking for the receiver on: {', '.join(ports)} ...")

    opened: dict[str, serial.Serial] = {}
    pending: dict[str, bytes] = {}
    for p in ports:
        try:
            s = open_serial(p, baud)
            s.timeout = 0
            opened[p], pending[p] = s, b""
        except (serial.SerialException, OSError) as e:
            print(f"  {p}: cannot open ({e}); is another program using it?")

    found = None
    t_end = time.monotonic() + probe_seconds
    while found is None and opened and time.monotonic() < t_end:
        for p, s in opened.items():
            pending[p] += s.read(4096)
            *lines, pending[p] = pending[p].split(b"\n")
            if any(l.strip().startswith(pref.encode()) for l in lines for pref in RECEIVER_PREFIXES):
                found = p
                break
        time.sleep(0.05)

    for p, s in opened.items():
        if p != found:
            s.close()
    if found is None:
        sys.exit("Receiver not found: none of the ports sent WiSense receiver output.\n"
                 "  - Is the receiver (OLED board) plugged into this laptop and flashed?\n"
                 "  - Is a serial monitor or the HTML dashboard holding its port?\n"
                 "  - Or pass --port explicitly.")
    ser = opened[found]
    ser.timeout = 1.0
    print(f"Receiver found on {found}")
    return found, ser


def open_serial(port: str, baud: int) -> serial.Serial:
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = port, baud, 1.0
    # Keep DTR/RTS low so opening the port does not reset the ESP32.
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record a labelled WiSense CSI session.")
    p.add_argument("--label", required=True, choices=LABELS,
                   help="occupancy class for this whole session")
    p.add_argument("--people", required=True, type=int,
                   help="actual number of people present (kept so classes can be re-binned later)")
    p.add_argument("--room", required=True, help="room identifier, e.g. lab1")
    p.add_argument("--setup", default="A",
                   help="TX/RX placement identifier, for cross-setup evaluation")
    p.add_argument("--duration", type=float, default=300, help="recording length in seconds")
    p.add_argument("--delay", type=float, default=10,
                   help="seconds to wait (data discarded) so people can take position")
    p.add_argument("--notes", default="", help="free text: activity, door/window, fan, etc.")
    p.add_argument("--port", help="receiver serial port (found automatically by probing if omitted)")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--out", default="data", help="output directory")
    p.add_argument("--min-rate", type=float, default=40,
                   help="CSI frames/s required for 2 s in a row before the session may start")
    p.add_argument("--wait-timeout", type=float, default=60,
                   help="give up if the stream is not steady within this many seconds")
    return p.parse_args(argv)


STEADY_SECONDS = 2  # consecutive seconds at --min-rate before the stream counts as ready


def wait_for_stream(ser: serial.Serial, min_rate: float, timeout: float) -> bool:
    """Block until CSI arrives at >= min_rate for STEADY_SECONDS in a row.

    Opening the port can reset the receiver, and the transmitter then needs a
    few seconds to rejoin its AP. Waiting here keeps that start-up period out
    of the recorded session. Returns False on timeout.
    """
    print(f"Waiting for a steady CSI stream (>= {min_rate:.0f} Hz for {STEADY_SECONDS} s)...")
    t_start = time.monotonic()
    bucket_start, count, steady, tx_known = t_start, 0, 0, None
    while True:
        now = time.monotonic()
        if now - t_start > timeout:
            hint = ("receiver never saw the transmitter: reset the transmitter"
                    if tx_known is False else
                    "no steady CSI: check both boards are powered and the receiver firmware")
            print(f"\n! Stream not ready after {timeout:.0f} s ({hint}). Nothing saved.")
            return False
        line = ser.readline().decode("ascii", errors="replace").strip()
        if line.startswith("CSI,") and parse_csi_line(line) is not None:
            count += 1
        elif line.startswith("STAT,"):
            parts = line.split(",")
            if len(parts) == 6:
                tx_known = parts[5] == "1"
        elif line.startswith("#"):
            print("\n" + line)
        if now - bucket_start >= 1.0:
            rate = count / (now - bucket_start)
            steady = steady + 1 if rate >= min_rate else 0
            print(f"\r  {now - t_start:4.0f}s  {rate:5.1f} Hz  steady {steady}/{STEADY_SECONDS} ",
                  end="", flush=True)
            if steady >= STEADY_SECONDS:
                print("\nStream ready.")
                return True
            bucket_start, count = now, 0


def append_session(path: Path, meta: dict) -> None:
    """Append one session row; upgrades an older sessions.csv to the current columns."""
    rows, rewrite = [], not path.exists()
    if path.exists():
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != SESSION_COLUMNS:
                rows, rewrite = list(reader), True  # old layout: rewrite, blank where unknown
    if rewrite:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SESSION_COLUMNS, restval="")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in SESSION_COLUMNS})
    with open(path, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=SESSION_COLUMNS, restval="").writerow(meta)


def record(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.port:
        port, ser = args.port, open_serial(args.port, args.baud)
    else:
        port, ser = find_receiver(args.baud)
    print(f"Port {port} @ {args.baud} baud")

    try:
        ready = wait_for_stream(ser, args.min_rate, args.wait_timeout)
    except KeyboardInterrupt:
        ready = False
        print("\nCancelled before recording; nothing saved.")
    except (serial.SerialException, OSError) as e:
        ready = False
        print(f"\n! Lost the receiver's serial port before recording ({e}). Nothing saved.\n"
              "  Check the USB cable, and that no other program has the port open.")
    if not ready:
        ser.close()
        sys.exit(1)

    try:
        if args.delay > 0:
            print(f"Get into position: recording starts in {args.delay:.0f} s ...")
            t_end = time.monotonic() + args.delay
            while time.monotonic() < t_end:
                ser.read(ser.in_waiting or 1)  # discard while waiting
    except KeyboardInterrupt:
        ser.close()
        sys.exit("\nCancelled before recording; nothing saved.")
    except (serial.SerialException, OSError) as e:
        sys.exit(f"\n! Lost the receiver's serial port before recording ({e}). Nothing saved.")
    ser.reset_input_buffer()
    ser.readline()  # drop a possibly partial first line

    start = datetime.now()
    session_id = f"{args.room}_{args.setup}_{args.label}_{start:%Y%m%d-%H%M%S}"
    raw_path = raw_dir / f"{session_id}.csv"
    print(f"Recording -> {raw_path}")

    frames = malformed = queue_drops = 0
    gap_counter = GapCounter()
    tx_known = None
    completed = False
    stop_reason = "duration"
    dropouts, max_silence = 0, 0.0
    t0 = time.monotonic()
    last_report, frames_at_report = t0, 0
    last_frame_time = t0
    last_warning = 0.0

    with open(raw_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RAW_COLUMNS)
        try:
            while True:
                now = time.monotonic()
                if now - t0 >= args.duration:
                    completed = True
                    break
                line = ser.readline().decode("ascii", errors="replace").strip()
                if line.startswith("CSI,"):
                    f = parse_csi_line(line)
                    if f is None:
                        malformed += 1
                    else:
                        gap_counter.update(f.sig_mode, f.rx_seq)
                        writer.writerow([f"{time.time():.6f}", f.seq, f.ts_us, f.rx_seq, f.rssi,
                                         f.sig_mode, f.cwb, f.first_invalid, f.len,
                                         " ".join(map(str, f.values))])
                        frames += 1
                        silence = now - last_frame_time
                        if silence > DROPOUT_SECONDS and frames > 1:
                            dropouts += 1
                        max_silence = max(max_silence, silence)
                        last_frame_time = now
                elif line.startswith("STAT,"):
                    parts = line.split(",")
                    if len(parts) == 6:
                        queue_drops = int(parts[3])
                        tx_known = parts[5] == "1"
                elif line.startswith("#"):
                    print("\n" + line)  # firmware info / boot messages

                # Status line: refreshed every second even when no data arrives
                if now - last_report >= 1.0:
                    rate = (frames - frames_at_report) / (now - last_report)
                    left = args.duration - (now - t0)
                    print(f"\r{now - t0:6.0f}s  {rate:5.1f} Hz  frames={frames}  "
                          f"gaps={gap_counter.gaps}  bad={malformed}  left={left:4.0f}s ",
                          end="", flush=True)
                    last_report, frames_at_report = now, frames

                    silent = now - last_frame_time
                    if now - t0 > 3 and (rate < 25 or silent > 2) and now - last_warning >= 5:
                        last_warning = now
                        if tx_known is False:
                            why = "receiver has not seen the transmitter (reset the transmitter)"
                        elif silent > 2:
                            why = f"no CSI for {silent:.0f} s (transmitter unplugged or disconnected?)"
                        else:
                            why = "low frame rate (check transmitter power and distance)"
                        print(f"\n! {why}")
        except KeyboardInterrupt:
            stop_reason = "cancelled"
            print("\nStopped early; saving partial session.")
        except (serial.SerialException, OSError) as e:
            stop_reason = "serial_error"
            print(f"\n! Lost the receiver's serial port ({e}).\n"
                  "  Usually a USB power dip or loose cable. Saving the partial session.")
        finally:
            try:
                ser.close()
            except (serial.SerialException, OSError):
                pass

    duration = time.monotonic() - t0
    if stop_reason != "duration":
        # silence at the end counts too (e.g. the stream died before the port did)
        max_silence = max(max_silence, time.monotonic() - last_frame_time)
    meta = {
        "session_id": session_id, "file": str(raw_path.relative_to(out)),
        "label": args.label, "people": args.people, "room": args.room, "setup": args.setup,
        "start_time": start.isoformat(timespec="seconds"), "duration_s": round(duration, 1),
        "frames": frames, "rate_hz": round(frames / duration, 2) if duration else 0,
        "rx_seq_gaps": gap_counter.gaps, "malformed": malformed, "queue_drops": queue_drops,
        "dropouts": dropouts, "max_silence_s": round(max_silence, 2),
        "completed": completed, "stop_reason": stop_reason, "notes": args.notes,
    }
    append_session(out / "sessions.csv", meta)

    print(f"\nSaved {frames} frames ({meta['rate_hz']} Hz) to {raw_path}")
    if frames == 0:
        print("! No CSI frames recorded. Check the receiver firmware and transmitter.")
    return meta


if __name__ == "__main__":
    record(parse_args())
