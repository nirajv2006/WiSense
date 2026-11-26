# WiSense

Human presence detection using ESP32 Wi-Fi CSI (Channel State Information). No PIR sensors, no cameras — just passive Wi-Fi signal analysis.

---

## How it works

When a person is in a room, their body absorbs and reflects Wi-Fi signals. This causes small fluctuations in the **CSI amplitude** (the raw per-subcarrier signal strength from the Wi-Fi PHY layer). WiSense captures these fluctuations on the receiver and classifies the room as occupied or empty based on the **variance** of the amplitude over a sliding window.

```
Transmitter (ESP32) ──── UDP packets ────▶ Receiver (ESP32)
                                               │
                                    CSI callback fires on each packet
                                    amplitude = avg( sqrt(I²+Q²) ) per subcarrier
                                    variance of last N amplitudes → classify
                                               │
                                        ┌──────┴──────┐
                                     OLED display   Serial CSV
                                                       │
                                                  dashboard.html
                                               (Web Serial API)
```

---

## Hardware

| Part | Details |
|---|---|
| ESP32 (×2) | NodeMCU-32S or any ESP32 dev board |
| SSD1306 OLED | 128×64, I2C — SDA → GPIO 21, SCL → GPIO 22 |
| USB cables | One per board for power + serial |

---

## Project structure

```
WiSense/
├── Transmitter/          # ESP32 that sends UDP packets
│   └── src/main.cpp
├── Receiver/             # ESP32 that captures CSI + classifies
│   ├── src/
│   │   ├── main.cpp
│   │   ├── csi_processor.cpp
│   │   └── classifier.cpp
│   └── include/
│       ├── csi_processor.h
│       ├── classifier.h
│       └── secrets.h      ← you create this (not committed)
├── pipeline/             # Python: CSI logger, parsing, (later) features + ML
│   ├── logger.py
│   ├── csi_io.py
│   └── tests/
├── data/                 # sessions.csv (metadata); raw/ is gitignored
└── dashboard.html        # Standalone browser dashboard
```

---

## Setup

### 1. Install PlatformIO

Install [PlatformIO](https://platformio.org/) as a VS Code extension or CLI.

### 2. Create secrets files

**`Receiver/include/secrets.h`**
```cpp
#define WIFI_SSID     "YourNetworkName"
#define WIFI_PASSWORD "YourPassword"
#define RECEIVER_PORT 3333
#define UDP_PORT      8888
// Optional: pin the transmitter's MAC. If omitted, the receiver learns it
// from the first station that joins its AP. Only frames from this MAC are used.
// #define TX_MAC {0x24, 0x6F, 0x28, 0x00, 0x00, 0x00}
```

**`Transmitter/src/secrets.h`** — same content (without `TX_MAC`).

> These files are gitignored and must never be committed.

### 3. Flash the transmitter

```bash
cd Transmitter
pio run --target upload
```

### 4. Flash the receiver

```bash
cd Receiver
pio run --target upload
```

Power both boards. The receiver creates a Wi-Fi AP (`WiSense-AP`), the transmitter connects to it and starts sending UDP packets. The receiver OLED will show **NOT OCCUPIED** or **OCCUPIED**.

---

## Dashboard

Open `dashboard.html` in **Chrome or Edge** (Web Serial requires a Chromium browser).

- Click **Connect to ESP32** and select the receiver's COM port
- Live CSI amplitude chart and occupancy status update in real time
- Click **Simulate** to test the dashboard without hardware

**Serial output (921600 baud).** The receiver prints three kinds of line:
```
CSI,seq,ts_us,rx_seq,rssi,sig_mode,cwb,first_invalid,len,<v0 v1 ...>   # every frame, ~50 Hz
DATA,ts_ms,amp,mean,var,std,roc,sigvar,state,level                     # threshold baseline, 2 Hz
STAT,ts_ms,frames,queue_drops,foreign_skipped,tx_known                 # health, every 5 s
```
`v0 v1 ...` are the raw signed 8-bit CSI values from the ESP32 driver, as
(imaginary, real) pairs per subcarrier. Only frames whose source MAC is the
transmitter are kept, so other Wi-Fi traffic nearby is ignored.

---

## Recording labelled CSI sessions (ML dataset)

One-time setup:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Close the dashboard and any serial monitor first (only one program can own the
port), then record one session per occupancy condition:
```bash
.venv/bin/python pipeline/logger.py --label medium --people 3 --room lab1 --setup A \
    --duration 300 --notes "2 seated, 1 walking; door closed; fan off"
```

- `--label` is `low`, `medium` or `high`. `--people` records the actual count,
  so classes can be re-binned later without re-recording.
- `--setup` names the TX/RX placement, for testing on placements the model has not seen.
- Each session is saved to `data/raw/<session_id>.csv`, and one row is appended to
  `data/sessions.csv` with frame rate, lost frames (`rx_seq_gaps`) and notes.
- The logger first waits until CSI arrives steadily (≥ 40 Hz for 2 s; `--min-rate`), so the
  receiver's reset and the transmitter rejoining are not recorded. It gives up after
  `--wait-timeout` seconds (default 60) without saving anything. Then the `--delay`
  countdown runs, and only then does recording start.
- Aim for a rate of ~50 Hz with few gaps. The logger warns if the rate drops.
- The receiver's port is found automatically: the logger listens on every USB serial port
  and picks the one printing receiver output. macOS port names follow the USB socket, not
  the board, so don't rely on them. Use `--port` only to override.

Load a session in Python:
```python
from csi_io import load_raw, amplitude
amp = amplitude(load_raw("data/raw/<session_id>.csv"))   # (frames, subcarriers)
```

Run the parser tests: `.venv/bin/python -m pytest pipeline/tests`

---

## Thresholds

Occupancy is determined by the variance of mean CSI amplitude over a 20-sample sliding window:

| Variance | Classification |
|---|---|
| < 150 | Not Occupied |
| 150 – 600 | Low Activity |
| > 600 | Occupied |

These are tunable in `Receiver/include/classifier.h`. Open the dashboard, walk around, and watch the variance values to calibrate for your environment.

---

## Built with

- [ESP-IDF](https://docs.espressif.com/projects/esp-idf) Wi-Fi CSI API (via the Arduino-ESP32 core) — `esp_wifi_set_csi_rx_cb()`
- [PlatformIO](https://platformio.org/) — build system
- [Adafruit SSD1306](https://github.com/adafruit/Adafruit_SSD1306) — OLED driver
- [Chart.js](https://www.chartjs.org/) — live amplitude chart
- Web Serial API — browser ↔ ESP32 serial bridge
