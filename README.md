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
```

**`Transmitter/include/secrets.h`** — same content.

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

**Serial data format (921600 baud):**
```
DATA,<timestamp_ms>,<amplitude>,<mean>,<variance>,<std_dev>,<status>     (2 Hz summary, on-board classifier)
CSI,<timestamp_ms>,<rssi>,<n_values>,<imag0> <real0> <imag1> <real1> ... (every packet, raw CSI for Python)
```

---

## Python ML pipeline & Streamlit dashboard

The receiver also streams **raw per-subcarrier CSI** for every packet. The `python/`
folder turns it into labelled datasets, trains and evaluates classifiers
(low / medium / high occupancy), recommends a power-usage level and shows
everything on a Streamlit dashboard. Full method and evaluation guide:
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

```
python/
├── wisense/            config, CSI parsing, preprocessing, features, models, power advisor, storage
├── collect.py          record a labelled recording            -> data/raw/<session>/
├── build_dataset.py    recordings -> windowed feature table   -> data/features.csv
├── train.py            6 models x 4 evaluation schemes        -> reports/, models/
├── evaluate_advisor.py evaluate the power-recommendation layer
├── ingest.py           live inference service (serial or replay) -> data/wisense.db
├── dashboard.py        Streamlit dashboard
├── simulate.py         SYNTHETIC data for testing the software only
└── tests/              unit tests (python -m pytest -q)
```

### Setup (macOS)

```bash
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Try everything without hardware (synthetic data)

```bash
python simulate.py
python build_dataset.py --include-synthetic
python train.py
python ingest.py --replay "data/raw/SYN-s01/*.csv" --loop      # terminal 1
streamlit run dashboard.py                                       # terminal 2
```

Delete `data/raw/SYN-*` and re-run `build_dataset.py` + `train.py` once you have real data.

### With the boards

Close the PlatformIO serial monitor first, since only one program can use the port.

```bash
python collect.py --port /dev/cu.usbserial-0001 --session s01 --label medium --people 3 --seconds 90
python build_dataset.py && python train.py && python evaluate_advisor.py
python ingest.py --port /dev/cu.usbserial-0001                   # terminal 1
streamlit run dashboard.py                                       # terminal 2
```

---

## Quick workflow: record → train → live power control (`occupancy.py`)

One program for the everyday loop. Close the PlatformIO serial monitor first.

```bash
cd python
# 1. Record 15 s clips. After each clip type l / m / h (low / medium / high).
#    It tracks progress and says when every class has the minimum 10 recordings.
python occupancy.py record --port /dev/cu.usbserial-0001

# 2. Train (refuses until each class has >= 10 recordings)
python occupancy.py train

# 3. Live: a decision every 0.5 s from the last 2 s of CSI, power mode sent to the receiver
python occupancy.py live --port /dev/cu.usbserial-0001
```

| Occupancy | Power mode | Receiver outputs |
|---|---|---|
| Low | **ECO** | CPU 80 MHz, OLED dimmed, GPIO 32 output 30 % |
| Medium | STANDARD | CPU 160 MHz, GPIO 32 output 60 % |
| High | FULL | CPU 240 MHz, GPIO 32 output 100 % |

No relay module is needed (`NUM_RELAYS 0`). An optional LED + 220 ohm resistor on GPIO 32 shows the
power level by brightness. To switch real loads later, set `NUM_RELAYS` and the `RELAYS_*` masks in `Receiver/src/main.cpp`.
Power steps **up instantly**; it steps **down after 3 s** of consistently lower occupancy
(`--down-hold 0` for fully instant). If the PC stops sending for 10 s the board falls back to STANDARD.

Useful options: `occupancy.py status`, `train --window 3` (steadier, slower),
`live --load-watts 1500` (energy-saved estimate), `record --simulate` (try the flow without boards).
Live results are logged to `data/wisense.db`, so `streamlit run dashboard.py` shows them too.

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

- [ESP-IDF](https://docs.espressif.com/projects/esp-idf) — CSI callback via `esp_wifi_set_csi_rx_cb()`
- [PlatformIO](https://platformio.org/) — build system
- [Adafruit SSD1306](https://github.com/adafruit/Adafruit_SSD1306) — OLED driver
- [Chart.js](https://www.chartjs.org/) — live amplitude chart
- Web Serial API — browser ↔ ESP32 serial bridge
