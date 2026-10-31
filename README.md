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

**Serial data format (115200 baud):**
```
DATA,<timestamp_ms>,<amplitude>,<mean>,<variance>,<std_dev>,<status>
```

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
