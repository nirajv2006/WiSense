#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "esp_wifi.h"
#include "esp_idf_version.h"
#include "csi_processor.h"
#include "classifier.h"
#include "secrets.h"

// ─── OLED ─────────────────────────────────────────────────────────────────────
#define SCREEN_WIDTH    128
#define SCREEN_HEIGHT    64
#define OLED_RESET       -1
#define SCREEN_ADDRESS 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ─── Wi-Fi / UDP ──────────────────────────────────────────────────────────────
const char*  ssid      = WIFI_SSID;
const char*  password  = WIFI_PASSWORD;
unsigned int localPort = RECEIVER_PORT;

WiFiUDP       udp;
char          rxBuf[255];
unsigned long packetCount = 0;

// ─── Signal processing & classification ──────────────────────────────────────
CSIProcessor csiProcessor;
Classifier   classifier;

// ─── Timing ───────────────────────────────────────────────────────────────────
unsigned long lastFeatureUpdate = 0;
unsigned long lastSerialUpdate  = 0;
unsigned long lastOledUpdate    = 0;

#define FEATURE_UPDATE_MS  3000   // re-classify every 3 s
#define SERIAL_UPDATE_MS    500   // CSV serial output at 2 Hz
#define OLED_UPDATE_MS     1000   // OLED refresh at 1 Hz

// ─── Shared state between CSI callback and main loop ─────────────────────────
static volatile float s_latestAmp = 0.0f;
static volatile bool  s_newSample = false;

// ─── Raw CSI streaming to the PC (WiSense Python pipeline) ───────────────────
// Every CSI frame from the transmitter is queued and printed as one line:
//   CSI,<millis>,<rssi>,<n_values>,<imag0> <real0> <imag1> <real1> ...
// Lines starting with '#' or 'DATA' are ignored by the Python side.
#define STREAM_RAW_CSI   1      // 0 = old behaviour (summary DATA lines only)
#define CSI_MAX_BYTES  128      // LLTF: 64 subcarriers x (imag, real)

typedef struct {
    int8_t   buf[CSI_MAX_BYTES];
    uint16_t len;
    int8_t   rssi;
    uint32_t ts;
} CsiFrame;

static QueueHandle_t     s_csiQueue   = NULL;
static volatile uint32_t s_dropped    = 0;
static uint8_t           s_txMac[6]   = {0};
static volatile bool     s_txMacKnown = false;

// Prediction sent back by the PC ("S,<label>,<power_level>") for the OLED
String        pcLabel = "", pcPower = "";
unsigned long pcLastUpdate = 0;
String        serialIn = "";

// ─── Power control (driven by the PC: python occupancy.py live) ──────────────
// The PC sends "S,<occupancy>,<ECO|STANDARD|FULL>" every 0.5 s. The receiver
// applies it to real outputs so low occupancy really means low consumption:
//
//   mode      relays                PWM load (lights/fan)   OLED     CPU clock
//   ECO       RELAYS_ECO mask       30 %                    dimmed   80 MHz
//   STANDARD  RELAYS_STANDARD mask  60 %                    normal  160 MHz
//   FULL      RELAYS_FULL mask     100 %                    normal  240 MHz
//
// If the PC goes quiet for POWER_FAILSAFE_MS the board falls back to STANDARD
// so a crashed laptop never leaves the room dark or fully powered.
//
// Optional relay module wiring (only if NUM_RELAYS > 0):
//   module VCC -> ESP32 VIN (5 V)    module GND -> ESP32 GND
//   module IN1 -> GPIO 25   IN2 -> GPIO 26   IN3 -> GPIO 27   IN4 -> GPIO 33
// Test with a low-voltage load (LED strip, 5/12 V fan) before any mains wiring.
#define NUM_RELAYS         0     // 0 = no relay module (default). Set 1-4 if you add one later.
static const uint8_t RELAY_PINS[4] = {25, 26, 27, 33};
// Which relays are ON in each mode: bit 0 = IN1, bit 1 = IN2, ...
// Default for a 1-channel module: the load is OFF in ECO, ON otherwise.
// Put a non-essential load on it (fan, AC, extra lights) - "low" can still mean 1 person.
#define RELAYS_ECO       0b0000
#define RELAYS_STANDARD  0b0001
#define RELAYS_FULL      0b0001
// Example for a 4-channel module:  ECO 0b0001  STANDARD 0b0011  FULL 0b1111
#define LOAD_PWM_PIN      32     // optional: an LED (+220 ohm to GND) here shows the power level by brightness
#define RELAY_ACTIVE_LOW   1     // most relay modules switch ON when IN is LOW
#define ECO_CPU_SCALING    1     // set 0 if packet rate drops in ECO mode
#define POWER_FAILSAFE_MS 10000

enum PowerMode { PM_ECO = 0, PM_STANDARD = 1, PM_FULL = 2 };
static const char*  PM_NAME[]   = {"ECO", "STANDARD", "FULL"};
static const uint8_t PM_RELAYS[] = {RELAYS_ECO, RELAYS_STANDARD, RELAYS_FULL};
static const uint8_t PM_PWM[]    = {77, 153, 255};     // 30 / 60 / 100 % of 8-bit duty
static const uint32_t PM_CPU[]   = {80, 160, 240};
PowerMode powerMode = PM_STANDARD;
bool      powerFromPc = false;

void writeRelay(uint8_t pin, bool on) {
    digitalWrite(pin, (on ^ (RELAY_ACTIVE_LOW == 1)) ? HIGH : LOW);
}

void setupPwm() {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcAttach(LOAD_PWM_PIN, 1000, 8);
#else
    ledcSetup(0, 1000, 8);
    ledcAttachPin(LOAD_PWM_PIN, 0);
#endif
}

void writePwm(uint8_t duty) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcWrite(LOAD_PWM_PIN, duty);
#else
    ledcWrite(0, duty);
#endif
}

void applyPowerMode(PowerMode m, bool force = false) {
    if (m == powerMode && !force) return;
    powerMode = m;
    for (int i = 0; i < NUM_RELAYS; i++) writeRelay(RELAY_PINS[i], PM_RELAYS[m] & (1 << i));
    writePwm(PM_PWM[m]);
    display.dim(m == PM_ECO);
#if ECO_CPU_SCALING
    setCpuFrequencyMhz(PM_CPU[m]);            // >= 80 MHz keeps Wi-Fi and UART running
#endif
    Serial.printf("# POWER %s (relay mask 0x%X, pwm %u%%, cpu %lu MHz)\n", PM_NAME[m],
                  PM_RELAYS[m], (unsigned)(PM_PWM[m] * 100 / 255), (unsigned long)getCpuFrequencyMhz());
}

void setupPowerOutputs() {
    for (int i = 0; i < NUM_RELAYS; i++) {
        writeRelay(RELAY_PINS[i], false);      // set the level first: no click at boot
        pinMode(RELAY_PINS[i], OUTPUT);
    }
    setupPwm();
    applyPowerMode(PM_STANDARD, true);         // safe default until the model speaks
}

void powerFromString(const String& lvl) {
    if      (lvl.startsWith("ECO"))      applyPowerMode(PM_ECO);
    else if (lvl.startsWith("STANDARD")) applyPowerMode(PM_STANDARD);
    else if (lvl.startsWith("FULL"))     applyPowerMode(PM_FULL);
    else return;
    powerFromPc = true;
}

void checkPowerFailsafe() {
    if (powerFromPc && millis() - pcLastUpdate > POWER_FAILSAFE_MS) {
        powerFromPc = false;
        Serial.println(F("# PC silent - power failsafe to STANDARD"));
        applyPowerMode(PM_STANDARD);
    }
}

// ─── Application state ────────────────────────────────────────────────────────
bool        calibrating  = true;
bool        occupied     = false;
CSIFeatures lastFeatures = {};

// ─── CSI callback ─────────────────────────────────────────────────────────────
// Called from the Wi-Fi driver task each time a frame with CSI data arrives.
// Computes the mean amplitude across active OFDM subcarriers and posts it to
// the main-loop via a volatile flag — keep this function short.
static void csiCallback(void* ctx, wifi_csi_info_t* info) {
    if (!info || !info->buf || info->len < 4) return;

    // Only use frames from our transmitter (ignore phones, other APs, probes).
    if (!s_txMacKnown || memcmp(info->mac, s_txMac, 6) != 0) return;

#if STREAM_RAW_CSI
    if (s_csiQueue) {
        CsiFrame f;
        f.len  = info->len < CSI_MAX_BYTES ? info->len : CSI_MAX_BYTES;
        memcpy(f.buf, info->buf, f.len);
        f.rssi = info->rx_ctrl.rssi;
        f.ts   = (uint32_t)millis();
        if (xQueueSend(s_csiQueue, &f, 0) != pdTRUE) s_dropped++;
    }
#endif

    const int8_t* buf      = info->buf;
    const int     numPairs = info->len / 2;
    const int     start    = info->first_word_invalid ? 2 : 0;

    float sum  = 0.0f;
    int   cnt  = 0;
    for (int i = start; i < numPairs; i++) {
        float im = (float)buf[2 * i];
        float re = (float)buf[2 * i + 1];
        if (re == 0.0f && im == 0.0f) continue;  // skip null / guard subcarriers
        sum += sqrtf(re * re + im * im);
        cnt++;
    }

    if (cnt > 0) {
        s_latestAmp = sum / (float)cnt;
        s_newSample = true;
    }
}

// ─── OLED update ──────────────────────────────────────────────────────────────
// textSize 2 = 12 × 16 px per char (128 px wide display).
// "OCCUPIED"   = 8 ch × 12 = 96 px → center at x = 16
// "NOT"        = 3 ch × 12 = 36 px → center at x = 46
// "SCANNING"   = 8 ch × 12 = 96 px → center at x = 16
void updateOled() {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);

    // Header: "WiSense" on the left, current power mode on the right (always shown)
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.print("WiSense");
    const char* mode = PM_NAME[powerMode];
    display.setCursor(128 - 6 * strlen(mode), 0);
    display.print(mode);
    display.drawFastHLine(0, 10, 128, SSD1306_WHITE);

    if (pcLabel.length() && millis() - pcLastUpdate < 5000) {
        // ML prediction from the PC:  occupancy (big) + mode name (big)
        String occ = pcLabel; occ.toUpperCase();
        display.setTextSize(1);
        display.setCursor(0, 14);
        display.print("Occupancy");
        display.setTextSize(2);
        display.setCursor(64 - 6 * occ.length(), 23);
        display.print(occ);

        display.drawFastHLine(0, 41, 128, SSD1306_WHITE);
        display.setTextSize(1);
        display.setCursor(0, 46);
        display.print("Mode");
        display.setTextSize(2);
        display.setCursor(128 - 12 * strlen(mode), 46);   // right-aligned, fits "STANDARD"
        display.print(mode);
        display.display();
        return;
    }

    display.setTextSize(2);
    if (calibrating) {
        display.setCursor(16, 24);
        display.print("SCANNING");
    } else if (occupied) {
        display.setCursor(16, 24);
        display.print("OCCUPIED");
    } else {
        display.setCursor(46, 15);
        display.print("NOT");
        display.setCursor(16, 33);
        display.print("OCCUPIED");
    }

    display.setTextSize(1);
    char buf[22];
    snprintf(buf, sizeof(buf), "Amp:%4.1f Var:%.0f",
             lastFeatures.currentAmp, lastFeatures.ampVariance);
    display.setCursor(0, 54);
    display.print(buf);

    display.display();
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(921600);          // raw CSI needs more than 115200 baud
    delay(1000);
    s_csiQueue = xQueueCreate(64, sizeof(CsiFrame));

    Wire.begin(21, 22);
    if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        Serial.println(F("SSD1306 allocation failed"));
        for (;;);
    }
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println(F("WiSense Booting..."));
    display.display();

    setupPowerOutputs();

    // Start access point
    WiFi.mode(WIFI_AP);
    WiFi.softAP(ssid, password);
    delay(300);   // give the AP driver time to settle before enabling CSI

    // Enable Wi-Fi CSI collection (config + callback first, then enable)
    esp_wifi_set_ps(WIFI_PS_NONE);

    wifi_csi_config_t csi_cfg = {};
    csi_cfg.lltf_en           = true;   // LLTF → ~52 active subcarriers at 20 MHz
    csi_cfg.htltf_en          = false;
    csi_cfg.stbc_htltf2_en    = false;
    csi_cfg.ltf_merge_en      = true;
    csi_cfg.channel_filter_en = false;  // keep raw per-subcarrier data
    csi_cfg.manu_scale        = false;
    csi_cfg.shift             = 15;
    esp_wifi_set_csi_config(&csi_cfg);

    esp_wifi_set_csi_rx_cb(csiCallback, NULL);
    esp_wifi_set_csi(true);

    udp.begin(localPort);
    csiProcessor.reset();

    Serial.println(F("\n# WiSense CSI Occupancy Monitor"));
    Serial.printf("# AP: %s  IP: %s  Port: %u\n",
                  ssid, WiFi.softAPIP().toString().c_str(), localPort);
    Serial.println(F("# CSV: DATA,timestamp_ms,csi_amp,csi_mean,csi_variance,csi_std,status"));
    Serial.println(F("# RAW: CSI,timestamp_ms,rssi,n_values,imag0 real0 imag1 real1 ..."));
    Serial.println(F("# Waiting for the transmitter to connect..."));
}

// ─── Loop ─────────────────────────────────────────────────────────────────────
// Learn the transmitter's MAC from the soft-AP's station list
void updateTransmitterMac() {
    static unsigned long lastCheck = 0;
    if (millis() - lastCheck < 1000) return;
    lastCheck = millis();

    wifi_sta_list_t list;
    if (esp_wifi_ap_get_sta_list(&list) != ESP_OK || list.num == 0) {
        if (s_txMacKnown) Serial.println(F("# Transmitter disconnected"));
        s_txMacKnown = false;
        return;
    }
    if (!s_txMacKnown || memcmp(s_txMac, list.sta[0].mac, 6) != 0) {
        s_txMacKnown = false;
        memcpy(s_txMac, list.sta[0].mac, 6);
        s_txMacKnown = true;
        Serial.printf("# Transmitter MAC %02X:%02X:%02X:%02X:%02X:%02X%s\n",
                      s_txMac[0], s_txMac[1], s_txMac[2], s_txMac[3], s_txMac[4], s_txMac[5],
                      list.num > 1 ? "  (WARNING: more than one station connected)" : "");
    }
}

// Print queued raw CSI frames as text lines
void streamRawCsi() {
#if STREAM_RAW_CSI
    static char line[720];
    CsiFrame f;
    while (s_csiQueue && xQueueReceive(s_csiQueue, &f, 0) == pdTRUE) {
        int n = snprintf(line, sizeof(line), "CSI,%lu,%d,%u,",
                         (unsigned long)f.ts, f.rssi, (unsigned)f.len);
        for (int i = 0; i < f.len && n < (int)sizeof(line) - 6; i++)
            n += snprintf(line + n, sizeof(line) - n, i ? " %d" : "%d", f.buf[i]);
        Serial.println(line);
    }
#endif
}

// Read "S,<label>,<power>" lines sent back by the PC pipeline
void readPcStatus() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n') {
            if (serialIn.startsWith("S,")) {
                int c2 = serialIn.indexOf(',', 2);
                pcLabel = serialIn.substring(2, c2 < 0 ? serialIn.length() : c2);
                pcPower = c2 < 0 ? "" : serialIn.substring(c2 + 1);
                pcLastUpdate = millis();
                powerFromString(pcPower);
            }
            serialIn = "";
        } else if (c != '\r' && serialIn.length() < 48) {
            serialIn += c;
        }
    }
}

void loop() {
    // Drain incoming UDP queue so it does not overflow
    if (udp.parsePacket()) udp.read(rxBuf, sizeof(rxBuf) - 1);

    updateTransmitterMac();
    streamRawCsi();
    readPcStatus();
    checkPowerFailsafe();

    // Consume the latest CSI sample posted by the callback
    if (s_newSample) {
        s_newSample = false;
        csiProcessor.addSample(s_latestAmp, millis());
        packetCount++;
    }

    unsigned long now = millis();

    // Re-classify once the window has enough samples
    if (now - lastFeatureUpdate >= FEATURE_UPDATE_MS) {
        lastFeatureUpdate = now;
        if (csiProcessor.getSampleCount() >= 20) {
            calibrating  = false;
            lastFeatures = csiProcessor.getFeatures();
            ClassificationResult result = classifier.classify(lastFeatures);
            occupied = (result.state != STATE_EMPTY);
        }
    }

    // CSV serial output at 2 Hz
    if (now - lastSerialUpdate >= SERIAL_UPDATE_MS) {
        lastSerialUpdate = now;
        const char* status = calibrating ? "Calibrating"
                           : occupied    ? "Occupied"
                           :               "Not Occupied";
        Serial.printf("DATA,%lu,%.2f,%.2f,%.2f,%.2f,%s\n",
                      now,
                      lastFeatures.currentAmp,
                      lastFeatures.ampMean,
                      lastFeatures.ampVariance,
                      lastFeatures.ampStdDev,
                      status);
    }

    // OLED refresh at 1 Hz
    if (now - lastOledUpdate >= OLED_UPDATE_MS) {
        lastOledUpdate = now;
        updateOled();
    }
}
