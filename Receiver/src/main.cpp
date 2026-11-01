#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <math.h>
#include <stdio.h>
#include "esp_wifi.h"
#include "csi_processor.h"
#include "classifier.h"
#include "secrets.h"

// ─── OLED ─────────────────────────────────────────────────────────────────────
#define SCREEN_WIDTH    128
#define SCREEN_HEIGHT    64
#define OLED_RESET       -1

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
bool displayAvailable = false;

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

#define FEATURE_UPDATE_MS   250   // evaluate state 4 times a second
#define SERIAL_UPDATE_MS    500   // CSV & debug serial output at 2 Hz
#define OLED_UPDATE_MS      250   // OLED refresh at 4 Hz

// ─── Shared state between CSI callback and main loop ─────────────────────────
static volatile float s_latestAmp = 0.0f;
static volatile bool  s_newSample = false;

// ─── Application state ────────────────────────────────────────────────────────
bool                 calibrating  = true;
bool                 occupied     = false;
CSIFeatures          lastFeatures = {};
ClassificationResult lastResult   = {};

// Calibration accumulator
int calibrationSamples = 0;

// ─── CSI callback ─────────────────────────────────────────────────────────────
static void csiCallback(void* /*ctx*/, wifi_csi_info_t* info) {
    if (!info || !info->buf || info->len < 4) return;

    const int8_t* buf      = info->buf;
    const int    numPairs = info->len / 2;
    const int    start    = info->first_word_invalid ? 2 : 0;

    float sum = 0.0f;
    int   cnt = 0;
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
void updateOled() {
    if (!displayAvailable) return;

    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);

    display.setTextSize(1);
    display.setCursor(43, 0);
    display.print("WiSense");
    display.drawFastHLine(0, 10, 128, SSD1306_WHITE);

    display.setTextSize(2);
    if (calibrating) {
        display.setCursor(16, 22);
        display.print("CALIBRATING");
    } else if (occupied) {
        display.setCursor(16, 22);
        display.print("OCCUPIED");
    } else {
        display.setCursor(46, 14);
        display.print("NOT");
        display.setCursor(16, 30);
        display.print("OCCUPIED");
    }

    display.setTextSize(1);
    char buf[26];
    if (calibrating) {
        snprintf(buf, sizeof(buf), "Calib: %d/60 pkts", calibrationSamples);
    } else {
        snprintf(buf, sizeof(buf), "V:%3.1f/%3.1f D:%3.1f/%3.1f",
                 lastFeatures.ampVariance, classifier.getMotionThreshold(),
                 lastResult.diffAmp, classifier.getAmpThreshold());
    }
    display.setCursor(0, 54);
    display.print(buf);

    display.display();
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println(F("\n# WiSense Booting..."));

    // Initialize I2C on SDA=21, SCL=22
    Wire.begin(21, 22);
    delay(100);

    // OLED init
    if (display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        displayAvailable = true;
        Serial.println(F("# OLED found at 0x3C"));
    } else if (display.begin(SSD1306_SWITCHCAPVCC, 0x3D)) {
        displayAvailable = true;
        Serial.println(F("# OLED found at 0x3D"));
    } else {
        Serial.println(F("# OLED not detected"));
    }

    if (displayAvailable) {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(10, 20);
        display.println(F("WiSense Starting"));
        display.setCursor(10, 35);
        display.println(F("Calibrating empty..."));
        display.display();
    }

    // Start access point
    WiFi.mode(WIFI_AP);
    WiFi.softAP(ssid, password);
    delay(300);

    // Enable Promiscuous & CSI
    esp_wifi_set_promiscuous(true);

    wifi_csi_config_t csi_cfg = {};
    csi_cfg.lltf_en           = true;
    csi_cfg.htltf_en          = true;
    csi_cfg.stbc_htltf2_en    = true;
    csi_cfg.ltf_merge_en      = true;
    csi_cfg.channel_filter_en = false;
    csi_cfg.manu_scale        = false;
    csi_cfg.shift             = 15;
    esp_wifi_set_csi_config(&csi_cfg);

    esp_wifi_set_csi_rx_cb(csiCallback, NULL);
    esp_wifi_set_csi(true);

    udp.begin(localPort);
    csiProcessor.reset();

    Serial.println(F("# WiSense CSI Occupancy Monitor Initialized"));
    Serial.printf("# AP: %s  IP: %s  Port: %u\n", ssid, WiFi.softAPIP().toString().c_str(), localPort);
    Serial.println(F("# Starting empty-space calibration (60 packets)..."));
    Serial.println(F("# CSV: DATA,timestamp_ms,csi_amp,baseline,csi_variance,status"));
}

// ─── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
    // Drain incoming UDP queue
    if (udp.parsePacket()) {
        udp.read(rxBuf, sizeof(rxBuf) - 1);
    }

    // Consume CSI samples
    if (s_newSample) {
        s_newSample = false;
        csiProcessor.addSample(s_latestAmp, millis());
        packetCount++;

        // Initial empty-state calibration: collect 60 packets to determine noise floor
        if (calibrating) {
            calibrationSamples++;
            if (calibrationSamples >= 60) {
                CSIFeatures calib = csiProcessor.getFeatures();
                classifier.calibrateBaseline(calib.ampMean, calib.ampVariance, calib.ampStdDev);
                calibrating = false;

                Serial.println(F("\n# ════════════════════════════════════════════"));
                Serial.println(F("# CALIBRATION COMPLETE"));
                Serial.printf("# Baseline Amplitude:      %.2f\n", calib.ampMean);
                Serial.printf("# Ambient Noise Variance:  %.2f (StdDev: %.2f)\n", calib.ampVariance, calib.ampStdDev);
                Serial.printf("# Dynamic Motion Threshold: %.2f\n", classifier.getMotionThreshold());
                Serial.printf("# Dynamic Obstruction Dev:  %.2f\n", classifier.getAmpThreshold());
                Serial.println(F("# ════════════════════════════════════════════\n"));
            }
        }
    }

    unsigned long now = millis();

    // Re-classify every 250ms
    if (!calibrating && (now - lastFeatureUpdate >= FEATURE_UPDATE_MS)) {
        lastFeatureUpdate = now;
        if (csiProcessor.getSampleCount() >= 15) {
            lastFeatures = csiProcessor.getFeatures();
            lastResult   = classifier.classify(lastFeatures);
            occupied     = lastResult.isOccupied;
        }
    }

    // CSV and debug output at 2 Hz
    if (now - lastSerialUpdate >= SERIAL_UPDATE_MS) {
        lastSerialUpdate = now;
        const char* status = calibrating ? "Calibrating"
                           : occupied    ? "Occupied"
                                         : "Not Occupied";

        Serial.printf("DATA,%lu,%.2f,%.2f,%.2f,%s\n",
                      now,
                      lastFeatures.currentAmp,
                      classifier.getBaseline(),
                      lastFeatures.ampVariance,
                      status);

        if (!calibrating) {
            Serial.printf("# [DEBUG] Var: %4.1f (Limit: %4.1f) | Diff: %4.1f (Limit: %4.1f) => %s\n",
                          lastFeatures.ampVariance,
                          classifier.getMotionThreshold(),
                          lastResult.diffAmp,
                          classifier.getAmpThreshold(),
                          status);
        }
    }

    // OLED refresh at 4 Hz
    if (now - lastOledUpdate >= OLED_UPDATE_MS) {
        lastOledUpdate = now;
        updateOled();
    }
}
