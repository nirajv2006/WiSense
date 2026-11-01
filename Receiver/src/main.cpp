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

#define FEATURE_UPDATE_MS  1000   // re-classify every 1 s for responsive sensing
#define SERIAL_UPDATE_MS    500   // CSV serial output at 2 Hz
#define OLED_UPDATE_MS      500   // OLED refresh at 2 Hz

// ─── Shared state between CSI callback and main loop ─────────────────────────
static volatile float s_latestAmp = 0.0f;
static volatile bool  s_newSample = false;
static volatile unsigned long s_csiRawCount = 0;

// ─── Application state ────────────────────────────────────────────────────────
bool        calibrating  = true;
bool        occupied     = false;
CSIFeatures lastFeatures = {};

// ─── CSI callback ─────────────────────────────────────────────────────────────
// Called from the Wi-Fi driver task each time a frame with CSI data arrives.
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
        s_csiRawCount++;
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
    char buf[24];
    snprintf(buf, sizeof(buf), "A:%4.1f V:%4.0f P:%lu",
             lastFeatures.currentAmp, lastFeatures.ampVariance, packetCount);
    display.setCursor(0, 54);
    display.print(buf);

    display.display();
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println(F("\n# WiSense Booting..."));

    // Initialize I2C on standard SDA=21, SCL=22
    Wire.begin(21, 22);
    delay(100);

    // Try standard OLED address 0x3C first, then 0x3D
    if (display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        displayAvailable = true;
        Serial.println(F("# OLED found at 0x3C"));
    } else if (display.begin(SSD1306_SWITCHCAPVCC, 0x3D)) {
        displayAvailable = true;
        Serial.println(F("# OLED found at 0x3D"));
    } else {
        Serial.println(F("# OLED not detected at 0x3C or 0x3D (Check wiring: SDA->21, SCL->22, VCC, GND)"));
    }

    if (displayAvailable) {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(10, 20);
        display.println(F("WiSense Starting"));
        display.setCursor(10, 35);
        display.println(F("Connecting..."));
        display.display();
    }

    // Start access point
    WiFi.mode(WIFI_AP);
    WiFi.softAP(ssid, password);
    delay(300);   // allow AP to stabilize

    // 1. Enable Promiscuous / Monitor mode (required for ESP32 CSI hardware capture)
    esp_wifi_set_promiscuous(true);

    // 2. Configure CSI settings (enable both LLTF and HTLTF so 802.11n frames are captured)
    wifi_csi_config_t csi_cfg = {};
    csi_cfg.lltf_en          = true;  // legacy LTF subcarriers
    csi_cfg.htltf_en         = true;  // 802.11n HT-LTF subcarriers
    csi_cfg.stbc_htltf2_en   = true;
    csi_cfg.ltf_merge_en     = true;
    csi_cfg.channel_filter_en = false; // raw subcarrier data
    csi_cfg.manu_scale       = false;
    csi_cfg.shift            = 15;
    esp_wifi_set_csi_config(&csi_cfg);

    // 3. Register CSI receive callback
    esp_wifi_set_csi_rx_cb(csiCallback, NULL);

    // 4. Enable CSI collection
    esp_wifi_set_csi(true);

    udp.begin(localPort);
    csiProcessor.reset();

    // Adjust variance thresholds if needed (Low: 15.0f, High: 60.0f for sensitive detection)
    classifier.setThresholds(20.0f, 80.0f);

    Serial.println(F("# WiSense CSI Occupancy Monitor Initialized"));
    Serial.printf("# AP: %s  IP: %s  Port: %u\n",
                  ssid, WiFi.softAPIP().toString().c_str(), localPort);
    Serial.println(F("# CSV: DATA,timestamp_ms,csi_amp,csi_mean,csi_variance,csi_std,status"));
}

// ─── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
    // Drain incoming UDP queue so it does not overflow
    if (udp.parsePacket()) {
        udp.read(rxBuf, sizeof(rxBuf) - 1);
    }

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
        if (csiProcessor.getSampleCount() >= 15) {
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
                                         : "Not Occupied";
        Serial.printf("DATA,%lu,%.2f,%.2f,%.2f,%.2f,%s\n",
                      now,
                      lastFeatures.currentAmp,
                      lastFeatures.ampMean,
                      lastFeatures.ampVariance,
                      lastFeatures.ampStdDev,
                      status);
    }

    // OLED refresh at 2 Hz
    if (now - lastOledUpdate >= OLED_UPDATE_MS) {
        lastOledUpdate = now;
        updateOled();
    }
}