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
    char buf[22];
    snprintf(buf, sizeof(buf), "Amp:%4.1f Var:%.0f",
             lastFeatures.currentAmp, lastFeatures.ampVariance);
    display.setCursor(0, 54);
    display.print(buf);

    display.display();
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(1000);

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

    // Start access point
    WiFi.mode(WIFI_AP);
    WiFi.softAP(ssid, password);
    delay(300);   // give the AP driver time to settle before enabling CSI

    // Enable Wi-Fi CSI collection
    esp_wifi_set_csi(true);

    wifi_csi_config_t csi_cfg = {};
    csi_cfg.lltf_en           = true;   // LLTF → ~52 active subcarriers at 20 MHz
    csi_cfg.htltf_en          = false;
    csi_cfg.stbc_htltf2_en    = false;
    csi_cfg.ltf_merge_en      = true;
    csi_cfg.channel_filter_en = false;  // keep raw per-subcarrier data
    csi_cfg.manu_scale        = false;
    csi_cfg.shift             = 15;
    esp_wifi_set_csi_config(&csi_cfg);

    // esp_wifi_set_csi_cb is the current API (ESP-IDF ≥ 5.0).
    // On ESP-IDF 4.x replace with esp_wifi_set_csi_rx_cb(csiCallback, NULL).
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
    esp_wifi_set_csi_cb(csiCallback, NULL);
#else
    esp_wifi_set_csi_rx_cb(csiCallback, NULL);
#endif

    udp.begin(localPort);
    csiProcessor.reset();

    Serial.println(F("\n# WiSense CSI Occupancy Monitor"));
    Serial.printf("# AP: %s  IP: %s  Port: %u\n",
                  ssid, WiFi.softAPIP().toString().c_str(), localPort);
    Serial.println(F("# CSV: DATA,timestamp_ms,csi_amp,csi_mean,csi_variance,csi_std,status"));
}

// ─── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
    // Drain incoming UDP queue so it does not overflow
    if (udp.parsePacket()) udp.read(rxBuf, sizeof(rxBuf) - 1);

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
