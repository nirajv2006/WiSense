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

#ifndef SERIAL_BAUD
#define SERIAL_BAUD 921600   // raw CSI at 50 Hz needs ~25 KB/s; 115200 baud is too slow
#endif

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
unsigned long packetCount = 0;   // CSI frames processed since boot

// ─── Signal processing & classification ──────────────────────────────────────
CSIProcessor csiProcessor;
Classifier   classifier;

// ─── Timing ───────────────────────────────────────────────────────────────────
unsigned long lastFeatureUpdate = 0;
unsigned long lastSerialUpdate  = 0;
unsigned long lastOledUpdate    = 0;
unsigned long lastStatUpdate    = 0;

#define FEATURE_UPDATE_MS  3000   // re-classify every 3 s
#define SERIAL_UPDATE_MS    500   // DATA summary line at 2 Hz
#define STAT_UPDATE_MS     5000   // STAT health line every 5 s
#define OLED_UPDATE_MS     1000   // OLED refresh at 1 Hz

// ─── Raw CSI frame queue (callback → main loop) ──────────────────────────────
// The Wi-Fi driver task fills this; loop() drains it. A queue (instead of a
// single "latest value" slot) guarantees no frame is silently overwritten.
#define CSI_MAX_LEN     384   // bytes: LLTF(128) + HT-LTF(128) + STBC(128) worst case
#define CSI_QUEUE_DEPTH  32

typedef struct {
    uint32_t ts_us;        // rx_ctrl.timestamp (receiver clock, µs)
    uint16_t rx_seq;       // 802.11 sequence number — gaps = lost packets
    int8_t   rssi;
    uint8_t  sig_mode;     // 0 = non-HT (11b/g), 1 = HT (11n)
    uint8_t  cwb;          // 0 = 20 MHz, 1 = 40 MHz
    uint8_t  first_invalid;
    uint16_t len;          // bytes in buf (2 bytes per subcarrier: imag, real)
    int8_t   buf[CSI_MAX_LEN];
} CsiFrame;

static QueueHandle_t     s_csiQueue    = nullptr;
static volatile uint32_t s_queueDrops  = 0;   // frames lost because loop() fell behind
static volatile uint32_t s_foreignSkip = 0;   // frames ignored: not from transmitter

// MAC of the transmitter. Only its frames are used, so neighbouring Wi-Fi
// traffic does not leak into the dataset. Set TX_MAC in secrets.h to pin it,
// otherwise it is learned from the first station that joins our AP.
static uint8_t       s_txMac[6]   = {0};
static volatile bool s_haveTxMac  = false;

// ─── Application state ────────────────────────────────────────────────────────
bool           calibrating   = true;
OccupancyState lastState     = STATE_EMPTY;
OccupancyLevel lastLevel     = LEVEL_NONE;
CSIFeatures    lastFeatures  = {};
uint32_t       csiSeq        = 0;   // receiver-side frame counter

// ─── CSI callback ─────────────────────────────────────────────────────────────
// Runs in the Wi-Fi driver task for every received frame that carries CSI.
// Keep it short: filter by source MAC, copy, enqueue. No printing here.
static void csiCallback(void* ctx, wifi_csi_info_t* info) {
    if (!info || !info->buf || info->len < 4) return;
    if (!s_haveTxMac || memcmp(info->mac, s_txMac, 6) != 0) {
        s_foreignSkip++;
        return;
    }

    CsiFrame f;
    f.ts_us         = info->rx_ctrl.timestamp;
    f.rx_seq        = info->rx_seq;
    f.rssi          = info->rx_ctrl.rssi;
    f.sig_mode      = info->rx_ctrl.sig_mode;
    f.cwb           = info->rx_ctrl.cwb;
    f.first_invalid = info->first_word_invalid ? 1 : 0;
    f.len           = info->len > CSI_MAX_LEN ? CSI_MAX_LEN : info->len;
    memcpy(f.buf, info->buf, f.len);

    if (xQueueSend(s_csiQueue, &f, 0) != pdTRUE) s_queueDrops++;
}

// Learn the transmitter's MAC when it associates with our soft-AP.
static void onStationConnected(arduino_event_id_t event, arduino_event_info_t info) {
#ifndef TX_MAC
    memcpy(s_txMac, info.wifi_ap_staconnected.mac, 6);
    s_haveTxMac = true;
#endif
    Serial.printf("# TX joined: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  info.wifi_ap_staconnected.mac[0], info.wifi_ap_staconnected.mac[1],
                  info.wifi_ap_staconnected.mac[2], info.wifi_ap_staconnected.mac[3],
                  info.wifi_ap_staconnected.mac[4], info.wifi_ap_staconnected.mac[5]);
}

// Mean amplitude across non-null subcarriers — feeds the on-device
// threshold baseline (OLED + DATA lines). The ML pipeline uses raw CSI instead.
static float meanAmplitude(const CsiFrame& f) {
    const int numPairs = f.len / 2;
    const int start    = f.first_invalid ? 2 : 0;
    float sum = 0.0f;
    int   cnt = 0;
    for (int i = start; i < numPairs; i++) {
        float im = (float)f.buf[2 * i];
        float re = (float)f.buf[2 * i + 1];
        if (re == 0.0f && im == 0.0f) continue;   // null / guard subcarrier
        sum += sqrtf(re * re + im * im);
        cnt++;
    }
    return cnt > 0 ? sum / (float)cnt : 0.0f;
}

// One raw CSI row per frame. Raw int8 values are kept in the order the driver
// gives them (imag, real, imag, real, ...), space-separated in the last field:
//   CSI,seq,ts_us,rx_seq,rssi,sig_mode,cwb,first_invalid,len,v0 v1 v2 ...
static char s_line[16 + 64 + CSI_MAX_LEN * 5];
static void printCsiLine(const CsiFrame& f) {
    int n = snprintf(s_line, sizeof(s_line), "CSI,%lu,%lu,%u,%d,%u,%u,%u,%u,",
                     (unsigned long)csiSeq, (unsigned long)f.ts_us, f.rx_seq, f.rssi,
                     f.sig_mode, f.cwb, f.first_invalid, f.len);
    for (int i = 0; i < f.len && n < (int)sizeof(s_line) - 6; i++) {
        n += snprintf(s_line + n, sizeof(s_line) - n, i ? " %d" : "%d", f.buf[i]);
    }
    s_line[n++] = '\n';
    Serial.write((const uint8_t*)s_line, n);
}

// ─── OLED update ──────────────────────────────────────────────────────────────
// Layout (128×64):  header(y=0) │ divider(y=10) │ entity(y=14,sz2) │
//                   sub(y=32,sz1) │ level bar(y=44,h=6) │ stats(y=55,sz1)
void updateOled() {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);

    // Header
    display.setTextSize(1);
    display.setCursor(43, 0);
    display.print("WiSense");
    display.drawFastHLine(0, 10, 128, SSD1306_WHITE);

    if (calibrating) {
        display.setTextSize(2);
        display.setCursor(16, 20);
        display.print("SCANNING");
    } else {
        // Entity label (textSize 2: 12px/char)
        const char* entity = "";
        const char* sub    = "";
        switch (lastState) {
            case STATE_EMPTY:        entity = "EMPTY";  sub = "no presence";  break;
            case STATE_OBJECT:       entity = "OBJECT"; sub = "detected";     break;
            case STATE_HUMAN_STILL:  entity = "HUMAN";  sub = "stationary";   break;
            case STATE_HUMAN_ACTIVE: entity = "HUMAN";  sub = "MOVING";       break;
        }
        int ex = (128 - (int)strlen(entity) * 12) / 2;
        display.setTextSize(2);
        display.setCursor(ex, 14);
        display.print(entity);

        display.setTextSize(1);
        int sx = (128 - (int)strlen(sub) * 6) / 2;
        display.setCursor(sx, 32);
        display.print(sub);

        // Occupancy level bar
        int barW = 0;
        switch (lastLevel) {
            case LEVEL_NONE:   barW = 0;   break;
            case LEVEL_LOW:    barW = 43;  break;
            case LEVEL_MEDIUM: barW = 86;  break;
            case LEVEL_HIGH:   barW = 128; break;
        }
        display.drawRect(0, 44, 128, 6, SSD1306_WHITE);
        if (barW > 0) display.fillRect(0, 44, barW, 6, SSD1306_WHITE);
    }

    // Bottom stats line
    display.setTextSize(1);
    char buf[22];
    snprintf(buf, sizeof(buf), "A:%.1f V:%.0f",
             lastFeatures.currentAmp, lastFeatures.ampVariance);
    display.setCursor(0, 55);
    display.print(buf);

    // Level label on right
    const char* lvlStr = calibrating ? "CAL" :
                         lastLevel == LEVEL_NONE   ? "---" :
                         lastLevel == LEVEL_LOW    ? "LOW" :
                         lastLevel == LEVEL_MEDIUM ? "MED" : "HIGH";
    display.setCursor(98, 55);
    display.print(lvlStr);

    display.display();
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.setTxBufferSize(4096);   // absorb bursts of CSI lines
    Serial.begin(SERIAL_BAUD);
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

    s_csiQueue = xQueueCreate(CSI_QUEUE_DEPTH, sizeof(CsiFrame));
#ifdef TX_MAC
    {
        const uint8_t mac[6] = TX_MAC;
        memcpy(s_txMac, mac, 6);
        s_haveTxMac = true;
    }
#endif
    WiFi.onEvent(onStationConnected, ARDUINO_EVENT_WIFI_AP_STACONNECTED);

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

    esp_wifi_set_csi_rx_cb(csiCallback, NULL);

    udp.begin(localPort);
    csiProcessor.reset();

    Serial.println(F("\n# WiSense CSI Occupancy Monitor"));
    Serial.printf("# AP: %s  IP: %s  Port: %u\n",
                  ssid, WiFi.softAPIP().toString().c_str(), localPort);
    Serial.println(F("# CSI: CSI,seq,ts_us,rx_seq,rssi,sig_mode,cwb,first_invalid,len,<raw int8 imag/real pairs>"));
    Serial.println(F("# DATA: DATA,ts_ms,amp,mean,var,std,roc,sigvar,state,level"));
    Serial.println(F("# STAT: STAT,ts_ms,frames,queue_drops,foreign_skipped,tx_known"));
}

// ─── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
    // Drain incoming UDP queue so it does not overflow
    if (udp.parsePacket()) udp.read(rxBuf, sizeof(rxBuf) - 1);

    // Drain every queued CSI frame: stream it raw, and feed the baseline
    CsiFrame f;
    while (xQueueReceive(s_csiQueue, &f, 0) == pdTRUE) {
        printCsiLine(f);
        csiSeq++;
        float amp = meanAmplitude(f);
        if (amp > 0.0f) csiProcessor.addSample(amp, millis());
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
            lastState = result.state;
            lastLevel = result.level;
        }
    }

    // CSV serial output at 2 Hz
    // Format: DATA,ts,amp,mean,var,std,roc,sigvar,state,level
    if (now - lastSerialUpdate >= SERIAL_UPDATE_MS) {
        lastSerialUpdate = now;
        const char* stateStr = calibrating ? "Calibrating" : classifier.getStateName(lastState);
        const char* levelStr = calibrating ? "None"        : classifier.getLevelName(lastLevel);
        Serial.printf("DATA,%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s,%s\n",
                      now,
                      lastFeatures.currentAmp,
                      lastFeatures.ampMean,
                      lastFeatures.ampVariance,
                      lastFeatures.ampStdDev,
                      lastFeatures.rateOfChange,
                      lastFeatures.signalVariation,
                      stateStr,
                      levelStr);
    }

    // Health line: lets the logger flag dropped / foreign frames
    if (now - lastStatUpdate >= STAT_UPDATE_MS) {
        lastStatUpdate = now;
        Serial.printf("STAT,%lu,%lu,%lu,%lu,%d\n", now, packetCount,
                      (unsigned long)s_queueDrops, (unsigned long)s_foreignSkip,
                      s_haveTxMac ? 1 : 0);
    }

    // OLED refresh at 1 Hz
    if (now - lastOledUpdate >= OLED_UPDATE_MS) {
        lastOledUpdate = now;
        updateOled();
    }
}
