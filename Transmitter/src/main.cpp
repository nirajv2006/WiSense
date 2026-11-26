#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <string.h>
#include "esp_wifi.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "secrets.h"

const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

// Receiver IP and port
IPAddress receiverIp(192, 168, 4, 1);
unsigned int receiverPort = RECEIVER_PORT;

WiFiUDP udp;

// Fixed packet cadence: 20 ms (50 Hz) for consistent CSI sampling
const unsigned long PACKET_INTERVAL_MS = 20;
unsigned long nextPacketTime = 0;
unsigned long packetCount = 0;

// Fixed 64-byte payload structure to eliminate packet size/symbol variance
struct __attribute__((packed)) CsiPacket {
    uint32_t seqNum;
    uint32_t timestamp;
    char     padding[56]; // Padded to exactly 64 bytes total
};

CsiPacket txPacket;

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // disable brownout detector
  Serial.begin(115200);
  delay(1000);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to receiver AP");
  while (WiFi.status() != WL_CONNECTED) {
    delay(200);
    Serial.print(".");
  }

  // 1. Disable modem sleep to prevent RF PLL/amplifier clock-down and thermal drift
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);

  // 2. Lock to 802.11n and max transmission power for clean HT-LTF subcarriers
  esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11N);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

  Serial.println("\nConnected Successfully");
  Serial.print("Transmitter IP: ");
  Serial.println(WiFi.localIP());

  // Initialize fixed dummy payload data
  memset(txPacket.padding, 'A', sizeof(txPacket.padding));

  udp.begin(UDP_PORT);
  nextPacketTime = millis() + PACKET_INTERVAL_MS;
  
  Serial.println("Transmitter ready - sending uniform 50 Hz CSI packets");
}

// If the receiver's AP disappears (e.g. the receiver resets when its serial
// port is opened), rejoin quickly instead of waiting for the driver's slow
// built-in retry backoff, which was observed to take ~30 s.
const unsigned long RECONNECT_INTERVAL_MS = 3000;
unsigned long lastReconnectAttempt = 0;
bool wasConnected = true;

void loop() {
  unsigned long now = millis();

  if (WiFi.status() != WL_CONNECTED) {
    if (wasConnected) {
      Serial.println("Lost receiver AP - reconnecting");
      wasConnected = false;
      lastReconnectAttempt = now - RECONNECT_INTERVAL_MS;  // try immediately
    }
    if (now - lastReconnectAttempt >= RECONNECT_INTERVAL_MS) {
      lastReconnectAttempt = now;
      WiFi.disconnect();
      WiFi.begin(ssid, password);
    }
    return;  // no point sending while disconnected
  }
  if (!wasConnected) {
    wasConnected = true;
    WiFi.setSleep(false);
    esp_wifi_set_ps(WIFI_PS_NONE);
    nextPacketTime = now + PACKET_INTERVAL_MS;
    Serial.println("Reconnected - resuming 50 Hz packets");
  }

  // Non-blocking, jitter-free transmission scheduler
  if ((long)(now - nextPacketTime) >= 0) {
    nextPacketTime += PACKET_INTERVAL_MS;

    // Guard against burst catch-up if execution was delayed
    if ((long)(now - nextPacketTime) >= (long)PACKET_INTERVAL_MS) {
      nextPacketTime = now + PACKET_INTERVAL_MS;
    }

    packetCount++;

    txPacket.seqNum = packetCount;
    txPacket.timestamp = now;

    // Send constant-size binary packet
    udp.beginPacket(receiverIp, receiverPort);
    udp.write((const uint8_t*)&txPacket, sizeof(txPacket));
    udp.endPacket();

    // Throttled logging (every 250 packets / 5s) so UART doesn't delay packet transmission
    if (packetCount % 250 == 0) {
      Serial.printf("Packets sent: %lu\n", packetCount);
    }
  }
}
