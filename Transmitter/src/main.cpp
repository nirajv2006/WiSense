#include <Arduino.h>
#include <WiFi.h>
#if __has_include(<WiFiUdp.h>)
#include <WiFiUdp.h>
#elif __has_include(<NetworkUdp.h>)
#include <NetworkUdp.h>
#else
#error "No UDP header found; install/configure the ESP32 Arduino core"
#endif
#include "secrets.h"

//CHANGE LATER, TEST FOR NOW
const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

//Receiver ip and port
IPAddress receiverIp(192,168,4,1);
unsigned int receiverport = RECEIVER_PORT;

WiFiUDP udp;

//Setting up the connection
void setup() {
  Serial.begin(115200);
  delay(1000);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);   // no power-save: keeps the 50 Hz packet rate steady for CSI
  WiFi.begin(ssid,password);

  Serial.print("Connecting to reciever Ap");
  while(WiFi.status() != WL_CONNECTED){
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected Sucessfully");
  Serial.print("Transmitter IP Address : ");
  Serial.println(WiFi.localIP());

  udp.begin(UDP_PORT);
}

// If the receiver's AP disappears (it resets whenever the laptop opens its
// serial port), rejoin quickly instead of waiting for the Wi-Fi driver's slow
// built-in retry, which was observed to take ~30 s.
const unsigned long RECONNECT_INTERVAL_MS = 3000;
unsigned long lastReconnectAttempt = 0;
bool wasConnected = true;

//Sending packets continuously
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
    delay(20);
    return;  // nothing to send while disconnected
  }
  if (!wasConnected) {
    wasConnected = true;
    WiFi.setSleep(false);
    Serial.println("Reconnected - resuming packets");
  }

  udp.beginPacket(receiverIp,receiverport);
  udp.print("TEST_PACKET");
  udp.endPacket();
  delay(20);
}

