#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "secrets.h"

const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

// Receiver IP and port
IPAddress receiverIp(192,168,4,1);
unsigned int receiverPort = RECEIVER_PORT;

WiFiUDP udp;

// Packet statistics
unsigned long packetCount = 0;
unsigned long lastPacketTime = 0;

// Variable payload buffer for signal diversity
char payloadBuffer[64];

// Generate variable payload size (20-60 bytes) to create richer signal variation
void generatePayload(char* buffer, unsigned long seqNum, unsigned long timestamp) {
  // Format: {seq_num}:{timestamp}:{variable_data}
  // Variable data length cycles through 20-60 bytes pattern
  int baseLen = 20;
  int maxLen = 60;
  int cycleLen = (seqNum % 5) * 8; // Creates pattern: 0, 8, 16, 24, 32
  int payloadLen = baseLen + cycleLen;
  
  // Build structured packet
  int offset = snprintf(buffer, 16, "%lu:%lu:", seqNum, timestamp);
  
  // Fill remaining space with pattern data for signal diversity
  for (int i = offset; i < payloadLen && i < 63; i++) {
    buffer[i] = 'A' + (i % 26); // Cycles through A-Z
  }
  buffer[payloadLen] = '\0';
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to receiver AP");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected Successfully");
  Serial.print("Transmitter IP Address: ");
  Serial.println(WiFi.localIP());

  udp.begin(UDP_PORT);
  lastPacketTime = millis();
  
  Serial.println("Transmitter ready - sending structured CSI packets");
}

void loop() {
  packetCount++;
  unsigned long currentTimestamp = millis();
  
  // Generate structured packet with variable payload
  generatePayload(payloadBuffer, packetCount, currentTimestamp);
  
  // Send UDP packet to receiver
  udp.beginPacket(receiverIp, receiverPort);
  udp.write((const uint8_t*)payloadBuffer, strlen(payloadBuffer));
  udp.endPacket();
  
  // Debug output every 100 packets
  if (packetCount % 100 == 0) {
    Serial.printf("Packets sent: %lu | Last payload size: %d bytes\n", 
                  packetCount, strlen(payloadBuffer));
  }
  
  // Maintain 20ms packet interval for consistent signal flow
  delay(20);
}
