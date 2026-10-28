#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
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

//Sending packets continuously
void loop() {
  udp.beginPacket(receiverIp,receiverport);
  udp.print("TEST_PACKET");
  udp.endPacket();
  delay(20);
}

