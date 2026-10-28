#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "secrets.h"

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH,SCREEN_HEIGHT,&Wire, OLED_RESET);

const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

unsigned int localPort = RECEIVER_PORT;
WiFiUDP udp;
char incomingPacket[255];
unsigned long packetCount = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(21,22);
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)){
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  display.println(F("BOOTING"));
  display.display();
  delay(1500);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid, password);

  Serial.println("\nRecevier Acess Point Started");
  Serial.print("AP IP Address: ");
  Serial.println(WiFi.softAPIP());

  udp.begin(localPort);
  Serial.printf("Now listening for UDP packets on port %d...\n",localPort);
}

void loop() {
  int packetSize = udp.parsePacket();
  if(packetSize){
    int len = udp.read(incomingPacket, 255);
    if(len > 0)
      incomingPacket[len] = 0;
    packetCount++;
    Serial.printf("Received packet from %s | content: %s | Packet size : %d bytes\n",udp.remoteIP().toString().c_str(),incomingPacket,packetSize);
    display.clearDisplay();

    //This is for the top bar
    display.setTextSize(1);
    display.setCursor(0,0);
    display.println(F("CSI PACKET MONITOR"));
    display.drawFastHLine(0,12,128,SSD1306_WHITE); //line seperator

    //COUNTER DISPLAY
    display.setTextSize(1);
    display.setCursor(0,20);
    display.print(packetCount);

    display.display();
  }
}
