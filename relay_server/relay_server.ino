/*
 * =============================================================================
 *  JARVIS SMART ROOM SYSTEM — RELAY SERVER
 *  Target Hardware : NodeMCU 1.0 (ESP8266, CH340 USB chip)
 *  Peripheral      : 4-Channel 5V/3.3V Opto-Isolated Relay Module
 *  Role            : Exposes 4 relay channels as local REST endpoints
 * =============================================================================
 */

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ESP8266mDNS.h>
#include <ArduinoOTA.h>

// -----------------------------------------------------------------------------
// USER CONFIGURATION — put your WiFi in wifi_secrets.h
// -----------------------------------------------------------------------------
#include "wifi_secrets.h"   // gitignored - copy wifi_secrets.example.h

const char* MDNS_HOSTNAME = "jarvis-relays";

// -----------------------------------------------------------------------------
// OTA (OVER-THE-AIR) UPDATES — flash future firmware changes over WiFi
// instead of needing a USB cable each time. REQUIRES ONE USB FLASH FIRST:
// this code has to already be running on the board before OTA can accept
// the NEXT update wirelessly — there's no way around that one bootstrap
// flash. After that, use Arduino IDE's Sketch -> Upload, but pick this
// board's network port (shows up as "jarvis-relays at 192.168.0.8" under
// Tools -> Port -> Network Ports) instead of a COM port, or the
// command-line espota.py tool if you're not using the IDE.
// -----------------------------------------------------------------------------
const char* OTA_PASSWORD = "jarvis-relay-ota"; // <-- CHANGE THIS to your
                                                 // own password before
                                                 // flashing — anyone on
                                                 // your LAN who knows this
                                                 // password can push new
                                                 // firmware to the board.

// -----------------------------------------------------------------------------
// PIN MAPPING
// -----------------------------------------------------------------------------
#define RELAY_1_PIN D1   // GPIO5
#define RELAY_2_PIN D2   // GPIO4
#define RELAY_3_PIN D3   // GPIO0
#define RELAY_4_PIN D4   // GPIO2

// -----------------------------------------------------------------------------
// CRITICAL HARDWARE LOGIC — ACTIVE-LOW RELAY BOARD
// -----------------------------------------------------------------------------
#define RELAY_ON  LOW
#define RELAY_OFF HIGH

ESP8266WebServer server(80);

// -----------------------------------------------------------------------------
// STATE TRACKING
// -----------------------------------------------------------------------------
bool relayState[4] = { false, false, false, false };
const uint8_t relayPins[4] = { RELAY_1_PIN, RELAY_2_PIN, RELAY_3_PIN, RELAY_4_PIN };

// -----------------------------------------------------------------------------
// HEAP MONITOR CONFIGURATION (Debounced)
// -----------------------------------------------------------------------------
#define LOW_HEAP_THRESHOLD_BYTES 5000
#define LOW_HEAP_CONSECUTIVE_READINGS_REQUIRED 5
#define LOW_HEAP_MIN_SUSTAINED_MS 2000

uint8_t lowHeapReadings = 0;
uint32_t lowHeapStartTime = 0;

// -----------------------------------------------------------------------------
// HELPER: set a relay by index (0-3) and update tracked state
// -----------------------------------------------------------------------------
void setRelay(uint8_t index, bool turnOn) {
  digitalWrite(relayPins[index], turnOn ? RELAY_ON : RELAY_OFF);
  relayState[index] = turnOn;
}

// -----------------------------------------------------------------------------
// HELPER: build JSON status response
// -----------------------------------------------------------------------------
String relayJson(uint8_t index) {
  String json = "{\"relay\":";
  json += String(index + 1);
  json += ",\"state\":\"";
  json += relayState[index] ? "on" : "off";
  json += "\"}";
  return json;
}

String allRelaysJson() {
  String json = "{";
  for (uint8_t i = 0; i < 4; i++) {
    json += "\"relay" + String(i + 1) + "\":\"" + (relayState[i] ? "on" : "off") + "\"";
    if (i < 3) json += ",";
  }
  json += "}";
  return json;
}

// -----------------------------------------------------------------------------
// ROUTE HANDLERS
// -----------------------------------------------------------------------------

void handleRelayOn(uint8_t index) {
  setRelay(index, true);
  server.send(200, "application/json", relayJson(index));
}

void handleRelayOff(uint8_t index) {
  setRelay(index, false);
  server.send(200, "application/json", relayJson(index));
}

void handleRelayStatus(uint8_t index) {
  server.send(200, "application/json", relayJson(index));
}

void handleAllStatus() {
  server.send(200, "application/json", allRelaysJson());
}

// -----------------------------------------------------------------------------
// BRIGHTNESS STEPPING — this relay board is plain on/off switching, not a
// dimmer, so there's no real PWM brightness control here. "Increase/
// decrease lighting" is approximated by how many of the 4 relays are on
// at once: up turns on the next OFF relay (lowest index first), down
// turns off the highest-numbered ON relay, max turns all 4 on, min drops
// to just relay 1 (still lit, as dim as this hardware can represent
// without going fully off — full off is what /status's "everything off"
// command / individual /relayN/off calls are already for).
// -----------------------------------------------------------------------------
void handleBrightnessUp() {
  for (uint8_t i = 0; i < 4; i++) {
    if (!relayState[i]) {
      setRelay(i, true);
      break;
    }
  }
  server.send(200, "application/json", allRelaysJson());
}

void handleBrightnessDown() {
  for (int8_t i = 3; i >= 0; i--) {
    if (relayState[i]) {
      setRelay(i, false);
      break;
    }
  }
  server.send(200, "application/json", allRelaysJson());
}

void handleBrightnessMax() {
  for (uint8_t i = 0; i < 4; i++) {
    setRelay(i, true);
  }
  server.send(200, "application/json", allRelaysJson());
}

void handleBrightnessMin() {
  setRelay(0, true);
  for (uint8_t i = 1; i < 4; i++) {
    setRelay(i, false);
  }
  server.send(200, "application/json", allRelaysJson());
}

void handleRoot() {
  String html = "<h2>Jarvis Relay Server</h2><p>Status: Online</p><ul>";
  for (uint8_t i = 1; i <= 4; i++) {
    html += "<li>/relay" + String(i) + "/on</li>";
    html += "<li>/relay" + String(i) + "/off</li>";
    html += "<li>/relay" + String(i) + "/status</li>";
  }
  html += "<li>/status (all relays)</li>";
  html += "<li>/brightness/up</li><li>/brightness/down</li>";
  html += "<li>/brightness/max</li><li>/brightness/min</li></ul>";
  server.send(200, "text/html", html);
}

void handleNotFound() {
  server.send(404, "application/json", "{\"error\":\"endpoint not found\"}");
}

// -----------------------------------------------------------------------------
// SETUP
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n[BOOT] Jarvis Relay Server starting...");

  // ---------------------------------------------------------------------
  // STEP 1 — PIN SAFETY FIRST
  // ---------------------------------------------------------------------
  for (uint8_t i = 0; i < 4; i++) {
    pinMode(relayPins[i], OUTPUT);
    digitalWrite(relayPins[i], RELAY_OFF);
    relayState[i] = false;
  }
  Serial.println("[BOOT] All relays forced OFF (safe state).");

  // ---------------------------------------------------------------------
  // STEP 2 — CONNECT TO WI-FI
  // ---------------------------------------------------------------------
  WiFi.mode(WIFI_STA);
  
  // CRITICAL FIX: Disable WiFi Sleep Mode to stay online 24/7
  WiFi.setSleepMode(WIFI_NONE_SLEEP);

  IPAddress staticIP(192, 168, 0, 8);   // The IP you want it to have
  IPAddress gateway(192, 168, 0, 1);     // Your router's IP
  IPAddress subnet(255, 255, 255, 0);    // Standard subnet mask
  WiFi.config(staticIP, gateway, subnet);
  
  // NOTE: previously called WiFi.begin() twice in a row here — harmless
  // (the second call just re-issues the same connection attempt) but
  // redundant, same leftover ac_ir_server.ino already had fixed. One
  // call is all that's needed.
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WIFI] Connecting to ");
  Serial.print(WIFI_SSID);

  uint32_t wifiStartAttempt = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (millis() - wifiStartAttempt > 30000) {
      Serial.println("\n[WIFI] Connection timed out. Restarting...");
      ESP.restart();
    }
  }
  Serial.println();
  Serial.print("[WIFI] Connected. IP address: ");
  Serial.println(WiFi.localIP());

  // ---------------------------------------------------------------------
  // STEP 3 — START mDNS RESPONDER
  // ---------------------------------------------------------------------
  if (MDNS.begin(MDNS_HOSTNAME)) {
    Serial.print("[mDNS] Responder started. Reachable at http://");
    Serial.print(MDNS_HOSTNAME);
    Serial.println(".local");
    MDNS.addService("http", "tcp", 80);
  } else {
    Serial.println("[mDNS] Responder failed to start.");
  }

  // ---------------------------------------------------------------------
  // STEP 4 — REGISTER HTTP ROUTES
  // ---------------------------------------------------------------------
  server.on("/", handleRoot);
  server.on("/status", handleAllStatus);
  server.on("/brightness/up", handleBrightnessUp);
  server.on("/brightness/down", handleBrightnessDown);
  server.on("/brightness/max", handleBrightnessMax);
  server.on("/brightness/min", handleBrightnessMin);

  for (uint8_t i = 0; i < 4; i++) {
    String base = "/relay" + String(i + 1);
    server.on(base + "/on", [i]() { handleRelayOn(i); });
    server.on(base + "/off", [i]() { handleRelayOff(i); });
    server.on(base + "/status", [i]() { handleRelayStatus(i); });
  }

  server.onNotFound(handleNotFound);
  server.begin();
  Serial.println("[HTTP] Web server started on port 80.");

  // ---------------------------------------------------------------------
  // STEP 5 — START OTA LISTENER
  // ---------------------------------------------------------------------
  ArduinoOTA.setHostname(MDNS_HOSTNAME);
  ArduinoOTA.setPassword(OTA_PASSWORD);
  ArduinoOTA.onStart([]() {
    // Force all relays OFF before an update begins — a mid-flash reboot
    // shouldn't be able to leave something stuck ON with no way to
    // control it until the new firmware finishes booting.
    for (uint8_t i = 0; i < 4; i++) {
      digitalWrite(relayPins[i], RELAY_OFF);
    }
    Serial.println("[OTA] Update starting — relays forced off.");
  });
  ArduinoOTA.onEnd([]() {
    Serial.println("[OTA] Update complete, rebooting...");
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.print("[OTA] Error: ");
    Serial.println(error);
  });
  ArduinoOTA.begin();
  Serial.println("[OTA] Listening for wireless firmware updates.");

  Serial.println("[BOOT] Setup complete. Ready for commands.\n");
}

// -----------------------------------------------------------------------------
// MAIN LOOP
// -----------------------------------------------------------------------------
void loop() {
  server.handleClient();
  MDNS.update();
  ArduinoOTA.handle();

  // CRITICAL FIX 1: Auto-Reconnect Wi-Fi if it drops
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WIFI] Disconnected! Attempting to reconnect...");
    delay(1000);
    WiFi.reconnect();
  }

  // CRITICAL FIX 2: Debounced Memory Monitor
  // Requires 5 consecutive low readings over at least 2 seconds before rebooting.
  // This prevents transient heap dips from dropping relays mid-command.
  if (ESP.getFreeHeap() < LOW_HEAP_THRESHOLD_BYTES) {
    if (lowHeapReadings == 0) {
      lowHeapStartTime = millis();
    }
    lowHeapReadings++;

    if (lowHeapReadings >= LOW_HEAP_CONSECUTIVE_READINGS_REQUIRED &&
        (millis() - lowHeapStartTime) >= LOW_HEAP_MIN_SUSTAINED_MS) {
      Serial.println("[SYSTEM] Sustained low memory detected. Rebooting to prevent crash...");
      delay(100);
      ESP.restart();
    }
  } else {
    // Reset counter immediately if a healthy reading comes back
    lowHeapReadings = 0;
  }
}