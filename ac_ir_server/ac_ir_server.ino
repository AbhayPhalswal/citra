/*
 * =============================================================================
 *  JARVIS SMART ROOM SYSTEM — AC IR BLASTER SERVER
 *  Target Hardware : NodeMCU 1.0 (ESP8266, CH340 USB chip)
 *  Peripheral      : KY-005 38kHz IR Transmitter Module
 *  Role            : Exposes AC power/temp/mode/fan control as local REST endpoints
 * =============================================================================
 */

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ESP8266mDNS.h>
#include <ArduinoOTA.h>
#include <IRremoteESP8266.h>
#include <IRsend.h>
#include <ir_Coolix.h>           // Coolix protocol

// -----------------------------------------------------------------------------
// USER CONFIGURATION
// -----------------------------------------------------------------------------
#include "wifi_secrets.h"   // gitignored - see wifi_secrets.example.h


const char* MDNS_HOSTNAME = "jarvis-ac";  // reachable at http://jarvis-ac.local

// -----------------------------------------------------------------------------
// OTA (OVER-THE-AIR) UPDATES — same mechanism and same one-time USB-flash
// requirement as relay_server.ino; see that file's comment block for the
// full explanation. This board shows up as "jarvis-ac at 192.168.0.11"
// under Tools -> Port -> Network Ports once this version is flashed.
// -----------------------------------------------------------------------------
const char* OTA_PASSWORD = "jarvis-ac-ota"; // <-- CHANGE THIS to your own
                                              // password before flashing.

// -----------------------------------------------------------------------------
// PIN CONFIGURATION
// -----------------------------------------------------------------------------
const uint16_t IR_LED_PIN = D5;  // GPIO4

// -----------------------------------------------------------------------------
// IR SEND OBJECT + AC PROTOCOL OBJECT
// -----------------------------------------------------------------------------
IRCoolixAC ac(IR_LED_PIN);

ESP8266WebServer server(80);

// -----------------------------------------------------------------------------
// STATE TRACKING
// -----------------------------------------------------------------------------
struct AcState {
  bool    isOn      = false;
  uint8_t tempC     = 24;
  String  mode      = "cool";   // cool | heat | fan | dry | auto
  String  fanSpeed  = "auto";   // auto | low | med | high
};
AcState acState;

// -----------------------------------------------------------------------------
// !!! CRITICAL — COOLIX setMode()/setFan() CALL ORDER, READ BEFORE EDITING !!!
// -----------------------------------------------------------------------------
// Verified directly against the IRremoteESP8266 library's own source
// (ir_Coolix.cpp, IRCoolixAC::setMode()): setMode() INTERNALLY calls
// setFan() itself, every time, for every mode — not just Dry/Auto:
//
//   case kCoolixAuto: case kCoolixDry:    setFan(kCoolixFanAuto0, false);
//   case kCoolixCool/Heat/Fan:            setFan(kCoolixFanAuto, false);
//
// This means calling ac.setMode(...) UNCONDITIONALLY OVERWRITES whatever
// fan speed was set before it, regardless of mode. The only correct
// order is: setMode() FIRST, setFan() SECOND — call setFan() last, or
// your requested fan speed silently reverts to whatever setMode() forced
// it to, with no error and no visible sign anything went wrong. The
// PREVIOUS version of this file called setFan(kCoolixFanAuto) hardcoded
// AFTER the mode chain, which happened to "work" only by coincidence —
// it was always forcing Auto anyway, so the overwrite was invisible.
// Now that real fan-speed selection exists below, getting this order
// backwards would make every fan-speed request silently no-op.
//
// SEPARATELY — ALSO VERIFIED FROM SOURCE: Coolix's "Fan" mode is
// transmitted as a special case of "Dry" mode internally
// (actualmode = kCoolixDry when mode == kCoolixFan, with a distinct
// temperature code substituted in). This is documented, intentional
// library behavior, not a bug in this firmware — but it means your
// physical remote's "Fan" button and this firmware's fan mode may or
// may not produce an IR signal your AC's own firmware treats
// identically to Dry. Test /ac/mode?val=fan directly against your AC
// once flashed; if it behaves like Dry mode instead of true fan-only,
// that's this library's Coolix implementation, not a firmware bug here.
//
// FAN SPEED IS ONLY MEANINGFUL IN Cool/Heat/Fan MODES. setMode() forces
// fan speed to a fixed "Auto0" value for Dry and Auto modes regardless
// of what you request — this is the library enforcing that those two
// modes don't have an adjustable fan speed on real Coolix hardware.
// handleFan() below still accepts and stores the request (so switching
// back to Cool/Heat/Fan later restores it), but does not fight the
// library's override while in Dry/Auto.
// -----------------------------------------------------------------------------

uint8_t fanSpeedToCoolix(const String& speed) {
  if (speed == "low")  return kCoolixFanMin;
  if (speed == "med")  return kCoolixFanMed;
  if (speed == "high") return kCoolixFanMax;
  return kCoolixFanAuto;  // default / "auto"
}

// -----------------------------------------------------------------------------
// HELPER: send the full AC state to the unit
// -----------------------------------------------------------------------------
void pushStateToAc() {
  if (!acState.isOn) {
    ac.off();
    ac.send();
    Serial.println("[IR] Sent -> power=OFF (dedicated off-code)");
    return;
  }

  ac.on();
  ac.setTemp(acState.tempC);

  // STEP 1: setMode() FIRST. This call itself overwrites fan speed
  // internally (see the critical comment block above) — that is
  // EXPECTED and is why setFan() must come AFTER this, not before.
  if (acState.mode == "cool")      ac.setMode(kCoolixCool);
  else if (acState.mode == "heat") ac.setMode(kCoolixHeat);
  else if (acState.mode == "fan")  ac.setMode(kCoolixFan);
  else if (acState.mode == "dry")  ac.setMode(kCoolixDry);
  else if (acState.mode == "auto") ac.setMode(kCoolixAuto);
  else                             ac.setMode(kCoolixCool);

  // STEP 2: setFan() SECOND, AFTER setMode(). This is the line that
  // actually makes the requested fan speed stick. In Dry/Auto modes,
  // the library will still internally clamp this to Auto0 regardless
  // of what we pass — that's real Coolix hardware behavior, not
  // something this firmware can override, so we don't try to fight it.
  ac.setFan(fanSpeedToCoolix(acState.fanSpeed));

  ac.send();

  Serial.print("[IR] Sent -> power=ON temp=");
  Serial.print(acState.tempC);
  Serial.print(" mode=");
  Serial.print(acState.mode);
  Serial.print(" fan=");
  Serial.println(acState.fanSpeed);
}

// -----------------------------------------------------------------------------
// HELPER: build JSON status response
// -----------------------------------------------------------------------------
String acJson() {
  String json = "{\"power\":\"";
  json += acState.isOn ? "on" : "off";
  json += "\",\"temp\":";
  json += String(acState.tempC);
  json += ",\"mode\":\"";
  json += acState.mode;
  json += "\",\"fan\":\"";
  json += acState.fanSpeed;
  json += "\"}";
  return json;
}

// -----------------------------------------------------------------------------
// ROUTE HANDLERS
// -----------------------------------------------------------------------------

// GET /ac/power?state=on   or   /ac/power?state=off
void handlePower() {
  if (!server.hasArg("state")) {
    server.send(400, "application/json", "{\"error\":\"missing 'state' param\"}");
    return;
  }
  String state = server.arg("state");
  if (state != "on" && state != "off") {
    server.send(400, "application/json", "{\"error\":\"'state' must be 'on' or 'off'\"}");
    return;
  }
  acState.isOn = (state == "on");

  // Send HTTP response BEFORE the blocking IR blast, so the WiFi stack
  // isn't held up waiting on the IR transmission to finish.
  server.send(200, "application/json", acJson());
  server.handleClient();
  delay(10);
  pushStateToAc();
}

// GET /ac/temp?val=22
void handleTemp() {
  if (!server.hasArg("val")) {
    server.send(400, "application/json", "{\"error\":\"missing 'val' param\"}");
    return;
  }
  int requestedTemp = server.arg("val").toInt();

  const int MIN_TEMP = 17;
  const int MAX_TEMP = 30;

  if (requestedTemp < MIN_TEMP || requestedTemp > MAX_TEMP) {
    String err = "{\"error\":\"temp out of range (" + String(MIN_TEMP) +
                 "-" + String(MAX_TEMP) + ")\"}";
    server.send(400, "application/json", err);
    return;
  }

  acState.tempC = (uint8_t)requestedTemp;

  server.send(200, "application/json", acJson());
  server.handleClient();
  delay(10);
  pushStateToAc();
}

// GET /ac/mode?val=cool  (cool | heat | fan | dry | auto)
void handleMode() {
  if (!server.hasArg("val")) {
    server.send(400, "application/json", "{\"error\":\"missing 'val' param\"}");
    return;
  }
  String mode = server.arg("val");
  mode.toLowerCase();

  if (mode != "cool" && mode != "heat" && mode != "fan" &&
      mode != "dry"  && mode != "auto") {
    server.send(400, "application/json",
                "{\"error\":\"mode must be cool|heat|fan|dry|auto\"}");
    return;
  }

  acState.mode = mode;

  server.send(200, "application/json", acJson());
  server.handleClient();
  delay(10);
  pushStateToAc();
}

// GET /ac/fan?val=auto  (auto | low | med | high)
//
// NOTE: in dry/auto mode, the request is still STORED (so switching
// back to cool/heat/fan later restores it), but the library will
// override the actual transmitted fan speed to Auto0 regardless — see
// the critical comment block above pushStateToAc() for why. The JSON
// response always reflects what was REQUESTED and stored in
// acState.fanSpeed, which may momentarily not match what the AC
// actually received while in dry/auto mode. This mirrors how /ac/temp
// already behaves when the AC is off (the value is stored, takes
// effect once relevant) rather than silently rejecting the request.
void handleFan() {
  if (!server.hasArg("val")) {
    server.send(400, "application/json", "{\"error\":\"missing 'val' param\"}");
    return;
  }
  String speed = server.arg("val");
  speed.toLowerCase();

  if (speed != "auto" && speed != "low" && speed != "med" && speed != "high") {
    server.send(400, "application/json",
                "{\"error\":\"fan must be auto|low|med|high\"}");
    return;
  }

  acState.fanSpeed = speed;

  server.send(200, "application/json", acJson());
  server.handleClient();
  delay(10);
  pushStateToAc();
}

// -----------------------------------------------------------------------------
// RELATIVE FAN SPEED STEPPING — GET /ac/fan/up and /ac/fan/down
// -----------------------------------------------------------------------------
// "auto" isn't a rung on this ladder — it's a distinct mode (adaptive
// speed chosen by the AC itself), not a point between low and high, so
// there's no well-defined "one step up from auto". Stepping UP from auto
// lands on the ladder's lowest rung (low); stepping DOWN from auto has
// nowhere lower to go and is a no-op, same as already being at "low".
const char* FAN_SPEED_LADDER[3] = { "low", "med", "high" };

int8_t fanSpeedLadderIndex(const String& speed) {
  for (uint8_t i = 0; i < 3; i++) {
    if (speed == FAN_SPEED_LADDER[i]) return i;
  }
  return -1;  // "auto", or anything else off the ladder
}

// GET /ac/fan/up
void handleFanUp() {
  int8_t idx = fanSpeedLadderIndex(acState.fanSpeed);
  if (idx < 0) {
    acState.fanSpeed = FAN_SPEED_LADDER[0];       // from auto -> low
  } else if (idx < 2) {
    acState.fanSpeed = FAN_SPEED_LADDER[idx + 1];  // step up one rung
  }
  // idx == 2 ("high"): already at the top, no-op

  server.send(200, "application/json", acJson());
  server.handleClient();
  delay(10);
  pushStateToAc();
}

// GET /ac/fan/down
void handleFanDown() {
  int8_t idx = fanSpeedLadderIndex(acState.fanSpeed);
  if (idx > 0) {
    acState.fanSpeed = FAN_SPEED_LADDER[idx - 1];  // step down one rung
  }
  // idx == 0 ("low") or idx < 0 ("auto"): already at/below the floor, no-op

  server.send(200, "application/json", acJson());
  server.handleClient();
  delay(10);
  pushStateToAc();
}

// GET /ac/status
void handleStatus() {
  server.send(200, "application/json", acJson());
}

void handleRoot() {
  String html = "<h2>Jarvis AC IR Server</h2><p>Status: Online</p><ul>"
                "<li>/ac/power?state=on|off</li>"
                "<li>/ac/temp?val=17-30</li>"
                "<li>/ac/mode?val=cool|heat|fan|dry|auto</li>"
                "<li>/ac/fan?val=auto|low|med|high</li>"
                "<li>/ac/fan/up</li><li>/ac/fan/down</li>"
                "<li>/ac/status</li></ul>";
  server.send(200, "text/html", html);
}

void handleNotFound() {
  server.send(404, "application/json", "{\"error\":\"endpoint not found\"}");
}

// -----------------------------------------------------------------------------
// LOW-MEMORY MONITOR — DEBOUNCED, matching the fix already applied to
// relay_server.ino. See that file's comment block for the full
// reasoning; short version: a bare "if (freeHeap < X) restart" fires on
// normal transient dips from String concatenation (acJson() runs on
// every request) and causes unnecessary reboots. This requires several
// consecutive low readings AND a minimum elapsed time before acting.
// -----------------------------------------------------------------------------
#define LOW_HEAP_THRESHOLD_BYTES 5000
#define LOW_HEAP_CONSECUTIVE_READINGS_REQUIRED 5
#define LOW_HEAP_MIN_SUSTAINED_MS 2000

uint8_t lowHeapConsecutiveCount = 0;
uint32_t lowHeapFirstSeenAt = 0;

void checkHeapAndRestartIfNeeded() {
  uint32_t freeHeap = ESP.getFreeHeap();

  if (freeHeap < LOW_HEAP_THRESHOLD_BYTES) {
    if (lowHeapConsecutiveCount == 0) {
      lowHeapFirstSeenAt = millis();
    }
    lowHeapConsecutiveCount++;

    bool enoughReadings = (lowHeapConsecutiveCount >= LOW_HEAP_CONSECUTIVE_READINGS_REQUIRED);
    bool enoughTimeElapsed = (millis() - lowHeapFirstSeenAt >= LOW_HEAP_MIN_SUSTAINED_MS);

    if (enoughReadings && enoughTimeElapsed) {
      Serial.print("[SYSTEM] Sustained low memory (");
      Serial.print(freeHeap);
      Serial.print(" bytes free) for ");
      Serial.print(millis() - lowHeapFirstSeenAt);
      Serial.println("ms. Rebooting to prevent crash...");
      delay(100);
      ESP.restart();
    }
  } else {
    lowHeapConsecutiveCount = 0;
    lowHeapFirstSeenAt = 0;
  }
}

// -----------------------------------------------------------------------------
// SETUP
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n[BOOT] Jarvis AC IR Server starting...");

  ac.begin();
  Serial.println("[IR] AC IR wrapper initialized (Coolix protocol).");

  WiFi.mode(WIFI_STA);

  // Disable WiFi modem sleep to prevent random disconnections.
  WiFi.setSleepMode(WIFI_NONE_SLEEP);

  IPAddress staticIP(192, 168, 0, 11);   // The IP you want it to have
  IPAddress gateway(192, 168, 0, 1);     // Your router's IP
  IPAddress subnet(255, 255, 255, 0);    // Standard subnet mask
  WiFi.config(staticIP, gateway, subnet);

  // NOTE: the previous version of this file called WiFi.begin() TWICE
  // in a row here — harmless (the second call just re-issues the same
  // connection attempt) but redundant. Removed the duplicate; one
  // WiFi.begin() call is all that's needed.
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

  if (MDNS.begin(MDNS_HOSTNAME)) {
    Serial.print("[mDNS] Responder started. Reachable at http://");
    Serial.print(MDNS_HOSTNAME);
    Serial.println(".local");
    MDNS.addService("http", "tcp", 80);
  } else {
    Serial.println("[mDNS] Responder failed to start.");
  }

  server.on("/", handleRoot);
  server.on("/ac/power", handlePower);
  server.on("/ac/temp", handleTemp);
  server.on("/ac/mode", handleMode);
  server.on("/ac/fan", handleFan);
  server.on("/ac/fan/up", handleFanUp);
  server.on("/ac/fan/down", handleFanDown);
  server.on("/ac/status", handleStatus);
  server.onNotFound(handleNotFound);
  server.begin();
  Serial.println("[HTTP] Web server started on port 80.");

  // ---------------------------------------------------------------------
  // START OTA LISTENER
  // ---------------------------------------------------------------------
  ArduinoOTA.setHostname(MDNS_HOSTNAME);
  ArduinoOTA.setPassword(OTA_PASSWORD);
  ArduinoOTA.onStart([]() {
    // Send the AC its dedicated off-code before a mid-flash reboot can
    // leave it running uncontrolled until the new firmware boots back up.
    ac.off();
    ac.send();
    Serial.println("[OTA] Update starting — AC sent off-code.");
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
  checkHeapAndRestartIfNeeded();
}
