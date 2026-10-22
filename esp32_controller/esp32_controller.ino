/*
  AgroSpray - ESP32 low-level controller
  Receives from the Raspberry Pi over USB serial (115200):
      PING                      -> replies PONG
      ACT,<STATE>,<spray_ms>    -> shows status, sprays, drives forward, replies DONE
      STOP                      -> motors and pump off
  STATE: HEALTHY | MILD | MODERATE | SEVERE | DEAD | NOLEAF

  Library needed: "LiquidCrystal I2C" by Frank de Brabander (Library Manager)
  Board: "ESP32 Dev Module"
*/
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ---------------- Pins (avoid GPIO 12: boot strapping pin) ----------------
const int ENA = 25, IN1 = 26, IN2 = 27;     // L298N left motor
const int ENB = 17, IN3 = 14, IN4 = 16;     // L298N right motor
const int PUMP_PIN = 23;                    // MOSFET / relay module for the sprayer pump
const int LED_GREEN = 32;
const int LED_RED = 18;
// LCD I2C: SDA = 21, SCL = 22, address 0x27 (try 0x3F if blank)

// ---------------- Settings ----------------
const bool PUMP_ACTIVE_HIGH = true;         // set false for active-LOW relay modules
const uint8_t DRIVE_PWM = 200;              // 0-255 motor speed
const uint32_t MOVE_MS = 1500;              // forward travel between plants (calibrate)
const uint32_t MAX_SPRAY_MS = 4000;         // hard safety cap on pump time per stop

LiquidCrystal_I2C lcd(0x27, 16, 2);

// ---------------- PWM (works on ESP32 Arduino core 2.x and 3.x) ----------------
void pwmInit() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(ENA, 1000, 8);
  ledcAttach(ENB, 1000, 8);
#else
  ledcSetup(0, 1000, 8); ledcAttachPin(ENA, 0);
  ledcSetup(1, 1000, 8); ledcAttachPin(ENB, 1);
#endif
}

void setSpeed(uint8_t duty) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(ENA, duty);
  ledcWrite(ENB, duty);
#else
  ledcWrite(0, duty);
  ledcWrite(1, duty);
#endif
}

// ---------------- Actuators ----------------
void pump(bool on) { digitalWrite(PUMP_PIN, (on == PUMP_ACTIVE_HIGH) ? HIGH : LOW); }

void stopMotors() {
  setSpeed(0);
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

void forward() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  setSpeed(DRIVE_PWM);
}

void leds(bool green, bool red) {
  digitalWrite(LED_GREEN, green);
  digitalWrite(LED_RED, red);
}

void show(const String &l1, const String &l2) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(l1);
  lcd.setCursor(0, 1); lcd.print(l2);
}

// ---------------- One full stop: show -> spray -> move ----------------
void handleAct(const String &state, uint32_t sprayMs) {
  if (sprayMs > MAX_SPRAY_MS) sprayMs = MAX_SPRAY_MS;

  if (state == "HEALTHY") {
    leds(true, false);  show("Leaf: HEALTHY", "No spray");
  } else if (state == "DEAD") {
    leds(false, true);  show("Crop: DEAD", "Skipped");
  } else if (state == "NOLEAF") {
    leds(false, false); show("No leaf found", "Moving on");
  } else {                                     // MILD / MODERATE / SEVERE
    leds(false, true);
    show("Leaf: " + state, "Spray " + String(sprayMs) + " ms");
    if (sprayMs > 0) {
      pump(true);
      delay(sprayMs);
      pump(false);
    }
  }
  delay(300);

  forward();
  delay(MOVE_MS);
  stopMotors();
}

// ---------------- Setup / loop ----------------
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(200);

  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(PUMP_PIN, OUTPUT);
  pinMode(LED_GREEN, OUTPUT); pinMode(LED_RED, OUTPUT);
  pwmInit();
  stopMotors();
  pump(false);

  Wire.begin(21, 22);
  lcd.init();
  lcd.backlight();
  show("AgroSpray", "Ready");
  leds(true, true); delay(500); leds(false, false);
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line == "PING") {
    Serial.println("PONG");
  } else if (line == "STOP") {
    pump(false);
    stopMotors();
    leds(false, false);
    show("Stopped", "");
  } else if (line.startsWith("ACT,")) {
    int c1 = line.indexOf(',', 4);
    if (c1 < 0) { Serial.println("ERR"); return; }
    String state = line.substring(4, c1);
    uint32_t ms = (uint32_t) line.substring(c1 + 1).toInt();
    handleAct(state, ms);
    Serial.println("DONE");
  } else {
    Serial.println("ERR");
  }
}
