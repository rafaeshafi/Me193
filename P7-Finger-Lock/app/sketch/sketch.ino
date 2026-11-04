// Finger lock, the MCU side. Dumb and fast; every decision lives in the Python app over the Bridge.
//   button  D2 to GND (INPUT_PULLUP, 30 ms debounce)  -> Bridge.notify("button", 1) on each press
//   draw    13x8 LED matrix: 104 brightness bytes 0..7 row by row (as Arduino's led-matrix-painter example)
//   leds    bit 0 = red LED on D3, bit 1 = green LED on D4 (each through a 220 ohm resistor to GND)
//   unlock  hobby servo on D5 (signal; 5V and GND): the latch swings to kOpenUs for the given milliseconds, then back.
//           The 50 Hz pulse train is made in loop() with delayMicroseconds, so no servo library is needed.
#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include <vector>

const int kButton = 2, kRed = 3, kGreen = 4, kServo = 5;
const int kPixels = 104;
const int kClosedUs = 1000, kOpenUs = 2000;  // servo pulse widths: adjust to your latch
const uint32_t kDebounceMs = 30, kServoPeriodMs = 20;

Arduino_LED_Matrix matrix;
static uint8_t levels[kPixels];  // static: the matrix driver may keep the pointer it is given
static volatile uint32_t openUntilMs = 0;
static uint32_t nextPulseMs = 0, pressedSinceMs = 0;
static bool wasDown = false, reported = false;

void draw(std::vector<uint8_t> frame) {
  for (int i = 0; i < kPixels; i++) levels[i] = i < (int)frame.size() ? (frame[i] > 7 ? 7 : frame[i]) : 0;
  matrix.draw(levels);
}

void leds(int mask) {
  digitalWrite(kRed, (mask & 1) ? HIGH : LOW);
  digitalWrite(kGreen, (mask & 2) ? HIGH : LOW);
}

void unlock(int ms) {
  openUntilMs = millis() + (uint32_t)(ms < 0 ? 0 : ms);
}

void servoPulse(uint32_t now) {
  if ((int32_t)(now - nextPulseMs) < 0) return;
  nextPulseMs = now + kServoPeriodMs;
  int us = ((int32_t)(openUntilMs - now) > 0) ? kOpenUs : kClosedUs;
  digitalWrite(kServo, HIGH);
  delayMicroseconds(us);
  digitalWrite(kServo, LOW);
}

void pollButton(uint32_t now) {
  bool down = digitalRead(kButton) == LOW;
  if (down && !wasDown) { pressedSinceMs = now; reported = false; }
  if (down && !reported && now - pressedSinceMs >= kDebounceMs) {
    reported = true;
    Bridge.notify("button", 1);
  }
  wasDown = down;
}

void setup() {
  pinMode(kButton, INPUT_PULLUP);
  pinMode(kRed, OUTPUT);
  pinMode(kGreen, OUTPUT);
  pinMode(kServo, OUTPUT);
  matrix.begin();
  matrix.setGrayscaleBits(3);  // brightness 0..7
  matrix.clear();
  Bridge.begin();
  Bridge.provide("draw", draw);
  Bridge.provide("leds", leds);
  Bridge.provide("unlock", unlock);
}

void loop() {
  uint32_t now = millis();
  pollButton(now);
  servoPulse(now);
  delay(2);
}
