// Finger lock, the MCU side. Dumb and fast; every decision lives in the Python app over the Bridge.
//   button  D2 to GND (INPUT_PULLUP, 30 ms debounce)  -> Bridge.notify("button", 1) on each press
//   draw    13x8 LED matrix: 104 brightness bytes 0..7 row by row (as Arduino's led-matrix-painter example)
//   leds    bit 0 = red LED on D3, bit 1 = green LED on D4 (each through a 220 ohm resistor to GND)
//   unlock  bipolar stepper on an HW-627 (DRV8833 dual H-bridge): IN1 D6, IN2 D7 (coil A), IN3 D8, IN4 D9 (coil B),
//           EEP D10 (the chip sleeps until this is HIGH), VCC = motor supply 3-10 V, GND shared with the board.
//           The bolt turns kStepsOpen full steps open, waits the given milliseconds, turns back, then the coils are
//           released. One step every kStepMs in loop(); the MCU sequences the coils itself (no step/dir pin).
#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include <vector>

const int kButton = 2, kRed = 3, kGreen = 4;
const int kIn[4] = {6, 7, 8, 9}, kSleep = 10;  // IN1..IN4, EEP
const int kPixels = 104;
const int kStepsOpen = 512;  // full steps from closed to open: 512 = a quarter turn of a 28BYJ-48 (2048/rev), 50 for a 200-step NEMA 17
const uint32_t kDebounceMs = 30, kStepMs = 3;  // 3 ms a step ~ 330 steps/s; raise it if the motor stalls or skips
// Two-phase full-step sequence, IN1 IN2 IN3 IN4. Swap one coil's two wires if the bolt turns the wrong way.
const uint8_t kPhase[4][4] = {{1, 0, 1, 0}, {0, 1, 1, 0}, {0, 1, 0, 1}, {1, 0, 0, 1}};

Arduino_LED_Matrix matrix;
static uint8_t levels[kPixels];  // static: the matrix driver may keep the pointer it is given
enum Latch { CLOSED, OPENING, OPEN, CLOSING };
static volatile Latch latch = CLOSED;
static volatile uint32_t holdMs = 0;
static uint32_t openUntilMs = 0, nextStepMs = 0, pressedSinceMs = 0;
static int pos = 0, phase = 0;  // pos: steps from closed; phase: index into kPhase
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
  holdMs = (uint32_t)(ms < 0 ? 0 : ms);
  if (latch == OPEN) openUntilMs = millis() + holdMs;  // already open: just hold longer
  else latch = OPENING;
}

void coils(bool on) {
  digitalWrite(kSleep, on ? HIGH : LOW);
  for (int i = 0; i < 4; i++) digitalWrite(kIn[i], on ? kPhase[phase][i] : LOW);
}

void step(int dir) {  // one full step; dir +1 opens, -1 closes
  phase = (phase + dir + 4) % 4;
  pos += dir;
  coils(true);
}

void stepLatch(uint32_t now) {
  if ((int32_t)(now - nextStepMs) < 0) return;
  nextStepMs = now + kStepMs;
  switch (latch) {
    case OPENING:
      if (pos < kStepsOpen) step(+1);
      else { latch = OPEN; openUntilMs = now + holdMs; coils(false); }
      break;
    case OPEN:
      if ((int32_t)(now - openUntilMs) >= 0) latch = CLOSING;
      break;
    case CLOSING:
      if (pos > 0) step(-1);
      else { latch = CLOSED; coils(false); }
      break;
    case CLOSED:
      break;
  }
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
  for (int i = 0; i < 4; i++) pinMode(kIn[i], OUTPUT);
  pinMode(kSleep, OUTPUT);
  coils(false);
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
  stepLatch(now);
  delay(1);
}
