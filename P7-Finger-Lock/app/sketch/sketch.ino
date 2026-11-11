// Finger lock, the MCU side. Dumb and fast; every decision lives in the Python app over the Bridge.
//   button  D2 to GND (INPUT_PULLUP, 30 ms debounce)  -> Bridge.notify("button", 1) on each press
//   draw    13x8 LED matrix: 104 brightness bytes 0..7 row by row (as Arduino's led-matrix-painter example)
//   leds    bit 0 = red LED on D3, bit 1 = green LED on D4 (each through a 220 ohm resistor to GND)
//   unlock  bipolar stepper on an HW-627 (DRV8833 dual H-bridge): IN1 D6, IN2 D7 (coil A), IN3 D8, IN4 D9 (coil B),
//           EEP D10 (the chip sleeps until this is HIGH), VCC = motor supply 3-10 V, GND shared with the board.
//           The bolt turns kStepsOpen full steps open, waits the given milliseconds, turns back, then the coils are
//           released. One step every kStepMs in loop(); the MCU sequences the coils itself (no step/dir pin).
//   boot    with kSelfTest the latch opens and closes once right after power-up or a flash, before the Bridge is up:
//           if the bolt does not move then, the problem is the wiring or the motor supply, not the Python side.
//   motor   manual control from the page, only while the latch is CLOSED: move(n) queues n full steps (signed),
//           run(dir) steps until stop(), release() switches the coils off, speed(ms) sets the manual step time,
//           zero() declares where the bolt is now as closed, pos() answers the steps from closed. Manual moves
//           move pos too, so the next unlock still opens kStepsOpen from closed and comes back to it.
//   screen  SPI TFT showing the state and the digits entered: SCK D13, MOSI D11, CS D5, DC A0, RST A1, 3.3 V.
//           kTftChip picks the controller; the Python app sends screen(state, digits) whenever either changes.
#include <Adafruit_GFX.h>
#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include <vector>

const int kButton = 2, kRed = 3, kGreen = 4;
const int kIn[4] = {6, 7, 8, 9}, kSleep = 10;  // IN1..IN4, EEP
const int kPixels = 104;
#define kTftChip 9341  // 9341 = ILI9341 320x240, 7789 = ST7789 (kTftW x kTftH), 7735 = ST7735 160x128, 0 = no screen
const int kTftCs = 5, kTftDc = A0, kTftRst = A1, kTftW = 240, kTftH = 240;  // W, H only matter for the ST7789
const int kStepsOpen = 50;  // full steps from closed to open: 50 = a quarter turn of a 200-step (1.8 deg) 4-wire motor; 512 for a 28BYJ-48 (2048/rev)
const uint32_t kDebounceMs = 30, kStepMs = 20;  // 20 ms a step = 50 steps/s, a quarter turn in 1 s; 5 ms stalled the motor (buzz, no turn)
const bool kSelfTest = true;                   // one open-hold-close cycle at boot (1 s hold) to prove the motor and the driver
// Two-phase full-step sequence, IN1 IN2 IN3 IN4. Swap one coil's two wires if the bolt turns the wrong way.
const uint8_t kPhase[4][4] = {{1, 0, 1, 0}, {0, 1, 1, 0}, {0, 1, 0, 1}, {1, 0, 0, 1}};

Arduino_LED_Matrix matrix;
#if kTftChip == 9341
#include <Adafruit_ILI9341.h>
Adafruit_ILI9341 tft(kTftCs, kTftDc, kTftRst);
#elif kTftChip == 7789
#include <Adafruit_ST7789.h>
Adafruit_ST7789 tft(kTftCs, kTftDc, kTftRst);
#elif kTftChip == 7735
#include <Adafruit_ST7735.h>
Adafruit_ST7735 tft(kTftCs, kTftDc, kTftRst);
#endif
const char* const kStateName[] = {"IDLE", "LISTENING", "UNLOCKED", "DENIED", "LOCKOUT"};
const uint16_t kStateColor[] = {0x8410, 0xFFE0, 0x07E0, 0xF800, 0xF800};  // grey, yellow, green, red, red (RGB565)
static uint8_t levels[kPixels];  // static: the matrix driver may keep the pointer it is given
enum Latch { CLOSED, OPENING, OPEN, CLOSING };
static volatile Latch latch = CLOSED;
static volatile uint32_t holdMs = 0;
static uint32_t openUntilMs = 0, nextStepMs = 0, pressedSinceMs = 0;
static int pos = 0, phase = 0;  // pos: steps from closed; phase: index into kPhase
static bool wasDown = false, reported = false;
static volatile int manual = 0, runDir = 0;  // from the page: steps still queued, continuous direction (0 = none)
static volatile uint32_t manualMs = 20;      // milliseconds a manual step

void draw(std::vector<uint8_t> frame) {
  for (int i = 0; i < kPixels; i++) levels[i] = i < (int)frame.size() ? (frame[i] > 7 ? 7 : frame[i]) : 0;
  matrix.draw(levels);
}

void screenBegin() {
#if kTftChip == 9341
  tft.begin();
  tft.setRotation(1);
#elif kTftChip == 7789
  tft.init(kTftW, kTftH);
  tft.setRotation(kTftW < kTftH ? 1 : 0);
#elif kTftChip == 7735
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(1);
#endif
#if kTftChip
  tft.fillScreen(0);
#endif
}

void screen(int state, std::vector<uint8_t> digits) {  // state 0..4 as kStateName; digits = the ones entered so far
#if kTftChip
  if (state < 0 || state > 4) state = 0;
  int w = tft.width(), h = tft.height(), big = w >= 240 ? 6 : 3, small = w >= 240 ? 3 : 2;
  tft.fillScreen(0);
  tft.setTextColor(kStateColor[state]);
  tft.setTextSize(small);
  tft.setCursor((w - (int)strlen(kStateName[state]) * 6 * small) / 2, h / 6);
  tft.print(kStateName[state]);
  char line[8] = "_ _ _";
  for (size_t i = 0; i < 3 && i < digits.size(); i++) line[2 * i] = '0' + (digits[i] % 10);
  tft.setTextColor(0xFFFF);
  tft.setTextSize(big);
  tft.setCursor((w - 5 * 6 * big) / 2, h / 2 - 4 * big);
  tft.print(line);
#endif
}

void leds(int mask) {
  digitalWrite(kRed, (mask & 1) ? HIGH : LOW);
  digitalWrite(kGreen, (mask & 2) ? HIGH : LOW);
}

void unlock(int ms) {
  manual = 0; runDir = 0;  // a manual move gives way to the latch
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

void moveSteps(int n) { runDir = 0; manual += n; }
void runMotor(int dir) { manual = 0; runDir = dir > 0 ? 1 : (dir < 0 ? -1 : 0); }
void stopMotor() { manual = 0; runDir = 0; }
void releaseMotor() { stopMotor(); coils(false); }
void setSpeed(int ms) { manualMs = ms < 2 ? 2 : (ms > 500 ? 500 : ms); }
void zero() { pos = 0; }
int getPos() { return pos; }

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
    case CLOSED:  // idle: the page may move the motor by hand
      if (runDir) step(runDir);
      else if (manual > 0) { step(+1); manual--; }
      else if (manual < 0) { step(-1); manual++; }
      else break;
      nextStepMs = now + manualMs;
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
  screenBegin();
  screen(0, {});
  matrix.begin();
  matrix.setGrayscaleBits(3);  // brightness 0..7
  matrix.clear();
  Bridge.begin();
  Bridge.provide("draw", draw);
  Bridge.provide("leds", leds);
  Bridge.provide("screen", screen);
  Bridge.provide("unlock", unlock);
  Bridge.provide("move", moveSteps);
  Bridge.provide("run", runMotor);
  Bridge.provide("stop", stopMotor);
  Bridge.provide("release", releaseMotor);
  Bridge.provide("speed", setSpeed);
  Bridge.provide("zero", zero);
  Bridge.provide("pos", getPos);
  if (kSelfTest) unlock(1000);
}

void loop() {
  uint32_t now = millis();
  pollButton(now);
  stepLatch(now);
  delay(1);
}
