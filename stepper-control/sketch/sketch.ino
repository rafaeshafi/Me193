// Manual stepper control, the MCU side. Takes orders over the Bridge from python/main.py (the buttons on port 8080).
//   motor   bipolar stepper on an HW-627 (DRV8833): IN1 D6, IN2 D7 (coil A), IN3 D8, IN4 D9 (coil B), EEP D10,
//           VCC = motor supply 3-10 V, GND shared with the board. Full steps, two coils on, sequenced here.
//   move(n)     n full steps, negative = the other way, added to what is still queued
//   run(dir)    keep stepping +1 / -1 until stop
//   stop()      stop now; the coils keep holding
//   release()   stop and switch the coils off, the shaft turns freely
//   speed(ms)   milliseconds a step, 2..500
//   pos()       steps from where the sketch started      queued()  steps still to do (1 while running continuously)
#include <Arduino_RouterBridge.h>

const int kIn[4] = {6, 7, 8, 9}, kSleep = 10;  // IN1..IN4, EEP
// Two-phase full-step sequence, IN1 IN2 IN3 IN4. Swap one coil's two wires if the motor turns the wrong way.
const uint8_t kPhase[4][4] = {{1, 0, 1, 0}, {0, 1, 1, 0}, {0, 1, 0, 1}, {1, 0, 0, 1}};

static volatile int queued = 0, runDir = 0;  // written by Bridge calls, read in loop()
static volatile uint32_t stepMs = 20;
static int position = 0, phase = 0;
static uint32_t nextStepMs = 0;

void coils(bool on) {
  digitalWrite(kSleep, on ? HIGH : LOW);
  for (int i = 0; i < 4; i++) digitalWrite(kIn[i], on ? kPhase[phase][i] : LOW);
}

void stepOnce(int dir) {
  phase = (phase + dir + 4) % 4;
  position += dir;
  coils(true);
}

void moveSteps(int n) { runDir = 0; queued += n; }
void runMotor(int dir) { queued = 0; runDir = dir > 0 ? 1 : (dir < 0 ? -1 : 0); }
void stopMotor() { queued = 0; runDir = 0; }
void releaseMotor() { stopMotor(); coils(false); }
void setSpeed(int ms) { stepMs = ms < 2 ? 2 : (ms > 500 ? 500 : ms); }
int getPos() { return position; }
int getQueued() { return runDir ? 1 : queued; }

void setup() {
  for (int i = 0; i < 4; i++) pinMode(kIn[i], OUTPUT);
  pinMode(kSleep, OUTPUT);
  coils(false);
  Bridge.begin();
  Bridge.provide("move", moveSteps);
  Bridge.provide("run", runMotor);
  Bridge.provide("stop", stopMotor);
  Bridge.provide("release", releaseMotor);
  Bridge.provide("speed", setSpeed);
  Bridge.provide("pos", getPos);
  Bridge.provide("queued", getQueued);
}

void loop() {
  uint32_t now = millis();
  if ((int32_t)(now - nextStepMs) >= 0) {
    nextStepMs = now + stepMs;
    if (runDir) stepOnce(runDir);
    else if (queued > 0) { stepOnce(+1); queued--; }
    else if (queued < 0) { stepOnce(-1); queued++; }
  }
  delay(1);
}
