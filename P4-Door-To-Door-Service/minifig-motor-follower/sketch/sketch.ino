#include <Arduino_RouterBridge.h>
#include <vector>

// --- Cytron Maker Drive wiring -------------------------------------------
// Maker Drive uses sign-magnitude (PWM_PWM) control: each channel takes TWO
// inputs, not a PWM + DIR pair. Drive one input with the speed and hold the
// other low for forward; swap them for reverse; both low to brake.
//
// Speed is generated in software (bit-banged in loop()) with digitalWrite
// rather than analogWrite, so it does not matter whether a given pin is
// wired to a hardware PWM timer on this board.
// DIAGNOSTIC SWAP (temporary): physical wiring is unchanged (M1A/M1B still on
// board pins 3/11, M2A/M2B still on 4/7 -- see README), but the software
// channel that used to drive 3/11 now drives 4/7 and vice versa. This tells
// us whether a non-spinning motor follows the Maker Drive command (cmd[0]/[1]
// vs cmd[2]/[3], i.e. software bug) or the physical MCU pins (i.e. hardware).
const int MOTOR_1A_PIN = 3;   // drives M2A now (was M1A/pin 3)
const int MOTOR_1B_PIN = 9;   // drives M2B now (was M1B/pin 11)
const int MOTOR_2A_PIN = 5;   // drives M1A now (was M2A/pin 4)
const int MOTOR_2B_PIN = 11;  // drives M1B now (was M2B/pin 7)

const unsigned long PWM_PERIOD_US = 2000;  // 500 Hz software PWM

struct MotorChannel {
  int pinA;
  int pinB;
  volatile int direction;  // -1 reverse, 0 stop, 1 forward
  volatile int duty;       // 0-255
};

MotorChannel motor1 = {MOTOR_1A_PIN, MOTOR_1B_PIN, 0, 0};
MotorChannel motor2 = {MOTOR_2A_PIN, MOTOR_2B_PIN, 0, 0};

void setup() {
  pinMode(MOTOR_1A_PIN, OUTPUT);
  pinMode(MOTOR_1B_PIN, OUTPUT);
  pinMode(MOTOR_2A_PIN, OUTPUT);
  pinMode(MOTOR_2B_PIN, OUTPUT);
  digitalWrite(MOTOR_1A_PIN, LOW);
  digitalWrite(MOTOR_1B_PIN, LOW);
  digitalWrite(MOTOR_2A_PIN, LOW);
  digitalWrite(MOTOR_2B_PIN, LOW);

  Serial.begin(115200);

  Bridge.begin();
  // Single call carrying both motors' values as one vector, rather than two
  // separate scalar-argument calls (which only ever delivered the first one).
  Bridge.provide("set_motors", setMotors);
}

// Bit-bang one channel: high for `duty`/255 of the period on whichever input
// matches the requested direction, low on the other.
void serviceChannel(MotorChannel &channel, unsigned long phase) {
  unsigned long onTime = (unsigned long)channel.duty * PWM_PERIOD_US / 255;
  bool on = channel.direction != 0 && phase < onTime;
  digitalWrite(channel.pinA, (on && channel.direction > 0) ? HIGH : LOW);
  digitalWrite(channel.pinB, (on && channel.direction < 0) ? HIGH : LOW);
}

void loop() {
  unsigned long phase = micros() % PWM_PERIOD_US;
  serviceChannel(motor1, phase);
  serviceChannel(motor2, phase);
}

// cmd: [leftDirection, leftSpeed, rightDirection, rightSpeed]
// direction: -1 = backward, 0 = stop, 1 = forward
// speed: 0-255 duty cycle (ignored when direction == 0)
void setMotors(std::vector<int> cmd) {
  if (cmd.size() < 4) {
    return;
  }
  motor1.direction = -cmd[0];  // M1 is mounted/wired mirrored relative to M2
  motor1.duty = constrain(cmd[1], 0, 255);
  motor2.direction = cmd[2];
  motor2.duty = constrain(cmd[3], 0, 255);

  Serial.print("set_motors L dir=");
  Serial.print(motor1.direction);
  Serial.print(" duty=");
  Serial.print(motor1.duty);
  Serial.print(" | R dir=");
  Serial.print(motor2.direction);
  Serial.print(" duty=");
  Serial.println(motor2.duty);
}
