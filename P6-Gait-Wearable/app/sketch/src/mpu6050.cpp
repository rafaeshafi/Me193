#include "mpu6050.h"

#include <Arduino.h>
#include <Wire.h>

namespace gait {

namespace {
const uint8_t kAddr = 0x68;
const uint8_t kSmplrtDiv = 0x19, kConfig = 0x1A, kGyroConfig = 0x1B, kAccelConfig = 0x1C, kFifoEn = 0x23,
              kIntStatus = 0x3A, kUserCtrl = 0x6A, kPwrMgmt1 = 0x6B, kFifoCountH = 0x72, kFifoRW = 0x74,
              kWhoAmI = 0x75;
const int kFifoBytes = 1024;
}  // namespace

bool Mpu6050::write(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(kAddr);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool Mpu6050::readRegs(uint8_t reg, uint8_t* buf, int len) {
  Wire.beginTransmission(kAddr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)kAddr, len) != len) return false;
  for (int i = 0; i < len; ++i) buf[i] = Wire.read();
  return true;
}

bool Mpu6050::begin() {
  present_ = false;
  Wire.begin();
  Wire.setClock(400000);
  uint8_t who = 0;
  // genuine parts answer 0x68; common clones answer 0x70/0x72/0x98, so only an absent device (0x00/0xFF) is rejected
  if (!readRegs(kWhoAmI, &who, 1) || who == 0x00 || who == 0xFF) return false;
  bool ok = write(kPwrMgmt1, 0x80);  // reset
  delay(100);
  ok = write(kPwrMgmt1, 0x01) && ok;    // wake, clock from the gyro X PLL
  ok = write(kSmplrtDiv, 9) && ok;      // 1 kHz / (1 + 9) = 100 Hz
  ok = write(kConfig, 0x03) && ok;      // DLPF 44 Hz accel / 42 Hz gyro
  ok = write(kGyroConfig, 0x10) && ok;  // +-1000 deg/s
  ok = write(kAccelConfig, 0x10) && ok; // +-8 g
  ok = write(kUserCtrl, 0x04) && ok;    // FIFO reset
  ok = write(kFifoEn, 0x78) && ok;      // gyro x y z + accel x y z into the FIFO (12 bytes per sample)
  ok = write(kUserCtrl, 0x40) && ok;    // FIFO enable
  present_ = ok;
  return ok;
}

int Mpu6050::read(uint8_t* out, int maxSamples) {
  if (!present_) return -1;
  uint8_t status = 0, cnt[2];
  if (!readRegs(kIntStatus, &status, 1) || !readRegs(kFifoCountH, cnt, 2)) {
    present_ = false;
    return -1;
  }
  int bytes = (cnt[0] << 8) | cnt[1];
  if ((status & 0x10) || bytes >= kFifoBytes - kBytesPerSample) {  // overflow: the stream is misaligned, start again
    write(kUserCtrl, 0x04);
    write(kUserCtrl, 0x40);
    return 0;
  }
  int n = bytes / kBytesPerSample;
  if (n > maxSamples) n = maxSamples;
  for (int i = 0; i < n; ++i) {
    if (!readRegs(kFifoRW, out + i * kBytesPerSample, kBytesPerSample)) {
      present_ = false;
      return -1;
    }
  }
  return n;
}

}  // namespace gait
