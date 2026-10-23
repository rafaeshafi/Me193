#pragma once
#include <stdint.h>

namespace gait {

// MPU6050 at 0x68 (AD0 low) on Wire, 100 Hz into the chip's own FIFO: accel +-8 g, gyro +-1000 deg/s, DLPF 44 Hz.
// loop() drains the FIFO, so the sampling clock is the sensor's and the sketch may be late by up to ~800 ms (1024 B).
// A sample is 12 bytes straight from the FIFO: ax ay az gx gy gz as big-endian int16.
class Mpu6050 {
 public:
  static const int kBytesPerSample = 12;
  bool begin();                      // true when WHO_AM_I answered and the FIFO runs; false leaves present() false
  bool present() const { return present_; }
  // Drains up to maxSamples whole samples into out (kBytesPerSample each) in one burst. Returns the count, or -1 on a
  // bus error, after which present() is false and begin() must be called again. Restarts the FIFO after an overflow
  // (returns 0). Every restart, including begin(), bumps resets(): samples before and after it are not contiguous.
  int read(uint8_t* out, int maxSamples);
  uint8_t resets() const { return resets_; }

 private:
  bool write(uint8_t reg, uint8_t value);
  bool readRegs(uint8_t reg, uint8_t* buf, int len);
  bool present_ = false;
  uint8_t resets_ = 0;
};

}  // namespace gait
