// Gait wearable, the MCU side. Dumb and fast; all classification lives in the Python app over the Bridge.
//   IMU:  the MPU6050's FIFO is drained in loop(); every 10 samples (100 ms) a batch goes to Python with
//         Bridge.notify("imu", bytes): byte 0 = sequence number, byte 1 = sample count, byte 2 = FIFO reset counter
//         (changes when samples stop being contiguous), then 12 bytes per sample (ax ay az gx gy gz, int16 big-endian,
//         raw). An empty batch every 500 ms says "alive, no IMU".
//   TFT:  provider "tft" takes one short key=value payload (see tft_ui.h); "tfth" takes waveform bytes to sweep.
// Wiring: TFT SCK D13, MOSI D11, CS D10, DC D9, RST D8, VCC 5V, LED 3V3; MPU6050 on the SDA/SCL header at 3V3, AD0 to GND.

#include <Arduino_RouterBridge.h>
#include <Wire.h>
#include <vector>

#include "src/mpu6050.h"
#include "src/tft_ui.h"

const int kTftCs = 10, kTftDc = 9, kTftRst = 8;
const int kBatch = 10;  // 10 x 12 B + 3 B header = 123 B, well under the Bridge's message size limit
const int kHeader = 3;
const uint32_t kProbeMs = 2000, kHeartbeatMs = 500;

gait::Mpu6050 imu;
gait::TftUi tft;
// The providers below are registered with provide_safe, so they run between loop() calls on the same thread and these
// buffers need no locking.
static char tftPayload[256];
static bool tftDirty = false;
static uint8_t wave[64];
static int waveCount = -1;  // -1 = nothing new
static uint8_t batch[kHeader + kBatch * gait::Mpu6050::kBytesPerSample];
static int batched = 0;
static uint8_t seq = 0, sentResets = 0;
static uint32_t nextProbeMs = 0, nextHeartbeatMs = 0;

bool showTft(MsgPack::str_t payload) {
  strncpy(tftPayload, payload.c_str(), sizeof tftPayload - 1);
  tftPayload[sizeof tftPayload - 1] = '\0';
  tftDirty = true;
  return true;
}

bool pushWave(std::vector<uint8_t> samples) {
  int n = samples.size() > sizeof wave ? (int)sizeof wave : (int)samples.size();
  memcpy(wave, samples.data() + (samples.size() - n), n);
  waveCount = n;
  return true;
}

void flush() {
  batch[0] = seq++;
  batch[1] = (uint8_t)batched;
  batch[2] = sentResets;
  std::vector<uint8_t> msg(batch, batch + kHeader + batched * gait::Mpu6050::kBytesPerSample);
  Bridge.notify("imu", msg);
  batched = 0;
  nextHeartbeatMs = millis() + kHeartbeatMs;
}

void setup() {
  tft.begin(kTftCs, kTftDc, kTftRst);
  Bridge.begin();
  Bridge.provide_safe("tft", showTft);
  Bridge.provide_safe("tfth", pushWave);
}

void loop() {
  uint32_t now = millis();
  if (!imu.present()) {
    if ((int32_t)(now - nextProbeMs) >= 0) {
      nextProbeMs = now + kProbeMs;
      imu.begin();
    }
  } else {
    int n;  // drain everything that is waiting, however long the last render took
    while ((n = imu.read(batch + kHeader + batched * gait::Mpu6050::kBytesPerSample, kBatch - batched)) > 0) {
      batched += n;
      if (batched >= kBatch) flush();
    }
  }
  if (imu.resets() != sentResets) {  // the stream restarted: what is batched belongs to the old stream
    if (batched > 0) flush();
    sentResets = imu.resets();
  }
  if ((int32_t)(now - nextHeartbeatMs) >= 0) flush();  // partial batch, or empty = alive but no IMU
  if (tftDirty) {
    tftDirty = false;
    tft.setPayload(tftPayload);
  }
  if (waveCount >= 0) {
    int n = waveCount;
    waveCount = -1;
    tft.pushWave(wave, n);
  }
  tft.render();
  delay(5);
}
