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
// The providers run on the Bridge's own thread; uiLock guards the hand-over of these buffers to loop().
static char tftPayload[256];
static bool tftDirty = false;
static uint8_t wave[64];
static int waveCount = -1;  // -1 = nothing new
static uint8_t batch[kHeader + kBatch * gait::Mpu6050::kBytesPerSample];
static int batched = 0;
static uint8_t seq = 0, sentResets = 0;
static uint32_t nextProbeMs = 0, nextHeartbeatMs = 0;
K_MUTEX_DEFINE(uiLock);

bool showTft(MsgPack::str_t payload) {
  k_mutex_lock(&uiLock, K_FOREVER);
  strncpy(tftPayload, payload.c_str(), sizeof tftPayload - 1);
  tftPayload[sizeof tftPayload - 1] = '\0';
  tftDirty = true;
  k_mutex_unlock(&uiLock);
  return true;
}

bool pushWave(std::vector<uint8_t> samples) {
  k_mutex_lock(&uiLock, K_FOREVER);
  int have = waveCount < 0 ? 0 : waveCount;
  int room = (int)sizeof wave - have;
  int n = (int)samples.size() < room ? (int)samples.size() : room;
  memcpy(wave + have, samples.data() + (samples.size() - n), n);
  waveCount = have + n;
  k_mutex_unlock(&uiLock);
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
  Bridge.provide("tft", showTft);
  Bridge.provide("tfth", pushWave);
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
  static char payloadCopy[sizeof tftPayload];
  static uint8_t waveCopy[sizeof wave];
  bool haveText = false;
  int haveWave = -1;
  k_mutex_lock(&uiLock, K_FOREVER);
  if (tftDirty) {
    tftDirty = false;
    memcpy(payloadCopy, tftPayload, sizeof payloadCopy);
    haveText = true;
  }
  if (waveCount >= 0) {
    haveWave = waveCount;
    waveCount = -1;
    memcpy(waveCopy, wave, haveWave);
  }
  k_mutex_unlock(&uiLock);
  if (haveText) tft.setPayload(payloadCopy);
  if (haveWave >= 0) tft.pushWave(waveCopy, haveWave);
  tft.render();
  delay(5);
}
