#pragma once
#include <stdint.h>

#include "ili9341.h"

namespace gait {

// The gait screen. Python sends one payload per change as key=value pairs joined by ';':
//   act=STAND|WALK|RUN|BIKE|NO IMU|---   (Model 2, all IMU channels: the big word)
//   m1=STAND|WALK|RUN|BIKE|---           (Model 1, peak g only)
//   agree=1|0  cad=123|--  pk=2.3|--  prompt=WALK 18s (empty hides it)  st=REC|--- (short status)
// The waveform arrives separately as bytes (pushWave): thigh-axis accel, 0..255 = -4..+8 g, newest last, 50 Hz.
// It is drawn as an oscilloscope sweep: one column per sample, cursor wraps at the right edge.
class TftUi {
 public:
  static const int kWaveX = 8, kWaveY = 112, kWaveW = 240, kWaveH = 120;
  void begin(int cs, int dc, int rst);
  bool setPayload(const char* payload);  // false if malformed (previous payload kept)
  void pushWave(const uint8_t* samples, int count);
  void render();  // call from loop(); redraws only fields that changed

 private:
  struct Fields {
    char act[8], m1[8], cad[6], pk[6], prompt[24], st[8];
    int agree;  // 1 agree, 0 differ, -1 neutral
  };
  static bool parse(const char* payload, Fields& out);
  static uint16_t colorFor(const char* act);
  static int yOf(int sample);
  void field(int x, int y, int scale, const char* text, int maxChars, uint16_t color);
  void drawStatic();
  Ili9341 lcd_;
  Fields want_{}, shown_{};
  bool dirty_ = true, first_ = true;
  int waveCol_ = 0, waveLast_ = -1;
};

}  // namespace gait
