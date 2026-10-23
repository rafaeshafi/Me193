#pragma once
#include <stdint.h>

namespace gait {
class TftUi {
 public:
  void begin(int, int, int) {}
  bool setPayload(const char*) { return true; }
  void pushWave(const uint8_t*, int) {}
  void render() {}
};
}  // namespace gait
