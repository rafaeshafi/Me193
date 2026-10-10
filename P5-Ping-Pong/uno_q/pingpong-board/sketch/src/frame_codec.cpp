#include "frame_codec.h"

namespace pingpong {

void normalizeFrame(const uint8_t* values, size_t length, uint8_t* levels) {
  for (int i = 0; i < kPixels; ++i) {
    uint8_t v = (values != nullptr && static_cast<size_t>(i) < length) ? values[i] : 0;
    levels[i] = v > kBrightest ? kBrightest : v;
  }
}

}  // namespace pingpong
