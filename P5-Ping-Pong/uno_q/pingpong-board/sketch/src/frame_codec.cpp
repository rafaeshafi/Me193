#include "frame_codec.h"

namespace pingpong {

void decodeFrame(const char* text, size_t length, uint8_t* levels) {
  for (int i = 0; i < kPixels; ++i) {
    char c = static_cast<size_t>(i) < length ? text[i] : '0';
    levels[i] = (c >= '0' && c <= '7') ? static_cast<uint8_t>(c - '0') : 0;
  }
}

}  // namespace pingpong
