#pragma once
#include <stddef.h>
#include <stdint.h>

namespace pingpong {

const int kRows = 8;
const int kCols = 13;
const int kPixels = kRows * kCols;
const uint8_t kBrightest = 7;  // the matrix is set to 3 bits of brightness

// Copies the levels the Linux side sends (one byte for each of the 104 pixels, row by row) into `levels`, which the matrix can be
// given as it is: a level above the brightest is the brightest, and a frame too short to say a pixel leaves it dark.
void normalizeFrame(const uint8_t* values, size_t length, uint8_t* levels);

}  // namespace pingpong
