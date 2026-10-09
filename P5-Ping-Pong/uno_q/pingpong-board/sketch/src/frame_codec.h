#pragma once
#include <stddef.h>
#include <stdint.h>

namespace pingpong {

const int kRows = 8;
const int kCols = 13;
const int kPixels = kRows * kCols;

// Turns the text the Linux side sends (one character '0'..'7' for each of the 104 pixels, row by row) into the matrix's brightness
// levels. Anything else, and anything the text is too short to say, is dark.
void decodeFrame(const char* text, size_t length, uint8_t* levels);

}  // namespace pingpong
