// PingPong scoreboard, the MCU side: draws on the UNO Q's 8x13 LED matrix the frame the Linux side sends it over the Bridge (the
// game on the Mac works out what to show; this only draws it). It is written the way Arduino's own LED matrix example app is: one
// provider, "draw", that takes the 104 brightness levels as bytes and hands them to the matrix, set to 3 bits (0..7). Only the
// provider draws, so nothing else needs to be kept out of the matrix driver's way.

#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include <vector>

#include "src/frame_codec.h"

Arduino_LED_Matrix matrix;
static uint8_t levels[pingpong::kPixels];  // static: the matrix driver may keep the pointer it is given

void draw(std::vector<uint8_t> frame) {
  pingpong::normalizeFrame(frame.data(), frame.size(), levels);
  matrix.draw(levels);
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);  // brightness 0..7
  matrix.clear();
  Bridge.begin();
  Bridge.provide("draw", draw);
}

void loop() { delay(10); }
