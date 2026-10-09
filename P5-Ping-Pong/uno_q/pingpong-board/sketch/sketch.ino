// PingPong scoreboard, the MCU side: draws on the UNO Q's 8x13 LED matrix the frame the Linux side sends it over the Bridge (the
// game on the Mac works out what to show; this only draws it). The handler is registered with provide_safe, so the Bridge runs it
// from loop() and not from its own thread: the matrix driver is not made to be called from two threads.

#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>

#include "src/frame_codec.h"

Arduino_LED_Matrix matrix;
static uint8_t levels[pingpong::kPixels];  // static: the matrix driver may keep the pointer it is given

void draw(String frame) {
  pingpong::decodeFrame(frame.c_str(), frame.length(), levels);
  matrix.draw(levels);
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);  // brightness 0..7
  Bridge.begin();
  Bridge.provide_safe("draw", draw);
}

void loop() { delay(5); }
