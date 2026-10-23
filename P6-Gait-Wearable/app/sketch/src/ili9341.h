#pragma once
#include <stdint.h>

namespace gait {

// Minimal ILI9341 driver over hardware SPI (320x240 landscape, RGB565), enough for the WHOOP HR screen: filled rectangles
// and a scalable 5x7 font (digits, A-Z, space - . / :). Wiring: SCK D13, MOSI D11, CS/DC/RST as given to begin().
class Ili9341 {
 public:
  static const int kWidth = 320;
  static const int kHeight = 240;
  static const int kGlyphW = 5;
  static const int kGlyphH = 7;

  void begin(int cs, int dc, int rst);
  void fillScreen(uint16_t color) { fillRect(0, 0, kWidth, kHeight, color); }
  void fillRect(int x, int y, int w, int h, uint16_t color);
  // Draws text at (x, y) with each font pixel `scale` wide, one scaled column of spacing between glyphs, on `bg`.
  void drawText(int x, int y, const char* text, int scale, uint16_t color, uint16_t bg);
  static int textWidth(int chars, int scale) { return chars > 0 ? chars * (kGlyphW + 1) * scale - scale : 0; }
  static int textHeight(int scale) { return kGlyphH * scale; }
  static uint16_t rgb(uint8_t r, uint8_t g, uint8_t b) { return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3); }

 private:
  void command(uint8_t c);
  void data(uint8_t d);
  void data(const uint8_t* buf, int len);
  void window(int x, int y, int w, int h);
  void pushColor(uint16_t color, int count);
  static const uint8_t* glyph(char c);
  int cs_ = 10, dc_ = 9, rst_ = 8;
};

}  // namespace gait
