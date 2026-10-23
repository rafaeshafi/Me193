#include "ili9341.h"

#include <Arduino.h>
#include <SPI.h>

namespace gait {

namespace {
const uint32_t kSpiHz = 24000000;  // ILI9341 takes 40 MHz on a PCB; 24 is safe on jumper wires
const uint8_t kMadctlLandscape = 0x28;  // row/column exchange, BGR order (what these red modules want)

// 5x7 font, one byte per column, bit 0 = top row. Order: space - . / : 0-9 A-Z % (classic GLCD glyphs).
const uint8_t kFont[][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00},  // space
    {0x08, 0x08, 0x08, 0x08, 0x08},  // -
    {0x00, 0x60, 0x60, 0x00, 0x00},  // .
    {0x20, 0x10, 0x08, 0x04, 0x02},  // /
    {0x00, 0x36, 0x36, 0x00, 0x00},  // :
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, {0x00, 0x42, 0x7F, 0x40, 0x00}, {0x42, 0x61, 0x51, 0x49, 0x46},
    {0x21, 0x41, 0x45, 0x4B, 0x31}, {0x18, 0x14, 0x12, 0x7F, 0x10}, {0x27, 0x45, 0x45, 0x45, 0x39},
    {0x3C, 0x4A, 0x49, 0x49, 0x30}, {0x01, 0x71, 0x09, 0x05, 0x03}, {0x36, 0x49, 0x49, 0x49, 0x36},
    {0x06, 0x49, 0x49, 0x29, 0x1E},  // 0-9
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, {0x7F, 0x49, 0x49, 0x49, 0x36}, {0x3E, 0x41, 0x41, 0x41, 0x22},
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, {0x7F, 0x49, 0x49, 0x49, 0x41}, {0x7F, 0x09, 0x09, 0x09, 0x01},
    {0x3E, 0x41, 0x49, 0x49, 0x7A}, {0x7F, 0x08, 0x08, 0x08, 0x7F}, {0x00, 0x41, 0x7F, 0x41, 0x00},
    {0x20, 0x40, 0x41, 0x3F, 0x01}, {0x7F, 0x08, 0x14, 0x22, 0x41}, {0x7F, 0x40, 0x40, 0x40, 0x40},
    {0x7F, 0x02, 0x0C, 0x02, 0x7F}, {0x7F, 0x04, 0x08, 0x10, 0x7F}, {0x3E, 0x41, 0x41, 0x41, 0x3E},
    {0x7F, 0x09, 0x09, 0x09, 0x06}, {0x3E, 0x41, 0x51, 0x21, 0x5E}, {0x7F, 0x09, 0x19, 0x29, 0x46},
    {0x46, 0x49, 0x49, 0x49, 0x31}, {0x01, 0x01, 0x7F, 0x01, 0x01}, {0x3F, 0x40, 0x40, 0x40, 0x3F},
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, {0x3F, 0x40, 0x38, 0x40, 0x3F}, {0x63, 0x14, 0x08, 0x14, 0x63},
    {0x07, 0x08, 0x70, 0x08, 0x07}, {0x61, 0x51, 0x49, 0x45, 0x43},  // A-Z
    {0x23, 0x13, 0x08, 0x64, 0x62},  // %
};
}  // namespace

const uint8_t* Ili9341::glyph(char c) {
  if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  if (c == '-') return kFont[1];
  if (c == '.') return kFont[2];
  if (c == '/') return kFont[3];
  if (c == ':') return kFont[4];
  if (c >= '0' && c <= '9') return kFont[5 + (c - '0')];
  if (c >= 'A' && c <= 'Z') return kFont[15 + (c - 'A')];
  if (c == '%') return kFont[41];
  return kFont[0];
}

void Ili9341::command(uint8_t c) {
  digitalWrite(dc_, LOW);
  digitalWrite(cs_, LOW);
  SPI.transfer(c);
  digitalWrite(cs_, HIGH);
}

void Ili9341::data(uint8_t d) { data(&d, 1); }

void Ili9341::data(const uint8_t* buf, int len) {
  uint8_t tmp[32];
  digitalWrite(dc_, HIGH);
  digitalWrite(cs_, LOW);
  while (len > 0) {  // transfer() overwrites its buffer with the read-back, so copy
    int n = len > (int)sizeof tmp ? (int)sizeof tmp : len;
    memcpy(tmp, buf, n);
    SPI.transfer(tmp, n);
    buf += n;
    len -= n;
  }
  digitalWrite(cs_, HIGH);
}

void Ili9341::begin(int cs, int dc, int rst) {
  cs_ = cs; dc_ = dc; rst_ = rst;
  pinMode(cs_, OUTPUT); pinMode(dc_, OUTPUT); pinMode(rst_, OUTPUT);
  digitalWrite(cs_, HIGH);
  SPI.begin();
  SPI.beginTransaction(SPISettings(kSpiHz, MSBFIRST, SPI_MODE0));
  digitalWrite(rst_, HIGH); delay(5);
  digitalWrite(rst_, LOW); delay(20);
  digitalWrite(rst_, HIGH); delay(150);
  command(0x01); delay(150);                       // software reset
  command(0xCF); { uint8_t d[] = {0x00, 0x83, 0x30}; data(d, 3); }
  command(0xED); { uint8_t d[] = {0x64, 0x03, 0x12, 0x81}; data(d, 4); }
  command(0xE8); { uint8_t d[] = {0x85, 0x01, 0x79}; data(d, 3); }
  command(0xCB); { uint8_t d[] = {0x39, 0x2C, 0x00, 0x34, 0x02}; data(d, 5); }
  command(0xF7); data(0x20);
  command(0xEA); { uint8_t d[] = {0x00, 0x00}; data(d, 2); }
  command(0xC0); data(0x26);                       // power control 1
  command(0xC1); data(0x11);                       // power control 2
  command(0xC5); { uint8_t d[] = {0x35, 0x3E}; data(d, 2); }  // VCOM
  command(0xC7); data(0xBE);
  command(0x36); data(kMadctlLandscape);           // memory access control
  command(0x3A); data(0x55);                       // 16 bits per pixel
  command(0xB1); { uint8_t d[] = {0x00, 0x1B}; data(d, 2); }  // frame rate
  command(0xF2); data(0x08);                       // 3-gamma off
  command(0x26); data(0x01);                       // gamma curve 1
  command(0xB7); data(0x07);                       // entry mode
  command(0xB6); { uint8_t d[] = {0x0A, 0x82, 0x27, 0x00}; data(d, 4); }  // display function
  command(0x11); delay(120);                       // sleep out
  command(0x29); delay(20);                        // display on
  fillScreen(0x0000);
}

void Ili9341::window(int x, int y, int w, int h) {
  uint8_t c[4] = {(uint8_t)(x >> 8), (uint8_t)x, (uint8_t)((x + w - 1) >> 8), (uint8_t)(x + w - 1)};
  uint8_t p[4] = {(uint8_t)(y >> 8), (uint8_t)y, (uint8_t)((y + h - 1) >> 8), (uint8_t)(y + h - 1)};
  command(0x2A); data(c, 4);
  command(0x2B); data(p, 4);
  command(0x2C);
}

void Ili9341::pushColor(uint16_t color, int count) {
  uint8_t buf[64];
  for (int i = 0; i < 32; ++i) { buf[2 * i] = color >> 8; buf[2 * i + 1] = color & 0xFF; }
  digitalWrite(dc_, HIGH);
  digitalWrite(cs_, LOW);
  while (count > 0) {
    int n = count > 32 ? 32 : count;
    uint8_t tmp[64];
    memcpy(tmp, buf, 2 * n);
    SPI.transfer(tmp, 2 * n);
    count -= n;
  }
  digitalWrite(cs_, HIGH);
}

void Ili9341::fillRect(int x, int y, int w, int h, uint16_t color) {
  if (x < 0) { w += x; x = 0; }
  if (y < 0) { h += y; y = 0; }
  if (x + w > kWidth) w = kWidth - x;
  if (y + h > kHeight) h = kHeight - y;
  if (w <= 0 || h <= 0) return;
  window(x, y, w, h);
  pushColor(color, w * h);
}

void Ili9341::drawText(int x, int y, const char* text, int scale, uint16_t color, uint16_t bg) {
  for (const char* p = text; *p; ++p) {
    const uint8_t* g = glyph(*p);
    for (int col = 0; col < kGlyphW; ++col) {
      for (int row = 0; row < kGlyphH; ++row) {
        bool on = (g[col] >> row) & 1;
        fillRect(x + col * scale, y + row * scale, scale, scale, on ? color : bg);
      }
    }
    fillRect(x + kGlyphW * scale, y, scale, kGlyphH * scale, bg);  // inter-glyph gap
    x += (kGlyphW + 1) * scale;
  }
}

}  // namespace gait
