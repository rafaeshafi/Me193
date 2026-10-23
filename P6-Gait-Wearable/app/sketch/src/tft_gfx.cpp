#include "tft_gfx.h"

#include <math.h>

namespace gait {

uint16_t brand::recoveryColor(int rec) {
  if (rec < 0) return kGrey;
  return rec >= 67 ? kRecGreen : rec >= 34 ? kRecYellow : kRecRed;
}

Mood moodFor(int recovery) {
  if (recovery < 0) return Mood::Unknown;
  return recovery >= 67 ? Mood::Good : recovery >= 34 ? Mood::Middling : Mood::Bad;
}

void drawRing(Ili9341& lcd, int cx, int cy, int r, int thick, float fraction, uint16_t color) {
  if (fraction < 0) fraction = 0;
  if (fraction > 1) fraction = 1;
  const float lit = fraction * 2.0f * (float)M_PI;
  const int ro2 = r * r, ri2 = (r - thick) * (r - thick);
  for (int dy = -r; dy <= r; ++dy) {
    for (int dx = -r; dx <= r; ++dx) {
      int d2 = dx * dx + dy * dy;
      if (d2 > ro2 || d2 < ri2) continue;
      float a = atan2f((float)dx, (float)-dy);  // 0 at the top, clockwise
      if (a < 0) a += 2.0f * (float)M_PI;
      lcd.fillRect(cx + dx, cy + dy, 1, 1, a <= lit && fraction > 0 ? color : brand::kTrack);
    }
  }
}

namespace {
// The swimmer in 20x28 cells. W cap/goggle frame, S skin, L lens, B beard, K dark line, N nose shade, . empty.
// Row 7 (eyebrows) and rows 18-19 (mouth) are overwritten per mood.
const char* const kFigure[28] = {
    "......WWWWWWWW......",
    "....WWWWWWWWWWWW....",
    "...WWWWWWWWWWWWWW...",
    "..WWWWWWWWWWWWWWWW..",
    "..WWWWWWWWWWWWWWWW..",
    "..WWWWWWWWWWWWWWWW..",
    "..SSSSSSSSSSSSSSSS..",
    "..SSSSSSSSSSSSSSSS..",
    "..SSSSSSSSSSSSSSSS..",
    ".WWWWWWWWWWWWWWWWWW.",
    ".WLLLLLLWWWWLLLLLLW.",
    ".WLLLLLLWWWWLLLLLLW.",
    ".WLLLLLLWWWWLLLLLLW.",
    "..WWWWWWWWWWWWWWWW..",
    "..SSSSSSSSSSSSSSSS..",
    "..SSSSSSSSNNSSSSSS..",
    "..SSSSSSSNSSNSSSSS..",
    "..SBSSSSSSSSSSSSBS..",
    "..SBBSSSSSSSSSSBBS..",
    "..SBBBSSSSSSSSBBBS..",
    "...BBBBSSSSSSBBBB...",
    "...BBBBBBBBBBBBBB...",
    "....BBBBBBBBBBBB....",
    ".....BBBBBBBBBB.....",
    "......SSSSSSSS......",
    "......SSSSSSSS......",
    ".....SSSSSSSSSS.....",
    "....SSSSSSSSSSSS....",
};

uint16_t cellColor(char c) {
  switch (c) {
    case 'W': return brand::kWhite;
    case 'S': return Ili9341::rgb(205, 150, 110);
    case 'L': return Ili9341::rgb(40, 175, 215);
    case 'B': return Ili9341::rgb(55, 38, 30);
    case 'K': return Ili9341::rgb(30, 20, 15);
    case 'N': return Ili9341::rgb(170, 115, 85);
    case 'M': return Ili9341::rgb(225, 120, 130);
    default: return brand::kBlack;
  }
}

void cell(Ili9341& lcd, int x, int y, int scale, int col, int row, char c) {
  uint16_t color = cellColor(c);
  if (c == 'B') color = ((row + col) & 1) ? Ili9341::rgb(150, 105, 80) : cellColor('S');  // light stubble: dithered
  lcd.fillRect(x + col * scale, y + row * scale, scale, scale, color);
}
}  // namespace

void drawFigure(Ili9341& lcd, int x, int y, int scale, Mood mood) {
  for (int row = 0; row < 28; ++row)
    for (int col = 0; col < 20; ++col) cell(lcd, x, y, scale, col, row, kFigure[row][col]);
  // eyebrows: good/middling flat and relaxed; bad: inner ends dropped (tired)
  if (mood == Mood::Bad) {
    for (int c = 3; c <= 5; ++c) cell(lcd, x, y, scale, c, 7, 'K');
    for (int c = 6; c <= 7; ++c) cell(lcd, x, y, scale, c, 8, 'K');
    for (int c = 12; c <= 13; ++c) cell(lcd, x, y, scale, c, 8, 'K');
    for (int c = 14; c <= 16; ++c) cell(lcd, x, y, scale, c, 7, 'K');
  } else {
    for (int c = 3; c <= 7; ++c) cell(lcd, x, y, scale, c, 7, 'K');
    for (int c = 12; c <= 16; ++c) cell(lcd, x, y, scale, c, 7, 'K');
  }
  // mouth
  switch (mood) {
    case Mood::Good:
      cell(lcd, x, y, scale, 6, 18, 'M'); cell(lcd, x, y, scale, 13, 18, 'M');
      for (int c = 7; c <= 12; ++c) cell(lcd, x, y, scale, c, 19, 'M');
      break;
    case Mood::Bad:
      for (int c = 7; c <= 12; ++c) cell(lcd, x, y, scale, c, 18, 'M');
      cell(lcd, x, y, scale, 6, 19, 'M'); cell(lcd, x, y, scale, 13, 19, 'M');
      break;
    case Mood::Middling:
      for (int c = 7; c <= 12; ++c) cell(lcd, x, y, scale, c, 18, 'M');
      break;
    case Mood::Unknown:
      for (int c = 8; c <= 11; ++c) cell(lcd, x, y, scale, c, 18, 'M');  // pursed, like the photo
      cell(lcd, x, y, scale, 9, 19, 'M'); cell(lcd, x, y, scale, 10, 19, 'M');
      break;
  }
}

}  // namespace gait
