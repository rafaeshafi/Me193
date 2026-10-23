#pragma once
#include <stdint.h>

#include "ili9341.h"

namespace gait {

// WHOOP-styled drawing helpers for the TFT: the brand palette, ring gauges, and the pixel figure of the swimmer whose
// mouth and eyebrows follow his recovery (good / middling / bad).
namespace brand {
const uint16_t kBlack = 0x0000, kWhite = 0xFFFF;
const uint16_t kGrey = Ili9341::rgb(130, 130, 130);
const uint16_t kTrack = Ili9341::rgb(38, 38, 38);
const uint16_t kStrain = Ili9341::rgb(0, 147, 231);      // WHOOP strain blue
const uint16_t kRecGreen = Ili9341::rgb(22, 236, 6);     // WHOOP recovery colours
const uint16_t kRecYellow = Ili9341::rgb(255, 222, 0);
const uint16_t kRecRed = Ili9341::rgb(255, 0, 38);
const uint16_t kSleep = Ili9341::rgb(123, 161, 255);     // WHOOP sleep lavender
uint16_t recoveryColor(int rec);                         // -1 = unknown -> grey
}  // namespace brand

enum class Mood { Unknown, Good, Middling, Bad };
Mood moodFor(int recovery);  // -1 = unknown

// Ring gauge: track in dim grey, `fraction` (0..1) of the ring lit clockwise from the top.
void drawRing(Ili9341& lcd, int cx, int cy, int r, int thick, float fraction, uint16_t color);
// The swimmer, 20x28 cells at `scale` pixels per cell, top-left at (x, y).
void drawFigure(Ili9341& lcd, int x, int y, int scale, Mood mood);

}  // namespace gait
