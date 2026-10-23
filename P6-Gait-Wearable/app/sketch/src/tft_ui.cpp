#include "tft_ui.h"

#include <stdlib.h>
#include <string.h>

#include "tft_gfx.h"

namespace gait {

namespace {
using brand::kBlack; using brand::kWhite; using brand::kGrey;
const uint16_t kGreen = brand::kRecGreen;
const uint16_t kAmber = brand::kRecYellow;
const uint16_t kRed = brand::kRecRed;
const uint16_t kBlue = brand::kStrain;
const uint16_t kOrange = Ili9341::rgb(255, 120, 0);
const uint16_t kTrace = Ili9341::rgb(0, 230, 230);
const int kOneG = 106;  // (1 g + 4) / 12 * 255
const int kRightX = 256;

bool take(const char*& p, const char* key, char* out, int cap) {
  // copies the value of "key=" if the pair at p has that key; always advances p past the pair
  const char* eq = strchr(p, '=');
  const char* end = strchr(p, ';');
  if (!eq) return false;
  if (!end) end = p + strlen(p);
  if (eq > end) return false;
  bool match = (int)strlen(key) == eq - p && strncmp(p, key, eq - p) == 0;
  if (match && out) {
    int n = (int)(end - (eq + 1));
    if (n > cap - 1) n = cap - 1;
    memcpy(out, eq + 1, n);
    out[n] = '\0';
  }
  p = *end ? end + 1 : end;
  return match;
}
}  // namespace

bool TftUi::parse(const char* payload, Fields& f) {
  memset(&f, 0, sizeof f);
  strcpy(f.act, "---"); strcpy(f.m1, "---"); strcpy(f.cad, "--"); strcpy(f.pk, "--");
  f.agree = true;
  const char* p = payload;
  bool any = false;
  char num[8];
  while (*p) {
    const char* start = p;
    if (take(p, "act", f.act, sizeof f.act) || take((p = start), "m1", f.m1, sizeof f.m1) ||
        take((p = start), "cad", f.cad, sizeof f.cad) || take((p = start), "pk", f.pk, sizeof f.pk) ||
        take((p = start), "prompt", f.prompt, sizeof f.prompt) || take((p = start), "st", f.st, sizeof f.st)) {
      any = true;
    } else if (take((p = start), "agree", num, sizeof num)) {
      f.agree = atoi(num) != 0;
      any = true;
    } else {
      p = start;
      take(p, "", nullptr, 0);  // unknown key: skip the pair
      if (p == start) break;
    }
  }
  return any;
}

uint16_t TftUi::colorFor(const char* act) {
  if (strcmp(act, "STAND") == 0) return kWhite;
  if (strcmp(act, "WALK") == 0) return kGreen;
  if (strcmp(act, "RUN") == 0) return kOrange;
  if (strcmp(act, "BIKE") == 0) return kBlue;
  if (strcmp(act, "NO IMU") == 0) return kRed;
  return kGrey;
}

int TftUi::yOf(int sample) { return kWaveY + kWaveH - 2 - sample * (kWaveH - 4) / 255; }

void TftUi::begin(int cs, int dc, int rst) {
  lcd_.begin(cs, dc, rst);
  parse("act=---;m1=---", want_);
  first_ = true;
  dirty_ = true;
}

bool TftUi::setPayload(const char* payload) {
  Fields f;
  if (!parse(payload, f)) return false;
  want_ = f;
  dirty_ = true;
  return true;
}

void TftUi::field(int x, int y, int scale, const char* text, int maxChars, uint16_t color) {
  lcd_.fillRect(x, y, Ili9341::textWidth(maxChars, scale) + scale, Ili9341::textHeight(scale), kBlack);
  lcd_.drawText(x, y, text, scale, color, kBlack);
}

void TftUi::drawStatic() {
  lcd_.fillScreen(kBlack);
  lcd_.drawText(8, 50, "MODEL 2: ALL SIX IMU CHANNELS", 1, kGrey, kBlack);
  lcd_.drawText(8, 62, "MODEL 1: PEAK G ONLY", 1, kGrey, kBlack);
  lcd_.drawText(kRightX, kWaveY, "CADENCE", 1, kGrey, kBlack);
  lcd_.drawText(kRightX, kWaveY + 48, "PEAK G", 1, kGrey, kBlack);
  lcd_.fillRect(kWaveX, kWaveY, kWaveW, 1, kGrey);
  lcd_.fillRect(kWaveX, kWaveY + kWaveH - 1, kWaveW, 1, kGrey);
  lcd_.fillRect(kWaveX, yOf(kOneG), kWaveW, 1, brand::kTrack);
  waveCol_ = 0;
  waveLast_ = -1;
}

void TftUi::pushWave(const uint8_t* samples, int count) {
  if (first_) return;  // the static frame is not up yet
  for (int i = 0; i < count; ++i) {
    int x = kWaveX + waveCol_;
    int y = yOf(samples[i]);
    int y0 = waveLast_ < 0 ? y : yOf(waveLast_);
    lcd_.fillRect(x, kWaveY + 1, 1, kWaveH - 2, kBlack);
    lcd_.fillRect(x, yOf(kOneG), 1, 1, brand::kTrack);
    int top = y < y0 ? y : y0;
    int h = (y > y0 ? y - y0 : y0 - y) + 1;
    lcd_.fillRect(x, top, 1, h, kTrace);
    waveLast_ = samples[i];
    waveCol_ = (waveCol_ + 1) % kWaveW;
    lcd_.fillRect(kWaveX + waveCol_, kWaveY + 1, 1, kWaveH - 2, kBlack);  // gap ahead of the cursor
  }
}

void TftUi::render() {
  if (!dirty_) return;
  dirty_ = false;
  bool full = first_;
  if (first_) {
    drawStatic();
    first_ = false;
  }
  const Fields& w = want_;
  Fields& s = shown_;
  if (full || strcmp(w.act, s.act) != 0) field(8, 4, 6, w.act, 6, colorFor(w.act));
  if (full || strcmp(w.m1, s.m1) != 0 || w.agree != s.agree) {
    field(8, 72, 2, w.m1, 6, w.agree ? kGreen : kAmber);
    lcd_.fillRect(90, 72, 120, 14, kBlack);
    lcd_.drawText(90, 72, w.agree ? "AGREES" : "DIFFERS", 2, w.agree ? kGreen : kAmber, kBlack);
  }
  if (full || strcmp(w.prompt, s.prompt) != 0) field(8, 92, 2, w.prompt, 20, kWhite);
  if (full || strcmp(w.cad, s.cad) != 0) field(kRightX, kWaveY + 12, 3, w.cad, 3, kWhite);
  if (full || strcmp(w.pk, s.pk) != 0) field(kRightX, kWaveY + 60, 3, w.pk, 3, kWhite);
  if (full || strcmp(w.st, s.st) != 0) field(kRightX, kWaveY + 100, 1, w.st, 9, kAmber);
  s = w;
}

}  // namespace gait
