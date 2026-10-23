"""What the TFT gets: the short key=value payload (provider "tft") and waveform bytes (provider "tfth")."""

WAVE_LO_G, WAVE_HI_G = -4.0, 8.0
ACTS = ("STAND", "WALK", "RUN", "BIKE")


def _safe(text, limit):
    return str(text).replace(";", " ").replace("=", " ")[:limit]


def payload(act, m1, cad, pk, prompt="", st=""):
    """Keys: act (Model 2), m1 (Model 1), agree, cad (per minute), pk (peak g), prompt (recording), st (status)."""
    real = act in ACTS and m1 in ACTS
    pairs = [
        ("act", _safe(act, 7)),
        ("m1", _safe(m1, 7)),
        ("agree", (1 if act == m1 else 0) if real else "-"),
        ("cad", "--" if cad is None else int(cad)),
        ("pk", "--" if pk is None else "%.1f" % pk),
        ("prompt", _safe(prompt, 20)),
        ("st", _safe(st, 7)),
    ]
    return ";".join("%s=%s" % kv for kv in pairs)


def wave_byte(ax):
    """Thigh-axis acceleration in g -> one sweep sample byte (0 = -4 g, 255 = +8 g)."""
    v = int(round((ax - WAVE_LO_G) * 255 / (WAVE_HI_G - WAVE_LO_G)))
    return max(0, min(255, v))


def wave_bytes(ax_values):
    return bytes(wave_byte(a) for a in ax_values)
