"""The cast of the island: the three opponents (one per level card) and an avatar for every player, made from their name.

A Look is data, how to draw it is characters.py: skin, hair and its style, shirt, an accent colour and an accessory.  All the
characters are original cartoon people in the console-sports spirit (round heads, big friendly eyes), not anybody's property.
"""

import hashlib
from dataclasses import dataclass

from pingpong import levels

HAIR_STYLES = ("short", "spiky", "curly", "bob", "bun")
ACCESSORIES = ("none", "glasses", "sunglasses", "headband", "cap")


def rgb(r, g, b):
    """The colours below are written as RGB, as everyone reads them; the frames are BGR."""
    return (b, g, r)


@dataclass(frozen=True)
class Look:
    name: str
    tagline: str
    skin: tuple
    hair: tuple
    hair_style: str
    shirt: tuple
    trim: tuple                 # the accent: collar, stripe, the cap or the headband
    accessory: str = "none"
    freckles: bool = False


PIP = Look("PIP", "Let's rally!", rgb(244, 196, 150), rgb(190, 90, 40), "short", rgb(0, 168, 168), rgb(255, 214, 51),
           accessory="cap", freckles=True)
COCO = Look("COCO", "Bring your best!", rgb(150, 98, 62), rgb(60, 35, 60), "curly", rgb(255, 99, 132), rgb(255, 235, 120),
            accessory="headband")
MAX = Look("MAX", "Blink and you'll miss it.", rgb(255, 224, 196), rgb(30, 30, 38), "spiky", rgb(38, 58, 130),
           rgb(255, 196, 40), accessory="sunglasses")
OPPONENTS = {1: PIP, 2: COCO, 3: MAX}

SKINS = tuple(rgb(*c) for c in ((255, 224, 196), (241, 194, 150), (224, 172, 120), (198, 134, 86), (141, 85, 52), (96, 60, 40)))
HAIRS = tuple(rgb(*c) for c in ((35, 30, 30), (80, 50, 30), (125, 80, 45), (225, 185, 90), (190, 90, 40), (120, 70, 160),
                                 (40, 150, 150), (230, 120, 170)))
SHIRTS = tuple(rgb(*c) for c in ((255, 120, 90), (255, 200, 40), (80, 170, 255), (100, 210, 150), (150, 100, 230),
                                  (255, 150, 40), (150, 210, 60), (240, 90, 160)))
TRIMS = tuple(rgb(*c) for c in ((255, 255, 255), (255, 232, 120), (40, 60, 130), (255, 120, 90)))


def opponent_for(level):
    """The opponent that stands for a level (the Insane row, which has no card, borrows the hardest one)."""
    return OPPONENTS.get(level.tag, MAX)


def stars(level):
    """How hard an opponent is, as 1..5 stars each for how fast its balls come, how well it covers the table and its spin."""
    def scale(value, lo, hi):
        return max(1, min(5, round(1 + 4 * (value - lo) / (hi - lo))))

    hardest, easiest = levels.LEVELS[4], levels.LEVELS[1]
    return {"speed": scale(level.v_tier, easiest.v_tier, hardest.v_tier),
            "reach": scale(level.cpu_speed_ms, easiest.cpu_speed_ms, hardest.cpu_speed_ms),
            "spin": scale(level.spin_variety, easiest.spin_variety, hardest.spin_variety)}


def player_look(name):
    """An avatar for a player, always the same one for the same name (the case of the name does not matter)."""
    shown = name.strip().upper() or "PLAYER"
    digest = hashlib.md5(shown.lower().encode()).digest()
    pick = lambda options, byte: options[digest[byte] % len(options)]            # noqa: E731
    return Look(shown, "", pick(SKINS, 0), pick(HAIRS, 1), pick(HAIR_STYLES, 2), pick(SHIRTS, 3), pick(TRIMS, 4),
                accessory=pick(("none", "none", "glasses", "headband", "cap"), 5), freckles=digest[6] % 4 == 0)
