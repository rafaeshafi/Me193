"""The cast of the island: the three opponents (one per level card) and an avatar for every player, made from their name.

A Look is data, how to draw it is characters.py: skin, hair and its style, shirt, an accent colour and an accessory.  The
characters are original cartoon people in the console-sports spirit (round heads, big friendly eyes), not anybody's property; the
exception is the hardest opponent, who is drawn after a photo of Prof. Rogers (the course's own, at the player's request).
"""

import hashlib
from dataclasses import dataclass

from pingpong import levels

HAIR_STYLES = ("short", "spiky", "curly", "bob", "bun")                       # what a player's avatar is made from
OPPONENT_HAIR_STYLES = HAIR_STYLES + ("wavy",)                                  # the opponents may have a style of their own
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
    eyes: tuple | None = None   # the colour of the irises; None: the usual round dark eyes
    wide_smile: bool = False    # a broad smile with all the teeth in it, whenever the mood is a happy one
    mature: bool = False        # an older face: laugh lines, thinner arched brows, a faint blush
    head_ry: float = 31.0       # how tall the head is (a radius, in the units of the portrait): a longer face is a larger one
    collar: str = "v"           # "v": the sporty V with a stripe along the shoulders; "polo": a polo shirt's two flaps and its placket


PIP = Look("PIP", "Let's rally!", rgb(244, 196, 150), rgb(190, 90, 40), "short", rgb(0, 168, 168), rgb(255, 214, 51),
           accessory="cap", freckles=True)
COCO = Look("COCO", "Bring your best!", rgb(150, 98, 62), rgb(60, 35, 60), "curly", rgb(255, 99, 132), rgb(255, 235, 120),
            accessory="headband")
ROGERS = Look("ROGERS", "Class is in session.", rgb(238, 200, 178), rgb(112, 102, 94), "wavy", rgb(32, 34, 38), rgb(56, 59, 65),
              eyes=rgb(112, 160, 198), wide_smile=True, mature=True, collar="polo", head_ry=34.0)
OPPONENTS = {1: PIP, 2: COCO, 3: ROGERS}

SKINS = tuple(rgb(*c) for c in ((255, 224, 196), (241, 194, 150), (224, 172, 120), (198, 134, 86), (141, 85, 52), (96, 60, 40)))
HAIRS = tuple(rgb(*c) for c in ((35, 30, 30), (80, 50, 30), (125, 80, 45), (225, 185, 90), (190, 90, 40), (120, 70, 160),
                                 (40, 150, 150), (230, 120, 170)))
SHIRTS = tuple(rgb(*c) for c in ((255, 120, 90), (255, 200, 40), (80, 170, 255), (100, 210, 150), (150, 100, 230),
                                  (255, 150, 40), (150, 210, 60), (240, 90, 160)))
TRIMS = tuple(rgb(*c) for c in ((255, 255, 255), (255, 232, 120), (40, 60, 130), (255, 120, 90)))


def opponent_by_name(level_name):
    """The opponent for the level's name as the HUD carries it (Rookie, Club, Pro, Insane)."""
    return {"Rookie": PIP, "Club": COCO, "Pro": ROGERS, "Insane": ROGERS}.get(level_name, PIP)


def opponent_look(level_name, friend=""):
    """Whoever stands behind the table: the friend being played online (their own avatar), else the opponent of the level."""
    return player_look(friend) if friend else opponent_by_name(level_name)


def opponent_for(level):
    """The opponent that stands for a level (the Insane row, which has no card, borrows the hardest one)."""
    return OPPONENTS.get(level.tag, ROGERS)


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
