"""Everything you might want to change on game day lives in this file.

Edit a value, save, rerun whistle_car.py. Nothing else needs touching.
"""

# --- MQTT -----------------------------------------------------------------
# Agree these strings with your opponent before the match. They just have to
# match on both laptops; any text works.
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "ME193/Rogers"

MSG_START = "start"         # sent by the instructor: begin driving
MSG_CAUGHT = "ball:caught"  # ball publishes this when the goalie reaches its sensor
MSG_GOAL = "ball:goal"      # ball publishes this after the goal whistle

# --- Sounds ---------------------------------------------------------------
# Each song is EITHER a path to a .wav file, e.g. "sounds/sad_trombone.wav"
# (relative paths are looked up next to this file), OR a list of
# (note, beats) pairs. Notes are names like "C4", "F#5", "Bb3"; "R" is a rest.
# Try one out with:  python songs.py death   (or victory)
TEMPO_BPM = 160

DEATH_SONG = [  # a slow sinking "wah wah wah waaah"
    ("G4", 1.5), ("F#4", 1.5), ("F4", 1.5), ("E4", 4),
]

VICTORY_SONG = [  # a quick fanfare
    ("C5", 0.5), ("E5", 0.5), ("G5", 0.5), ("C6", 1.5),
    ("R", 0.25), ("G5", 0.5), ("C6", 2.5),
]

# --- LEGO hardware --------------------------------------------------------
# The Connection Card pairs every device in the kit (motor and colour sensor).
CARD_COLOR = "green"
CARD_SERIAL = "0997"

MOTOR_DIRECTION = -1  # flip to 1 if the car drives backwards on "straight"
BASE_SPEED = 50       # motor % when whistling the middle note
TURN_GAIN = 1.0       # 1.0: the inside wheel stops at a full turn; 2.0: it reverses

# Goalie detection: the colour sensor faces forward into open air, so its
# reflection reading is near zero. Something close in front makes it jump.
REFLECT_DELTA = 20    # rise above the reading taken at "start" that counts as caught
REFLECT_HOLD = 0.1    # s it must stay high, so a single glitchy reading is ignored

# --- Whistle bands (Hz) ---------------------------------------------------
# Defaults only: `python whistle_car.py --calibrate` measures your own whistle
# and saves the result to calibration.json, which overrides these.
F_MIN = 1100      # lowest whistle: hardest right turn
F_CENTER = 1500   # comfortable middle whistle: straight
F_MAX = 2200      # highest steering whistle: hardest left turn
F_GOAL = 2800     # hold at or above this to claim a goal
DEAD_BAND = 80    # +/- Hz around F_CENTER that still counts as straight
GOAL_HOLD = 0.75  # seconds the goal whistle must be held
