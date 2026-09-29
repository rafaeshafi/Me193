"""Everything you might want to change on game day lives in this file.

Edit a value, save, rerun whistle_car.py. Nothing else needs touching.
"""

# --- MQTT -----------------------------------------------------------------
# Agree these strings with your opponent before the match. They just have to
# match on both laptops; any text works.
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "ME193"  # the ball publishes here; keep it unique to your team

MSG_START = "start"         # sent by the instructor: begin driving
MSG_CAUGHT = "ball:caught"  # ball publishes this when the goalie reaches its sensor
MSG_GOAL = "ball:goal"      # ball publishes this after the goal whistle

# What we ACCEPT from the other team's code. Their wording may differ from
# ours, so any phrase in these lists counts (upper/lower case and surrounding
# spaces ignored). Watch the terminal or mqtt_chat.py to see exactly what they
# send, then add it here.
HEAR_START = [MSG_START, "go"]
HEAR_GOAL = [MSG_GOAL, "goal", "scored", "ball scored", "we scored"]      # goalie stops: LOST
HEAR_CAUGHT = [MSG_CAUGHT, "caught", "fail", "failed", "ball failed"]    # goalie stops: WON
# Where the other team's ball publishes, if not on TOPIC (e.g. "ME193/TheirTeam").
OPPONENT_TOPIC = TOPIC

# Goalie team channel: our two laptops talk to each other here. Keep it unique
# to our team so another goalie pair can't move our glove.
TEAM_TOPIC = "ME193/Cucurella"
GLOVE_CMD = "glove"   # glove laptop -> robot laptop: "glove -45" (where the glove is, degrees)
STATE_CMD = "state"   # robot laptop -> glove laptop: "state DRIVING" (game state)

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
# The Connection Card pairs every device in the kit (motors and colour sensor).
CARD_COLOR = "green"
CARD_SERIAL = "0997"

MOTOR_DIRECTION = -1  # flip (1 / -1) if the car drives backwards on "straight"
SWAP_SIDES = True     # flip if the car turns right when the display says LEFT
BASE_SPEED = 50       # motor % when whistling the middle note
TURN_GAIN = 1.0       # 1.0: the inside wheel stops at a full turn; 2.0: it reverses

# Goalie detection: the colour sensor faces forward into open air, so its
# reflection reading is near zero. Something close in front makes it jump.
REFLECT_DELTA = 20    # rise above the reading taken at "start" that counts as caught
REFLECT_HOLD = 0.1    # s it must stay high, so a single glitchy reading is ignored

# Goalie robot: only slides forward/backward along the goal line, never turns.
# Above the middle note = forward, below = backward, faster the further from
# the middle; middle note or silence = stop.
GOALIE_SPEED = 60     # motor % at your highest / lowest whistle
GOALIE_DIRECTION = 1  # flip to -1 if a high whistle drives it backwards

# Goalie glove: a Single Motor standing upright with a big LEGO piece on it,
# from the same kit (same Connection Card). The GLOVE laptop connects to it and
# its whistle sets the angle: middle note = centre (0), higher = swing left, lower = swing
# right, silence = hold where it is. It is zeroed wherever it points at launch,
# so point the glove straight ahead before starting glove.py.
GLOVE_MAX_DEG = 90    # angle at your highest / lowest whistle
GLOVE_SPEED = 60      # motor % while swinging to a new angle
GLOVE_DIRECTION = 1   # flip to -1 if the glove swings right when it should go left
GLOVE_STEP = 5        # degrees; angles are rounded to this so pitch wobble doesn't twitch it

# --- Whistle bands (Hz) ---------------------------------------------------
# Defaults only: `python whistle_car.py --calibrate` measures your own whistle
# and saves the result to calibration.json, which overrides these.
F_MIN = 900       # lowest whistle: hardest right turn
F_CENTER = 1300   # comfortable middle whistle: straight
F_MAX = 2000      # highest steering whistle: hardest left turn
F_GOAL = 2600     # hold at or above this to claim a goal
DEAD_BAND = 100   # +/- Hz around F_CENTER that still counts as straight
GOAL_HOLD = 0.75  # seconds the goal whistle must be held

# --- Room noise -----------------------------------------------------------
# At launch the program listens to the room (stay quiet!) and learns its noise
# level at every frequency. Longer = a steadier estimate that catches more of
# the room's hums and fans, at the cost of a longer wait before driving.
NOISE_SECONDS = 3.0

# Only sounds louder than BOTH of these are read as a whistle; everything
# quieter is ignored. Raise them to react only to loud, close whistles; lower
# them if your real whistle shows "too quiet" / "not above room noise" in the
# window. The window's "gates" line shows the live values to compare against.
MIN_LOUDNESS = 0.03       # overall block loudness (RMS, 0..1); was 0.005
MIN_ABOVE_ROOM_DB = 25.0  # dB the whistle must beat the room noise by; was 15
