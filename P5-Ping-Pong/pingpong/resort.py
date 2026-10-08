"""The resort: a sunny seaside terrace on a pier with an island behind it, drawn through any camera.

The game looks at the terrace from behind the player's end of the table (scene.py draws the table, the net and the opponent
on top of this); the intro flies the same camera in from the clouds over the island and down onto the deck.  Everything is a
flat bright shape with a soft shade, in a clean console-sports style, and nothing is loaded from disk.
"""

from pingpong import cast, characters, resort_land as land, resort_sky as sky
from pingpong.resort_land import (BUNTING, DECK, DECK_ALT, DECK_EDGE, DECK_SEAM, DECK_X, DECK_Y, DECK_Z0, DECK_Z1, GRASS,  # noqa: F401
                                  ISLAND, RAIL, RAIL_TOP_Y, RAIL_Z, SAND)

CROWD = (("sam", -3.0, 5.1, "happy"), ("ana", -2.25, 5.0, "happy"), ("kai", -1.45, 4.6, "cheer"),
         ("dee", 1.45, 4.7, "cheer"), ("li", 2.25, 5.1, "grin"), ("zed", 3.0, 5.2, "happy"))
PLANTS_FAR = ((-4.3, 4.7), (4.3, 4.7))
PLANTS_NEAR = ((-4.3, -3.3), (4.3, -3.3))


def crowd(frame, cam, t_s):
    """The people watching from the deck, further ones first."""
    for name, x, z, mood in sorted(CROWD, key=lambda c: -cam.depth_of(c[1], land.DECK_Y, c[2])):
        characters.draw_spectator(frame, cam, cast.player_look(name), x_m=x, z_m=z, mood=mood, t=t_s + 0.37 * len(name))


def draw_backdrop(frame, cam, t_s=0.0, *, island=True):
    """Everything behind and around the table: sky, sun, sea, far islands, clouds, the island, the pier, the terrace, its
    railing and bunting, plants and the crowd."""
    sky.sky(frame, cam)
    sky.sun(frame, cam)
    sky.sea(frame, cam, t_s)
    sky.sparkles(frame, cam, t_s)
    sky.far_scenery(frame, cam, t_s)
    sky.clouds(frame, cam, t_s, far=True)
    if island:
        land.island(frame, cam, t_s)
    sky.clouds(frame, cam, t_s, far=False)
    land.walkway(frame, cam)
    land.deck(frame, cam)
    land.table_shadow(frame, cam)
    land.rails_far(frame, cam)
    land.bunting(frame, cam, t_s)
    for x, z in PLANTS_FAR:
        land.plant(frame, cam, x, z)
    crowd(frame, cam, t_s)


def draw_foreground(frame, cam, t_s=0.0):
    """What is in front of the table as the intro comes in: the railing on the walkway side, plants and gulls."""
    land.rails_near(frame, cam)
    for x, z in PLANTS_NEAR:
        land.plant(frame, cam, x, z)
    sky.birds(frame, cam, t_s)
