"""Drawing the cast: a head-and-shoulders portrait with moods, and the opponent standing behind the far end of the table."""

import numpy as np
import pytest

from pingpong import cast, characters, court3d

W, H = 640, 480
BG = (90, 90, 90)


def blank(w=W, h=H):
    return np.full((h, w, 3), BG, dtype=np.uint8)


def count(frame, color):
    return int((frame == np.array(color, dtype=np.uint8)).all(axis=2).sum())


def changed(a, b):
    return (a != b).any(axis=2)


def bounds(mask):
    rows, cols = np.where(mask)
    return cols.min(), rows.min(), cols.max(), rows.max()


def bust(look, size=200, **kw):
    frame = blank()
    characters.draw_bust(frame, W // 2, H // 2, size, look, **kw)
    return frame


# --- the portrait ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("who", [cast.PIP, cast.COCO, cast.ROGERS, cast.player_look("rafae"), cast.player_look("maya")])
def test_a_portrait_shows_the_skin_the_hair_and_the_shirt_of_its_look(who):
    frame = bust(who)
    for color in (who.skin, who.hair, who.shirt):
        assert count(frame, color) > 150, color


def test_a_portrait_stays_inside_the_box_it_was_given():
    frame = bust(cast.PIP, size=240)
    x0, y0, x1, y1 = bounds(changed(frame, blank()))
    assert x0 >= W // 2 - 0.62 * 240 and x1 <= W // 2 + 0.62 * 240
    assert y0 >= H // 2 - 0.62 * 240 and y1 <= H // 2 + 0.55 * 240


def test_a_bigger_portrait_is_bigger():
    small = changed(bust(cast.PIP, size=120), blank()).sum()
    big = changed(bust(cast.PIP, size=240), blank()).sum()
    assert 3.2 < big / small < 4.8


@pytest.mark.parametrize("a, b", [("happy", "sad"), ("happy", "cheer"), ("sad", "surprised"), ("neutral", "smug"),
                                  ("happy", "grin")])
def test_a_mood_changes_the_face_and_leaves_the_shoulders_alone(a, b):
    fa, fb = bust(cast.COCO, mood=a, bob=False), bust(cast.COCO, mood=b, bob=False)
    diff = changed(fa, fb)
    assert diff.sum() > 60
    _, y0, _, y1 = bounds(diff)
    assert y1 < H // 2 + 0.22 * 200                                              # nothing changes below the chin
    assert y0 > H // 2 - 0.33 * 200                                              # ... or in the hair


def test_every_mood_can_be_drawn():
    for mood in characters.MOODS:
        assert changed(bust(cast.ROGERS, mood=mood), blank()).sum() > 5000


def test_the_eyes_close_for_a_blink_and_a_blink_is_short():
    assert not characters.eyes_closed(0.0) and characters.eyes_closed(3.05)
    blinking = sum(characters.eyes_closed(k / 100) for k in range(1000))
    assert 25 <= blinking <= 60                                                  # about 0.12 s in every 3.2 s
    open_eyes, shut = bust(cast.PIP, t=0.0, bob=False), bust(cast.PIP, t=3.05, bob=False)
    assert 20 < changed(open_eyes, shut).sum() < 900


def test_a_head_bobs_gently_and_the_shoulders_stay_put():
    still = bust(cast.PIP, t=0.0)
    moved = bust(cast.PIP, t=0.4)
    diff = changed(still, moved)
    assert 100 < diff.sum()
    assert not diff[H // 2 + int(0.34 * 200):].any()                              # below the neck: the shoulders do not move


@pytest.mark.parametrize("style", cast.OPPONENT_HAIR_STYLES)
def test_every_hair_style_draws_and_differs_from_the_others(style):
    import dataclasses

    base = dataclasses.replace(cast.PIP, accessory="none", hair_style="short")
    mine = bust(dataclasses.replace(base, hair_style=style), bob=False)
    assert changed(mine, blank()).sum() > 6000
    if style != "short":
        assert changed(mine, bust(base, bob=False)).sum() > 200


@pytest.mark.parametrize("accessory", [a for a in cast.ACCESSORIES if a != "none"])
def test_every_accessory_changes_the_picture(accessory):
    import dataclasses

    plain = bust(dataclasses.replace(cast.PIP, accessory="none"), bob=False)
    worn = bust(dataclasses.replace(cast.PIP, accessory=accessory), bob=False)
    assert changed(plain, worn).sum() > 150


# --- the opponent behind the table -----------------------------------------------------------------------------------------
@pytest.fixture
def cam():
    return court3d.Camera.for_frame(1280, 720)


def court():
    return np.full((720, 1280, 3), BG, dtype=np.uint8)


def test_the_opponent_stands_behind_the_far_end_of_the_table_and_gets_smaller_further_away(cam):
    near, far = court(), court()
    characters.draw_opponent(near, cam, cast.PIP, x_m=0.0, z_m=3.1, grip_px=(600, 205), body_rows=210)
    characters.draw_opponent(far, cam, cast.PIP, x_m=0.0, z_m=4.2, grip_px=(600, 150), body_rows=210)
    cx, cy, scale = characters.head_px(cam, 0.0, 3.1)
    assert abs(cx - 640) < 1 and cy < 120
    far_scale = characters.head_px(cam, 0.0, 4.2)[2]
    ratio = count(near, cast.PIP.hair) / count(far, cast.PIP.hair)                       # the hair: the head alone, not the arm
    assert 0.9 * (scale / far_scale) ** 2 < ratio < 1.1 * (scale / far_scale) ** 2
    assert count(near, cast.PIP.hair) > 100 and count(near, cast.PIP.skin) > 100


def test_the_head_is_where_it_is_projected_and_the_body_stops_at_the_table_edge(cam):
    frame = court()
    characters.draw_opponent(frame, cam, cast.ROGERS, x_m=0.0, z_m=3.1, grip_px=(500, 150), body_rows=210)
    cx, cy, _ = characters.head_px(cam, 0.0, 3.1)
    assert (frame[round(cy), round(cx)] != np.array(BG)).any()
    below = changed(frame[210:], court()[210:])
    assert below.sum() == 0 or bounds(below)[0] < 560                           # nothing of the body behind the table edge
    assert count(frame[208:210], cast.ROGERS.shirt) > 20                             # but it reaches the edge


def test_the_opponent_follows_the_paddle_sideways_a_little(cam):
    left, right = characters.head_px(cam, -0.4, 3.1)[0], characters.head_px(cam, 0.4, 3.1)[0]
    assert left < 640 - 20 and right > 640 + 20


def test_an_arm_reaches_from_the_shoulder_to_the_grip_on_the_paddle(cam):
    frame = court()
    grip = (560, 190)
    characters.draw_opponent(frame, cam, cast.COCO, x_m=0.0, z_m=3.1, grip_px=grip, body_rows=210)
    sx, sy = characters.shoulder_px(cam, 0.0, 3.1)
    mid = (round(sx + 0.4 * (grip[0] - sx)), round(sy + 0.4 * (grip[1] - sy)))
    assert tuple(frame[mid[1], mid[0]]) == cast.COCO.shirt
    end = (round(sx + 0.97 * (grip[0] - sx)), round(sy + 0.97 * (grip[1] - sy)))
    assert tuple(frame[end[1], end[0]]) == cast.COCO.skin                         # the hand end of the arm is bare


def test_a_cheering_opponent_raises_the_other_arm_and_a_sad_one_does_not(cam):
    cheer, glum = court(), court()
    characters.draw_opponent(cheer, cam, cast.PIP, x_m=0.0, z_m=3.1, grip_px=(560, 190), body_rows=210, mood="cheer")
    characters.draw_opponent(glum, cam, cast.PIP, x_m=0.0, z_m=3.1, grip_px=(560, 190), body_rows=210, mood="sad")
    assert changed(cheer, glum).sum() > 1500
    assert bounds(changed(cheer, court()))[2] > bounds(changed(glum, court()))[2] + 10      # reaches further right


def test_drawing_into_a_view_of_the_top_of_the_frame_clips_the_body_at_its_last_row(cam):
    frame = court()
    characters.draw_opponent(frame[:210], cam, cast.PIP, x_m=0.0, z_m=3.1, grip_px=(600, 150), body_rows=None)
    assert changed(frame[210:], court()[210:]).sum() == 0


# --- the crowd beside the court ---------------------------------------------------------------------------------------------------
def test_a_spectator_stands_on_the_deck_with_the_head_above_the_floor_and_gets_smaller_further_away(cam):
    near, far = court(), court()
    characters.draw_spectator(near, cam, cast.player_look("kai"), x_m=-2.3, z_m=2.5, mood="happy")
    characters.draw_spectator(far, cam, cast.player_look("kai"), x_m=-2.3, z_m=4.8, mood="happy")
    assert changed(near, court()).sum() > 1.5 * changed(far, court()).sum()
    x0, y0, x1, y1 = bounds(changed(near, court()))
    floor_row = cam.project(-2.3, -0.76, 2.5)[1]
    assert abs(y1 - floor_row) < 6                                         # the body reaches the floor it stands on
    assert y0 < cam.project(-2.3, 0.2, 2.5)[1]                            # and the head is well above the table top


def test_a_cheering_spectator_has_both_arms_up_and_a_quiet_one_has_them_down(cam):
    cheer, calm = court(), court()
    look = cast.player_look("dee")
    characters.draw_spectator(cheer, cam, look, x_m=2.3, z_m=2.6, mood="cheer")
    characters.draw_spectator(calm, cam, look, x_m=2.3, z_m=2.6, mood="happy")
    assert bounds(changed(cheer, court()))[1] < bounds(changed(calm, court()))[1] - 8          # the hands are above the head
    assert changed(cheer, calm).sum() > 800


def test_every_spectator_look_draws_something_and_two_of_them_differ(cam):
    a, b = court(), court()
    characters.draw_spectator(a, cam, cast.player_look("ana"), x_m=-2.4, z_m=3.0)
    characters.draw_spectator(b, cam, cast.player_look("zed"), x_m=-2.4, z_m=3.0)
    assert changed(a, court()).sum() > 3000 and changed(a, b).sum() > 800


# --- a round portrait for a card -------------------------------------------------------------------------------------------------------
def test_a_round_portrait_is_a_circle_with_the_face_inside_and_a_white_rim():
    frame = blank()
    characters.portrait(frame, W // 2, H // 2, 120, cast.PIP, mood="happy", t=0.0, bg=((255, 220, 160), (255, 245, 210)))
    white = (frame == np.array((255, 255, 255), dtype=np.uint8)).all(axis=2)
    x0, y0, x1, y1 = bounds(white)
    assert abs((x1 - x0) - 120) <= 6 and abs((y1 - y0) - 120) <= 6                    # the rim is as wide as the diameter asked for
    assert count(frame, cast.PIP.skin) > 200 and white.sum() > 200                     # the face, and the white rim
    assert tuple(frame[H // 2 - 58, W // 2 - 58]) == BG                                # the corner of the square is not drawn


def test_the_portrait_has_a_different_look_for_a_different_person_and_mood():
    a, b, c = blank(), blank(), blank()
    characters.portrait(a, 200, 200, 120, cast.PIP)
    characters.portrait(b, 200, 200, 120, cast.COCO)
    characters.portrait(c, 200, 200, 120, cast.PIP, mood="sad")
    assert changed(a, b).sum() > 1500 and changed(a, c).sum() > 40


def test_a_portrait_still_growing_from_nothing_is_not_drawn_and_is_not_an_error():
    frame = blank()
    for d in (0, 0.4, 1, 3):
        characters.portrait(frame, 200, 200, d, cast.PIP)
    assert np.array_equal(frame, blank())


# --- the hardest opponent: after the professor's photo -------------------------------------------------------------------------------------
def lighter_than(frame, color, by=30):
    return (frame.astype(int) >= np.array(color, dtype=int) + by).all(axis=2)


def test_the_professor_has_light_blue_eyes_that_show():
    frame = bust(cast.ROGERS, size=300, bob=False)
    assert count(frame, cast.ROGERS.eyes) > 300
    assert count(bust(cast.ROGERS, size=300, bob=False, t=3.05), cast.ROGERS.eyes) < 40                     # closed for a blink: no eyes


def test_the_professor_smiles_with_all_his_teeth_and_the_others_do_not():
    def teeth(look, mood):
        frame = bust(look, size=300, bob=False, mood=mood)
        face = frame[H // 2 + 6:H // 2 + 60, W // 2 - 60:W // 2 + 60]                                     # from under the nose to the chin
        return int((face == 255).all(axis=2).sum())

    assert teeth(cast.ROGERS, "happy") > 250 and teeth(cast.ROGERS, "smug") > 250 and teeth(cast.ROGERS, "grin") > 250
    assert teeth(cast.COCO, "happy") < 20 and teeth(cast.ROGERS, "sad") < 20                                # a loser's mouth is not a smile


def test_the_professor_wears_a_black_polo_with_a_collar_of_its_own():
    import dataclasses

    frame = bust(cast.ROGERS, size=300, bob=False)
    sporty = bust(dataclasses.replace(cast.ROGERS, collar="v"), size=300, bob=False)
    assert count(frame, cast.ROGERS.shirt) > 2500 and count(frame, cast.ROGERS.trim) > 200                      # the flaps of the collar
    outside = lambda f: count(f[:, : W // 2 - 72], cast.ROGERS.trim) + count(f[:, W // 2 + 72:], cast.ROGERS.trim)      # noqa: E731
    assert outside(frame) == 0 and outside(sporty) > 100                                                        # and no stripe along the shoulders


def test_the_professors_gray_hair_has_lighter_streaks_in_it():
    frame = bust(cast.ROGERS, size=300, bob=False)
    crown = frame[: H // 2 - 120]                                                                            # above the top of the head: only hair
    assert lighter_than(crown, cast.ROGERS.hair).sum() > 150


def test_the_professors_older_face_is_drawn_on_the_face_and_nowhere_else():
    import dataclasses

    plain = dataclasses.replace(cast.ROGERS, mature=False)
    with_lines, without = bust(cast.ROGERS, size=300, bob=False), bust(plain, size=300, bob=False)
    diff = changed(with_lines, without)
    assert 150 < diff.sum() < 9000
    x0, y0, x1, y1 = bounds(diff)
    assert y0 > H // 2 - 0.33 * 300 and y1 < H // 2 + 0.2 * 300                                              # not in the hair, not below the chin
    assert x0 > W // 2 - 0.3 * 300 and x1 < W // 2 + 0.3 * 300


@pytest.mark.parametrize("a, b", [("happy", "sad"), ("happy", "cheer"), ("sad", "surprised"), ("neutral", "smug"), ("happy", "grin")])
def test_a_mood_changes_the_professors_face_and_leaves_his_shoulders_and_hair_alone(a, b):
    diff = changed(bust(cast.ROGERS, mood=a, bob=False), bust(cast.ROGERS, mood=b, bob=False))
    assert diff.sum() > 60
    _, y0, _, y1 = bounds(diff)
    assert y1 < H // 2 + 0.24 * 200 and y0 > H // 2 - 0.33 * 200


def test_the_professors_portrait_stays_inside_the_box_it_was_given():
    frame = bust(cast.ROGERS, size=240)
    x0, y0, x1, y1 = bounds(changed(frame, blank()))
    assert x0 >= W // 2 - 0.62 * 240 and x1 <= W // 2 + 0.62 * 240
    assert y0 >= H // 2 - 0.62 * 240 and y1 <= H // 2 + 0.55 * 240
