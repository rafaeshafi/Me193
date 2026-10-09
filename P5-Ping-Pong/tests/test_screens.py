"""The screens round the game: the title, the choice of game and of opponent, the face-off and the results.  Pure frames."""

import numpy as np
import pytest

from pingpong import cast, fonts, hud, menu_layout, ui
from pingpong.cast import rgb
from pingpong.uistate import OnlineView, Results, UiState

W, H = 1280, 720
GOLD_FILL = rgb(255, 214, 70)


def state(screen, **ui_kw):
    results = ui_kw.pop("results", None)
    return hud.HudState(screen=screen, ui=UiState(screen=screen, **ui_kw), player_name="rafae", level_name=ui_kw_level(ui_kw),
                        mode="match" if ui_kw.get("mode") == "match" else "survival", results=results, hub_status="ok",
                        mqtt_status="ok", anim_t=ui_kw.get("t_s", 0.0), leaderboard=(("maya", 21), ("rafae", 17)))


def ui_kw_level(kw):
    return {1: "Rookie", 2: "Club", 3: "Pro"}[kw.get("level_tag", 1)]


def render(screen, **kw):
    return hud.render(state(screen, **kw), size=(W, H))


def diff(a, b):
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).sum())


def rect_px(rect):
    x0, y0, x1, y1 = rect
    return slice(round(y0 * H), round(y1 * H)), slice(round(x0 * W), round(x1 * W))


def drawn_text(monkeypatch):
    seen = []
    real = fonts.draw

    def spy(frame, text, *a, **kw):
        seen.append(text)
        return real(frame, text, *a, **kw)

    monkeypatch.setattr(fonts, "draw", spy)
    return seen


MENUS = ("TITLE", "MODE", "OPPONENT", "VS", "RESULTS", "ONLINE", "WAIT")


@pytest.mark.parametrize("screen", MENUS)
def test_every_screen_is_a_full_picture_and_the_same_every_time(screen):
    results = Results(won=True, title="YOU WIN!", stats=(("HITS", "31"),)) if screen == "RESULTS" else None
    a, b = render(screen, t_s=1.7, results=results, online=HOSTING), render(screen, t_s=1.7, results=results, online=HOSTING)
    assert a.shape == (H, W, 3) and a.dtype == np.uint8 and np.array_equal(a, b)
    assert a.std() > 20                                                       # not a flat colour


@pytest.mark.parametrize("screen", MENUS)
def test_a_screen_comes_in_over_its_first_second_and_settles(screen):
    results = Results(won=True, title="YOU WIN!", stats=(("HITS", "31"),)) if screen == "RESULTS" else None
    first, later, settled = (render(screen, t_s=t, results=results, online=HOSTING) for t in (0.0, 0.5, 3.0))
    assert diff(first, later) > 20_000 and diff(later, settled) > 2_000


@pytest.mark.parametrize("screen", ("TITLE", "MODE", "OPPONENT", "RESULTS", "ONLINE", "WAIT"))
@pytest.mark.parametrize("a, b", [(0.4, 0.4), (0.99, 0.01)])
def test_the_pointer_is_drawn_wherever_the_hand_is_over_the_screen(screen, a, b):
    results = Results(title="GAME OVER") if screen == "RESULTS" else None
    plain = render(screen, t_s=3.0, results=results, online=HOSTING)
    pointed = render(screen, t_s=3.0, cursor=(a, b), results=results, online=HOSTING)
    cx, cy = round(min(1, a) * (W - 1)), round((1 - b) * (H - 1))
    spot = (slice(max(0, cy - 70), cy + 71), slice(max(0, cx - 70), cx + 71))
    assert diff(plain[spot], pointed[spot]) > 1_500


# --- the title ----------------------------------------------------------------------------------------------------------------------
def test_the_title_shows_the_name_of_the_game_the_start_button_and_who_is_playing(monkeypatch):
    seen = drawn_text(monkeypatch)
    render("TITLE", t_s=3.0)
    assert "PING-PONG" in seen and "START" in seen and "RAFAE" in " ".join(seen).upper()


def test_the_start_button_is_where_the_hand_has_to_hold_and_it_fills_and_pops_while_held():
    idle = render("TITLE", t_s=3.0)
    held = render("TITLE", t_s=3.0, hover="start", progress=0.5, cursor=(0.95, 0.9))
    x, y, bw, bh = (round(f * v) for f, v in zip(menu_layout.holdstart.BUTTON, (W, H, W, H)))
    area = (slice(y, y + bh), slice(x, x + bw))
    assert diff(idle[area], held[area]) > 8_000
    gold = lambda f: int((f[area].reshape(-1, 3) == np.array(GOLD_FILL, dtype=np.uint8)).all(axis=1).sum())          # noqa: E731
    assert gold(idle) == 0 < gold(held)
    more = render("TITLE", t_s=3.0, hover="start", progress=0.9, cursor=(0.95, 0.9))
    assert gold(more) > gold(held)


def test_the_title_says_what_is_wrong_with_the_hub_and_the_broker_in_a_notice_and_a_chip():
    ok = render("TITLE", t_s=3.0)
    bad = hud.render(hud.HudState(screen="TITLE", ui=UiState(screen="TITLE", t_s=3.0), hub_status="stale", mqtt_status="offline",
                                  message="UNCALIBRATED: run ./pp calibrate_swing"), size=(W, H))
    assert diff(ok, bad) > 10_000


# --- the choice of game -----------------------------------------------------------------------------------------------------------------
def test_the_choice_of_game_has_three_cards_and_a_back_button(monkeypatch):
    seen = drawn_text(monkeypatch)
    render("MODE", t_s=3.0)
    assert {"RALLY", "MATCH", "ONLINE", "BACK"} <= set(seen) and any("CHOOSE" in text for text in seen)


@pytest.mark.parametrize("name", ["rally", "match", "online", "back"])
def test_pointing_at_a_button_makes_it_pop_and_the_hold_fills_it(name):
    idle = render("MODE", t_s=3.0, focus=1 if name == "rally" else 0)        # (with nothing pointed at, the keys' choice stands out)
    over = render("MODE", t_s=3.0, hover=name, progress=0.0)
    held = render("MODE", t_s=3.0, hover=name, progress=0.6)
    rect = dict(menu_layout.MODE_CARDS, back=menu_layout.BACK_BUTTON)[name]
    near = (slice(max(0, round(rect[1] * H) - 30), round(rect[3] * H) + 30), slice(max(0, round(rect[0] * W) - 30), round(rect[2] * W) + 30))
    assert diff(idle[near], over[near]) > 5_000
    assert diff(over[near], held[near]) > 3_000


def test_the_card_in_focus_for_the_keys_looks_like_the_one_the_hand_is_on():
    focused = render("MODE", t_s=3.0, focus=1)
    pointed = render("MODE", t_s=3.0, hover="match")
    rect = rect_px(menu_layout.MODE_CARDS["match"])
    assert diff(focused[rect], pointed[rect]) < 0.15 * diff(render("MODE", t_s=3.0)[rect], pointed[rect]) + 3_000


# --- the choice of opponent -----------------------------------------------------------------------------------------------------------------
def test_each_opponent_has_a_card_with_a_name_a_level_and_stars(monkeypatch):
    seen = drawn_text(monkeypatch)
    render("OPPONENT", t_s=3.0)
    assert {"PIP", "COCO", "MAX", "ROOKIE", "CLUB", "PRO", "BACK"} <= set(seen)
    assert {"SPEED", "DEFENSE", "SPIN"} <= set(seen)
    assert {"Let's rally!", "Bring your best!"} <= set(seen)


def test_the_opponent_cards_are_where_the_hand_has_to_hold():
    for tag in (1, 2, 3):
        idle = render("OPPONENT", t_s=3.0)
        over = render("OPPONENT", t_s=3.0, hover=f"opp{tag}", progress=0.5)
        area = rect_px(menu_layout.OPPONENT_CARDS[tag])
        assert diff(idle[area], over[area]) > 10_000


def test_the_opponent_you_have_chosen_so_far_is_in_focus_and_the_game_chosen_is_named():
    a = render("OPPONENT", t_s=3.0, level_tag=1, focus=0, mode="match")
    b = render("OPPONENT", t_s=3.0, level_tag=3, focus=2, mode="match")
    assert diff(a[rect_px(menu_layout.OPPONENT_CARDS[3])], b[rect_px(menu_layout.OPPONENT_CARDS[3])]) > 5_000
    seen_a = render("OPPONENT", t_s=3.0, mode="match")
    seen_b = render("OPPONENT", t_s=3.0, mode="survival")
    assert diff(seen_a[:140], seen_b[:140]) > 500                              # a tag at the top says which game


# --- the face-off ------------------------------------------------------------------------------------------------------------------------
def test_the_face_off_shows_both_names_and_the_game(monkeypatch):
    seen = drawn_text(monkeypatch)
    render("VS", t_s=1.5, level_tag=2, mode="match")
    assert {"RAFAE", "COCO", "VS"} <= set(seen) and any("MATCH" in text for text in seen)
    seen.clear()
    render("VS", t_s=1.5, level_tag=3, mode="survival")
    assert "MAX" in seen and any("RALLY" in text for text in seen)


def test_the_two_come_in_from_the_sides_and_the_face_off_flashes_white_at_the_end():
    early, mid, late = render("VS", t_s=0.1), render("VS", t_s=1.2), render("VS", t_s=2.35)
    assert diff(early, mid) > 100_000
    assert late.mean() > mid.mean() + 8                                       # the white flash into the game


# --- the results -----------------------------------------------------------------------------------------------------------------------------
# --- playing a friend ---------------------------------------------------------------------------------------------------------------------------------
ROOMS = ({"code": "ABCDE", "host": "MAYA", "pace": 2, "target": 7, "t": 3}, {"code": "FGHJK", "host": "LEO", "pace": 1, "target": 11, "t": 2},
         {"code": "MNPQR", "host": "ZOE", "pace": 3, "target": 5, "t": 1})
BROWSING = OnlineView(status="browsing", rooms=ROOMS)
HOSTING = OnlineView(status="hosting", code="K7QMD", pace=2, target=7, live=True)


def friend_state(screen, view=None, **kw):
    ui_kw = {k: kw.pop(k) for k in list(kw) if k in ("t_s", "hover", "progress", "focus", "cursor", "level_tag", "rematch_pending", "opponent_gone")}
    return hud.HudState(screen=screen, ui=UiState(screen=screen, online=view, opponent=kw.pop("opponent", ""), **ui_kw), player_name="rafae",
                        level_name="Club", mode="match", hub_status="ok", mqtt_status="ok", anim_t=ui_kw.get("t_s", 0.0), **kw)


def render_friend(screen, view=None, **kw):
    return hud.render(friend_state(screen, view, **kw), size=(W, H))


def test_the_list_offers_a_game_to_host_at_each_pace_and_the_games_others_have_opened(monkeypatch):
    seen = drawn_text(monkeypatch)
    render_friend("ONLINE", BROWSING, t_s=3.0)
    assert {"PLAY A FRIEND", "HOST A GAME", "ROOKIE", "CLUB", "PRO", "OPEN GAMES", "BACK", "MAYA", "LEO", "ZOE"} <= set(seen)
    assert any("FIRST TO 11" in text for text in seen) and any("FIRST TO 7" in text for text in seen)


def test_only_the_games_on_the_list_get_a_row_and_an_empty_list_says_so(monkeypatch):
    seen = drawn_text(monkeypatch)
    render_friend("ONLINE", OnlineView(status="browsing", rooms=ROOMS[:1]), t_s=3.0)
    assert "MAYA" in seen and "LEO" not in seen
    seen.clear()
    render_friend("ONLINE", OnlineView(status="browsing"), t_s=3.0)
    assert any("NO OPEN GAMES" in text for text in seen)
    seen.clear()
    render_friend("ONLINE", OnlineView(status="connecting"), t_s=3.0)
    assert any("CONNECTING" in text for text in seen)
    seen.clear()
    render_friend("ONLINE", OnlineView(status="offline", message="CAN'T REACH THE GAME SERVER - IS THE INTERNET ON?"), t_s=3.0)
    assert any("GAME SERVER" in text for text in seen)


def test_what_went_wrong_with_the_last_try_is_said_on_the_list(monkeypatch):
    seen = drawn_text(monkeypatch)
    plain = render_friend("ONLINE", BROWSING, t_s=3.0)
    failed = render_friend("ONLINE", OnlineView(status="browsing", rooms=ROOMS, message="THAT GAME IS FULL"), t_s=3.0)
    assert "THAT GAME IS FULL" in seen and diff(plain, failed) > 5_000


@pytest.mark.parametrize("name", ["host1", "host2", "host3"])
def test_a_pace_chip_pops_and_fills_where_the_hand_holds(name):
    chip = menu_layout.HOST_CHIPS[int(name[4:])]
    idle = render_friend("ONLINE", BROWSING, t_s=3.0, focus=1)
    over = render_friend("ONLINE", BROWSING, t_s=3.0, hover=name)
    held = render_friend("ONLINE", BROWSING, t_s=3.0, hover=name, progress=0.6)
    area = rect_px(chip)
    assert diff(idle[area], over[area]) > 2_000 and diff(over[area], held[area]) > 1_500


@pytest.mark.parametrize("k", [0, 1, 2])
def test_an_open_game_pops_and_fills_where_the_hand_holds(k):
    idle = render_friend("ONLINE", BROWSING, t_s=3.0, focus=0)
    over = render_friend("ONLINE", BROWSING, t_s=3.0, hover=f"join{k}")
    held = render_friend("ONLINE", BROWSING, t_s=3.0, hover=f"join{k}", progress=0.6)
    area = rect_px(menu_layout.ROOM_ROWS[k])
    assert diff(idle[area], over[area]) > 5_000 and diff(over[area], held[area]) > 2_000


def test_the_game_the_keys_have_in_focus_stands_out_like_the_one_the_hand_is_on():
    focused = render_friend("ONLINE", BROWSING, t_s=3.0, focus=2)               # the host card, then the games: the second game
    plain = render_friend("ONLINE", BROWSING, t_s=3.0, focus=0)
    area = rect_px(menu_layout.ROOM_ROWS[1])
    assert diff(focused[area], plain[area]) > 5_000


def test_hosting_shows_the_code_one_letter_to_a_tile_and_how_to_back_out(monkeypatch):
    seen = drawn_text(monkeypatch)
    render_friend("WAIT", HOSTING, t_s=3.0)
    assert {"K", "7", "Q", "M", "D", "CANCEL"} <= set(seen) and any("WAITING FOR A FRIEND" in text for text in seen)
    assert any("CLUB" in text and "FIRST TO 7" in text for text in seen)


def test_a_game_not_yet_on_the_list_says_it_is_being_set_up_and_joining_says_so(monkeypatch):
    seen = drawn_text(monkeypatch)
    render_friend("WAIT", OnlineView(status="hosting", code="K7QMD", pace=2, target=7, live=False), t_s=3.0)
    assert any("setting up" in text.lower() for text in seen)
    seen.clear()
    render_friend("WAIT", OnlineView(status="joining", code="ABCDE"), t_s=3.0)
    assert any("JOINING" in text for text in seen) and "CANCEL" in seen and "A" in seen


def test_the_cancel_button_is_where_the_hand_has_to_hold_and_fills():
    idle = render_friend("WAIT", HOSTING, t_s=3.0)
    held = render_friend("WAIT", HOSTING, t_s=3.0, hover="cancel", progress=0.7)
    assert diff(idle[rect_px(menu_layout.BACK_BUTTON)], held[rect_px(menu_layout.BACK_BUTTON)]) > 3_000


def test_the_code_tiles_come_in_one_after_another():
    early, later = render_friend("WAIT", HOSTING, t_s=0.45), render_friend("WAIT", HOSTING, t_s=0.9)
    assert diff(early, later) > 20_000


def test_the_face_off_with_a_friend_shows_their_name_and_that_it_is_online(monkeypatch):
    seen = drawn_text(monkeypatch)
    render_friend("VS", t_s=1.5, opponent="MAYA", level_tag=3)
    assert {"RAFAE", "MAYA", "VS"} <= set(seen) and any("ONLINE" in text and "FIRST TO" in text for text in seen) and "COCO" not in seen


def test_the_face_off_with_a_friend_does_not_look_like_the_one_with_the_computer():
    friend = render_friend("VS", t_s=1.5, opponent="MAYA", level_tag=2)
    computer = render("VS", t_s=1.5, level_tag=2, mode="match")
    assert diff(friend, computer) > 30_000


FRIEND_WIN = Results(won=True, title="YOU WIN!", stats=(("HITS", "31"), ("LONGEST RALLY", "12"), ("TOP SPEED", "74 km/h"), ("TIME", "2:41")))


def test_the_results_against_a_friend_offer_a_rematch_and_leaving_and_name_them(monkeypatch):
    seen = drawn_text(monkeypatch)
    render_friend("RESULTS", t_s=3.0, opponent="MAYA", results=FRIEND_WIN, opponent_name="MAYA", player_points=7, cpu_points=4)
    assert {"YOU WIN!", "REMATCH", "LEAVE", "7  -  4"} <= set(seen) and any("RAFAE" in t and "MAYA" in t for t in seen)
    assert "PLAY AGAIN" not in seen and "OPPONENT" not in seen and "COCO" not in seen


def test_asking_for_a_rematch_says_it_is_waiting_for_the_friend_and_a_friend_who_left_leaves_only_leaving(monkeypatch):
    seen = drawn_text(monkeypatch)
    render_friend("RESULTS", t_s=3.0, opponent="MAYA", results=FRIEND_WIN, opponent_name="MAYA", rematch_pending=True)
    assert "WAITING..." in seen and "REMATCH" not in seen
    seen.clear()
    gone = Results(won=None, title="THEY LEFT", stats=FRIEND_WIN.stats)
    render_friend("RESULTS", t_s=3.0, opponent="MAYA", results=gone, opponent_name="MAYA", opponent_gone=True)
    assert "REMATCH" not in seen and "LEAVE" in seen and any("LEFT" in t for t in seen)


def test_the_friends_avatar_is_on_the_results():
    friend = render_friend("RESULTS", t_s=3.0, opponent="MAYA", results=FRIEND_WIN, opponent_name="MAYA")
    computer = render("RESULTS", t_s=3.0, results=FRIEND_WIN, mode="match")
    right = (slice(150, 480), slice(int(0.62 * W), int(0.9 * W)))
    assert diff(friend[right], computer[right]) > 10_000


WIN = Results(won=True, title="YOU WIN!", stats=(("HITS", "31"), ("LONGEST RALLY", "12"), ("TOP SPEED", "74 km/h"), ("TIME", "2:41")))
LOSE = Results(won=False, title="THE CPU WINS", stats=WIN.stats)
RECORD = Results(won=None, title="NEW RECORD!", new_record=True, stats=WIN.stats)


def test_the_results_say_the_result_and_the_numbers_and_offer_two_buttons(monkeypatch):
    seen = drawn_text(monkeypatch)
    render("RESULTS", t_s=3.0, results=WIN)
    assert {"YOU WIN!", "HITS", "31", "LONGEST RALLY", "TOP SPEED", "74 km/h", "TIME", "2:41", "PLAY AGAIN", "CHANGE", "OPPONENT"} <= set(seen)


def test_a_win_a_loss_and_a_record_do_not_look_the_same():
    win, lose, record = (render("RESULTS", t_s=3.0, results=r, mode="match") for r in (WIN, LOSE, RECORD))
    assert diff(win, lose) > 30_000 and diff(win, record) > 15_000


def test_paper_falls_for_a_win_and_a_record_and_not_for_a_loss(monkeypatch):
    thrown = []
    monkeypatch.setattr(ui, "confetti", lambda frame, t_s, **kw: thrown.append(t_s))
    for results, falls in ((WIN, True), (RECORD, True), (LOSE, False)):
        thrown.clear()
        render("RESULTS", t_s=3.0, results=results)
        assert bool(thrown) is falls, results.title


@pytest.mark.parametrize("name", ["again", "change"])
def test_the_results_buttons_pop_and_fill_where_the_hand_holds(name):
    idle = render("RESULTS", t_s=3.0, results=WIN)
    held = render("RESULTS", t_s=3.0, results=WIN, hover=name, progress=0.7)
    area = rect_px(dict(again=(0.808, 0.10, 1.0, 0.34), change=(0.0, 0.10, 0.192, 0.34))[name])
    assert diff(idle[area], held[area]) > 6_000


def test_the_leaderboard_is_on_the_results_and_the_players_own_row_is_marked():
    plain = hud.render(hud.HudState(screen="RESULTS", ui=UiState(screen="RESULTS", t_s=3.0), results=WIN, player_name="rafae"), size=(W, H))
    ranked = render("RESULTS", t_s=3.0, results=WIN)
    assert diff(plain, ranked) > 8_000


def test_a_screen_nobody_has_heard_of_is_a_clear_error():
    with pytest.raises(ValueError):
        hud.render(hud.HudState(screen="NOPE", ui=UiState(screen="NOPE")), size=(W, H))


def test_the_note_that_a_friend_left_stays_inside_the_screen_whatever_the_length_of_their_name():
    gone = Results(won=None, title="THEY LEFT", stats=FRIEND_WIN.stats)
    for name in ("AL", "A" * 16):
        frame = render_friend("RESULTS", t_s=3.0, opponent=name, results=gone, opponent_name=name, opponent_gone=True)
        edge = frame[100:142, W - 14:]                                               # the last columns of the pill's row: the court, not coral
        coral = np.array(cast.rgb(235, 76, 84), dtype=np.int32)
        assert (np.abs(edge.astype(np.int32) - coral).sum(axis=2) < 60).sum() == 0, name


# --- the face-off's background is the same picture every frame ------------------------------------------------------------------------------------
def reference_split(frame):
    """The face-off's two-colour slash as it was drawn every frame before it was kept."""
    import cv2

    from pingpong import ui
    from pingpong.cast import rgb

    h, w = frame.shape[:2]
    frame[:] = ui.gradient(h, w, rgb(160, 220, 255), rgb(70, 150, 240))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([(0.58 * w, 0), (w, 0), (w, h), (0.42 * w, h)], dtype=np.int32)], 255, cv2.LINE_AA)
    right = ui.gradient(h, w, rgb(255, 196, 170), rgb(240, 96, 110))
    frame[mask > 0] = right[mask > 0]
    cv2.line(frame, (round(0.58 * w) + 9, 0), (round(0.42 * w) + 9, h), (0, 0, 0), 22, cv2.LINE_AA)
    cv2.line(frame, (round(0.58 * w), 0), (round(0.42 * w), h), (255, 255, 255), 18, cv2.LINE_AA)


def test_the_face_offs_background_is_made_once_and_is_the_same_picture_as_drawing_it_each_time():
    import time

    from pingpong import screens_end

    screens_end._split_picture.cache_clear()                            # (another test may have made it already)
    for size in ((1280, 720), (960, 540)):
        w, h = size
        mine, theirs = np.empty((h, w, 3), np.uint8), np.empty((h, w, 3), np.uint8)
        t0 = time.perf_counter()
        screens_end._split(mine)
        cold = time.perf_counter() - t0
        reference_split(theirs)
        assert (mine == theirs).all()
        warm = []
        for _ in range(5):
            t0 = time.perf_counter()
            screens_end._split(mine)
            warm.append(time.perf_counter() - t0)
        assert min(warm) * 4 < cold                                      # kept: a copy of a picture, not a drawing
