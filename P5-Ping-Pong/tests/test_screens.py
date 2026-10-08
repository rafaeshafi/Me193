"""The screens round the game: the title, the choice of game and of opponent, the face-off and the results.  Pure frames."""

import numpy as np
import pytest

from pingpong import cast, fonts, hud, menu_layout, ui
from pingpong.cast import rgb
from pingpong.uistate import Results, UiState

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


MENUS = ("TITLE", "MODE", "OPPONENT", "VS", "RESULTS")


@pytest.mark.parametrize("screen", MENUS)
def test_every_screen_is_a_full_picture_and_the_same_every_time(screen):
    results = Results(won=True, title="YOU WIN!", stats=(("HITS", "31"),)) if screen == "RESULTS" else None
    a, b = render(screen, t_s=1.7, results=results), render(screen, t_s=1.7, results=results)
    assert a.shape == (H, W, 3) and a.dtype == np.uint8 and np.array_equal(a, b)
    assert a.std() > 20                                                       # not a flat colour


@pytest.mark.parametrize("screen", MENUS)
def test_a_screen_comes_in_over_its_first_second_and_settles(screen):
    results = Results(won=True, title="YOU WIN!", stats=(("HITS", "31"),)) if screen == "RESULTS" else None
    first, later, settled = (render(screen, t_s=t, results=results) for t in (0.0, 0.5, 3.0))
    assert diff(first, later) > 20_000 and diff(later, settled) > 2_000


@pytest.mark.parametrize("screen", ("TITLE", "MODE", "OPPONENT", "RESULTS"))
@pytest.mark.parametrize("a, b", [(0.4, 0.4), (0.99, 0.01)])
def test_the_pointer_is_drawn_wherever_the_hand_is_over_the_screen(screen, a, b):
    results = Results(title="GAME OVER") if screen == "RESULTS" else None
    plain = render(screen, t_s=3.0, results=results)
    pointed = render(screen, t_s=3.0, cursor=(a, b), results=results)
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
def test_the_choice_of_game_has_two_cards_and_a_back_button(monkeypatch):
    seen = drawn_text(monkeypatch)
    render("MODE", t_s=3.0)
    assert {"RALLY", "MATCH", "BACK"} <= set(seen) and any("CHOOSE" in text for text in seen)


@pytest.mark.parametrize("name", ["rally", "match", "back"])
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
