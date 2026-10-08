"""The music of the island, made of nothing but numpy: a marimba-and-bass loop for the menus, and a short piece for the intro whose big
chord lands on the moment the camera reaches the court.  (Original tunes, in C major, on a pentatonic scale: they cannot clash.)"""

import numpy as np
import pytest

from pingpong import audio, intro, music


def rms(clip, a, b):
    chunk = clip[round(a * music.RATE):round(b * music.RATE)]
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def test_the_tempo_makes_four_bars_exactly_as_long_as_the_flight_to_the_court():
    assert 4 * music.BAR_S == pytest.approx(intro.ARRIVE_S, abs=1e-6)
    assert music.BEAT_S * 4 == pytest.approx(music.BAR_S)


@pytest.mark.parametrize("name", ["intro", "menu"])
def test_a_track_is_float32_mono_quiet_enough_to_sit_under_the_game_and_never_clips(name):
    clip = music.TRACKS[name]()
    assert clip.dtype == np.float32 and clip.ndim == 1 and not np.isnan(clip).any()
    assert 0.25 < float(np.abs(clip).max()) <= 0.95


def test_the_intro_lasts_as_long_as_the_intro_and_the_menu_loop_is_eight_bars():
    assert len(music.TRACKS["intro"]()) / music.RATE == pytest.approx(intro.LENGTH_S, abs=0.02)
    assert len(music.TRACKS["menu"]()) / music.RATE == pytest.approx(8 * music.BAR_S, abs=0.02)


def test_the_big_chord_comes_down_on_the_moment_the_camera_lands():
    clip = music.TRACKS["intro"]()
    before, after = rms(clip, 8.2, 8.9), rms(clip, intro.ARRIVE_S + 0.02, intro.ARRIVE_S + 0.7)
    assert after > 1.4 * before


def test_the_intro_starts_softly_and_builds():
    clip = music.TRACKS["intro"]()
    assert rms(clip, 0.0, 0.5) < rms(clip, 6.0, 8.0)


def test_a_menu_loop_joins_up_with_itself_without_a_click():
    clip = music.TRACKS["menu"]()
    jump = abs(float(clip[-1]) - float(clip[0]))
    typical = float(np.abs(np.diff(clip[:20000])).mean())
    assert jump < 12 * typical + 0.02


def test_the_notes_are_all_in_the_scale_of_the_tune():
    assert {note % 12 for note in music.SCALE} == {0, 2, 4, 7, 9}                # C D E G A: the major pentatonic
    for track in (music.INTRO_NOTES, music.MENU_NOTES):
        assert all(note[1] % 12 in {0, 2, 4, 7, 9, 5, 11} for note in track), "a note outside C major"


def test_the_same_music_every_time():
    assert np.array_equal(music.TRACKS["menu"](), music.render("menu"))
    assert np.array_equal(music.render("intro"), music.render("intro"))


def test_the_endings_for_a_win_a_loss_and_a_face_off_are_short_stingers():
    for name in ("fanfare_win", "fanfare_lose", "vs", "menu_select", "menu_tick", "menu_back"):
        clip = music.STINGERS[name]
        assert clip.dtype == np.float32 and 0.03 < len(clip) / music.RATE < 3.0, name
        assert 0.2 < float(np.abs(clip).max()) <= 0.9, name
    assert len(music.STINGERS["fanfare_win"]) > len(music.STINGERS["menu_tick"])


def test_a_win_ends_higher_than_a_loss():
    def hz(clip):
        tail = clip[-int(0.25 * music.RATE):]
        spectrum = np.abs(np.fft.rfft(tail * np.hanning(len(tail))))
        return float(np.argmax(spectrum[1:]) + 1) * music.RATE / len(tail)

    assert hz(music.STINGERS["fanfare_win"]) > hz(music.STINGERS["fanfare_lose"]) + 100


def test_the_stingers_are_in_the_audio_library_and_the_game_ends_make_them():
    from pingpong.events import GameEvent
    from pingpong import levels

    a = audio.Audio(backend=None)
    club = levels.LEVELS[2]
    assert a.sound_for(GameEvent("match_over", 0, {"winner": "player"}), club) == "fanfare_win"
    assert a.sound_for(GameEvent("match_over", 0, {"winner": "cpu"}), club) == "fanfare_lose"
    assert a.sound_for(GameEvent("game_over", 0, {}), club) == "fanfare_lose"
    assert {"menu_tick", "menu_select", "menu_back", "vs", "fanfare_win", "fanfare_lose"} <= set(a.library)
