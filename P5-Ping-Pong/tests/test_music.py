"""The music of the island, made of nothing but numpy: a driving dance loop for the menus (a kick on every beat, claps on two and four,
hats, a pumping bass, stabs and a bright arpeggio over a marimba hook), and a short piece for the intro that builds up and drops its
biggest hit on the moment the camera reaches the court.  (Original tunes, in C major, on a pentatonic scale: they cannot clash.)"""

import numpy as np
import pytest

from pingpong import audio, intro, music


def rms(clip, a, b):
    chunk = clip[round(a * music.RATE):round(b * music.RATE)]
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def band(clip, lo, hi, a, b):
    """The energy of the clip between a and b seconds in the band lo..hi Hz (a 5 ms taper at both ends, so a hit at the start counts)."""
    chunk = clip[max(0, round(a * music.RATE)):round(b * music.RATE)].astype(np.float64)
    taper = np.ones(len(chunk))
    k = min(len(chunk) // 2, round(0.005 * music.RATE))
    taper[:k], taper[len(chunk) - k:] = np.linspace(0.0, 1.0, k), np.linspace(1.0, 0.0, k)
    spectrum = np.abs(np.fft.rfft(chunk * taper)) ** 2
    freqs = np.fft.rfftfreq(len(chunk), 1 / music.RATE)
    return float(spectrum[(freqs >= lo) & (freqs < hi)].sum())


def at_beats(clip, lo, hi, beats, *, offset_s=0.0, length_s=0.08, lead_s=0.005):
    """The band's energy in the stretch from just before each of the beats (counted from offset_s), as a list."""
    return [band(clip, lo, hi, offset_s + k * music.BEAT_S - lead_s, offset_s + k * music.BEAT_S - lead_s + length_s) for k in beats]


def test_the_tempo_is_a_dance_tempo_and_the_big_hit_falls_on_a_downbeat_of_the_groove():
    assert music.BPM >= 124 and music.BEAT_S == pytest.approx(60.0 / music.BPM)
    assert music.BEAT_S * 4 == pytest.approx(music.BAR_S)
    assert music.LAND_S == pytest.approx(intro.ARRIVE_S, abs=1e-6)                  # the music lands with the camera
    assert music.GROOVE_S + 4 * music.BAR_S == pytest.approx(music.LAND_S, abs=1e-6)    # after a build-up, four bars of groove, then the hit
    assert 0.5 < music.GROOVE_S < 3.0


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


def test_the_intro_builds_to_a_breath_and_then_a_drop_with_a_kick_a_crash_and_a_boom():
    clip, land = music.TRACKS["intro"](), intro.ARRIVE_S
    assert rms(clip, land - 0.11, land - 0.01) < 0.5 * rms(clip, land - 0.8, land - 0.2)        # a gap just before the hit
    assert band(clip, 25, 120, land, land + 0.15) > 3 * band(clip, 25, 120, land - 0.12, land - 0.01)     # a kick and a sub boom
    assert band(clip, 4000, 16000, land, land + 0.6) > 5 * band(clip, 4000, 16000, land - 0.12, land - 0.01)    # a crash


def test_the_intro_brings_its_drums_in_one_after_another():
    clip, bar = music.TRACKS["intro"](), music.BAR_S
    first, third = music.GROOVE_S, music.GROOVE_S + 2 * bar
    assert band(clip, 25, 120, first, first + 0.15) > 5 * band(clip, 25, 120, first - 0.5, first - 0.35)     # the kick starts with bar one
    hats_one = band(clip, 7000, 16000, first, first + bar)
    hats_three = band(clip, 7000, 16000, third, third + bar)
    assert hats_three > 3 * hats_one                                                                          # the hats arrive after it


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


# --- hype: what makes the menu music drive ---------------------------------------------------------------------------------------------------------
MENU_BEATS = range(32)


def test_the_menu_has_a_kick_on_every_beat():
    clip = music.TRACKS["menu"]()
    # the sub (30-75 Hz, where only the kick is: the bass is above it) just after each beat against just before it
    onsets = [band(clip, 30, 75, k * music.BEAT_S + 0.03, k * music.BEAT_S + 0.10) / band(clip, 30, 75, k * music.BEAT_S - 0.06, k * music.BEAT_S - 0.01)
              for k in range(1, 32)]
    assert min(onsets) > 8.0, [round(o, 1) for o in onsets]                                               # not one beat without its kick


def test_the_menu_claps_on_two_and_four():
    clip = music.TRACKS["menu"]()
    backbeat = at_beats(clip, 3000, 6000, range(1, 32, 2), length_s=0.08)                  # (above the stabs and the plucks, where a clap's noise is)
    downbeat = at_beats(clip, 3000, 6000, range(0, 32, 2), length_s=0.08)
    assert float(np.median(backbeat)) > 3 * float(np.median(downbeat))


def test_the_menu_has_hats_between_the_beats():
    clip = music.TRACKS["menu"]()
    between = at_beats(clip, 7000, 16000, MENU_BEATS, length_s=0.05, offset_s=music.BEAT_S / 2)
    assert min(between) > 0.25 * float(np.median(between))                                                # every half beat, all the way round: no gaps


def test_the_menu_is_loud_and_punchy_not_a_quiet_background():
    clip = music.TRACKS["menu"]()
    level = rms(clip, 0.0, len(clip) / music.RATE)
    assert level >= 0.19                                                                                  # the relaxed loop it replaced was 0.146
    assert float(np.abs(clip).max()) / level <= 4.6                                                       # compressed, not a few loud spikes


def test_the_menu_keeps_going_all_the_way_round_with_no_slack_bar():
    clip = music.TRACKS["menu"]()
    bars = [rms(clip, k * music.BAR_S, (k + 1) * music.BAR_S) for k in range(8)]
    assert min(bars) > 0.7 * max(bars)

