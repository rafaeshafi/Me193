"""audio: synthesized game sounds, a tiny mixer, and the event -> sound mapping (no speaker needed)."""

import numpy as np
import pytest

from pingpong import app, audio, levels
from pingpong.events import GameEvent
from pingpong.sources_fake import FakeSound


def dominant_hz(clip):
    spectrum = np.abs(np.fft.rfft(clip * np.hanning(len(clip))))
    return float(np.argmax(spectrum[1:]) + 1) * audio.RATE / len(clip)


def test_every_sound_is_a_short_float32_clip_that_never_clips():
    assert set(audio.SOUNDS) >= {"tick", "go", "hit_perfect", "hit_good", "hit_off", "miss", "point", "record"}
    for name, clip in audio.SOUNDS.items():
        assert clip.dtype == np.float32 and 0.02 < len(clip) / audio.RATE < 1.0, name
        assert 0.2 < float(np.abs(clip).max()) <= 0.9 and not np.isnan(clip).any(), name


def test_a_better_hit_pops_higher_and_a_miss_buzzes_low_and_longer():
    good, perfect = dominant_hz(audio.SOUNDS["hit_good"]), dominant_hz(audio.SOUNDS["hit_perfect"])
    assert perfect > good > dominant_hz(audio.SOUNDS["hit_off"]) - 1
    assert dominant_hz(audio.SOUNDS["miss"]) < 400 < good
    assert len(audio.SOUNDS["miss"]) > 2 * len(audio.SOUNDS["hit_good"])


def test_the_mixer_sums_overlapping_sounds_and_drops_the_finished_ones():
    mixer = audio.Mixer()
    mixer.add(np.full(100, 0.25, dtype=np.float32))
    mixer.add(np.full(60, 0.25, dtype=np.float32))
    first = mixer.read(50)
    assert np.allclose(first, 0.5)
    second = mixer.read(50)
    assert np.allclose(second[:10], 0.5) and np.allclose(second[10:], 0.25)          # the short one ended at 60
    assert np.allclose(mixer.read(50), 0.0) and mixer.active == 0


def test_the_mixer_never_exceeds_full_scale_however_many_sounds_pile_up():
    mixer = audio.Mixer()
    for _ in range(10):
        mixer.add(np.full(100, 0.9, dtype=np.float32))
    assert float(np.abs(mixer.read(100)).max()) <= 1.0


def test_events_become_sounds_in_order_and_boring_events_stay_silent():
    sound = FakeSound()
    a = audio.Audio(backend=sound)
    a.start()
    club = levels.LEVELS[2]
    a.play_events([GameEvent("hit", 0, {"label": "perfect"}), GameEvent("verdict", 0, {}), GameEvent("record", 0, {}),
                   GameEvent("miss", 0, {}), GameEvent("point", 0, {"scorer": "player"})], club)
    assert a.played == ["hit_perfect", "record", "miss", "point"]


def test_the_stream_callback_delivers_the_mixed_audio_to_the_output_buffer():
    sound = FakeSound()
    a = audio.Audio(backend=sound)
    a.start()
    a.play("tick")
    out = np.zeros((256, 1), dtype=np.float32)
    sound.stream.callback(out, 256, None, None)
    assert float(np.abs(out).max()) > 0.1
    silent = np.zeros((256, 1), dtype=np.float32)
    for _ in range(40):
        sound.stream.callback(silent, 256, None, None)
    assert float(np.abs(silent).max()) == 0.0                                   # the tick has ended


def test_muting_silences_everything_and_unmuting_brings_it_back():
    a = audio.Audio(backend=FakeSound())
    a.start()
    a.muted = True
    a.play("hit_good")
    assert a.played == []
    a.muted = False
    a.play("hit_good")
    assert a.played == ["hit_good"]


def test_a_machine_with_no_output_device_just_has_no_sound():
    messages = []
    a = audio.Audio(backend=FakeSound(fail=True), log=messages.append)
    a.start()
    a.play("hit_good")                                                          # must not raise
    assert a.enabled is False and a.played == [] and any("audio" in m.lower() for m in messages)


def test_stop_closes_the_stream_once():
    sound = FakeSound()
    a = audio.Audio(backend=sound)
    a.start()
    a.stop()
    a.stop()
    assert sound.stream.closed == 1


def test_the_countdown_ticks_once_per_digit_and_go_sounds_on_the_serve():
    sound = FakeSound()
    session = app.make_session()
    session.audio = audio.Audio(backend=sound)
    session.audio.start()
    session.on_start()
    for _ in range(40):                                                         # 4 s at 10 Hz covers 3-2-1 and the serve
        session.clock.advance_s(0.1)
        session.tick()
    names = session.audio.played
    assert names.count("tick") == 3 and names.count("go") == 1 and names.index("go") > names.index("tick")


def test_a_hit_in_the_game_makes_its_sound():
    session = app.make_session()
    session.audio = audio.Audio(backend=FakeSound())
    session.audio.start()
    app.play_until_hits(session, 2)
    assert [n for n in session.audio.played if n.startswith("hit_")] and "go" in session.audio.played


def test_the_s_key_toggles_the_sound():
    from pingpong import keys

    session = app.make_session()
    session.audio = audio.Audio(backend=FakeSound())
    keys.handle_key(session, ord("s"), fake=False)
    assert session.audio.muted is True
    keys.handle_key(session, ord("s"), fake=False)
    assert session.audio.muted is False


# --- the built-in speakers, not Bluetooth headphones -------------------------------------------------------------------------
DEVICES = [{"name": "MacBook Pro Microphone", "max_output_channels": 0}, {"name": "AirPods Pro", "max_output_channels": 2},
           {"name": "ZoomAudioDevice", "max_output_channels": 2}, {"name": "MacBook Pro Speakers", "max_output_channels": 2}]


def test_the_built_in_speakers_are_preferred_over_whatever_the_system_default_is():
    assert audio.pick_output_device(DEVICES) == 3
    assert audio.pick_output_device(list(reversed(DEVICES))) == 0
    assert audio.pick_output_device([DEVICES[0], DEVICES[1], DEVICES[2]]) is None            # none: use the default
    assert audio.pick_output_device([]) is None
    assert audio.pick_output_device([{"name": "Studio Speakers", "max_output_channels": 0}]) is None   # input only


def test_the_stream_opens_on_the_built_in_speakers_and_says_so():
    class Sound(FakeSound):
        def query_devices(self):
            return DEVICES

    sound, messages = Sound(), []
    a = audio.Audio(backend=sound, log=messages.append)
    a.start()
    assert a.enabled and sound.options["device"] == 3
    assert any("MacBook Pro Speakers" in m for m in messages)


def test_a_backend_that_cannot_list_devices_still_plays_on_the_default():
    sound = FakeSound()
    a = audio.Audio(backend=sound)
    a.start()
    assert a.enabled and sound.options.get("device") is None


# --- the music channel ------------------------------------------------------------------------------------------------------------------
def test_music_fades_in_loops_and_fades_out_through_the_mixer():
    mixer = audio.Mixer()
    tune = np.full(1000, 0.5, dtype=np.float32)
    mixer.set_music(tune, fade_s=0.01)                                    # 441 frames of fade
    first = mixer.read(200)
    assert first[0] < 0.05 < first[-1] < 0.5                              # rising
    mixer.read(400)
    steady = mixer.read(300)
    assert np.allclose(steady, 0.5 * audio.MUSIC_GAIN) and mixer.music_active
    for _ in range(10):
        looped = mixer.read(500)                                          # far past the end of the clip: it goes round
    assert np.allclose(looped, 0.5 * audio.MUSIC_GAIN)
    mixer.stop_music(fade_s=0.01)
    mixer.read(600)
    assert np.allclose(mixer.read(100), 0.0) and not mixer.music_active


def test_a_new_tune_waits_for_the_old_one_to_fade_and_then_comes_in():
    mixer = audio.Mixer()
    mixer.set_music(np.full(5000, 0.5, dtype=np.float32), fade_s=0.005)
    mixer.read(1000)
    mixer.set_music(np.full(5000, -0.5, dtype=np.float32), fade_s=0.005)
    mixer.read(600)
    later = mixer.read(1000)
    assert later[-1] < 0 and mixer.music_active


def test_the_music_is_mixed_under_the_effects_and_the_sum_still_never_clips():
    mixer = audio.Mixer()
    mixer.set_music(np.full(5000, 0.9, dtype=np.float32), fade_s=0.001)
    for _ in range(6):
        mixer.add(np.full(2000, 0.9, dtype=np.float32))
    mixer.read(300)
    assert float(np.abs(mixer.read(300)).max()) <= 1.0


def test_audio_plays_and_stops_music_by_name_and_mutes_it_with_everything_else():
    sound = FakeSound()
    a = audio.Audio(backend=sound)
    a.start()
    a.play_music("menu")
    out = np.zeros((4096, 1), dtype=np.float32)
    for _ in range(6):
        sound.stream.callback(out, 4096, None, None)
    assert a.mixer.music_active and float(np.abs(out).max()) > 0.01 and a.music_playing == "menu"
    a.muted = True
    assert a.music_playing is None                                       # the S key silences the music as well
    a.muted = False
    assert a.music_playing == "menu"                                     # and it comes back to the tune the screens want
    a.play_music(None)
    assert a.music_playing is None
    a.muted = True
    a.play_music("intro")
    assert a.music_playing is None
    a.muted = False
    assert a.music_playing == "intro"


def test_music_can_be_left_off_and_a_name_nobody_knows_is_ignored():
    a = audio.Audio(backend=FakeSound(), music=False)
    a.start()
    a.play_music("menu")
    assert a.music_playing is None
    b = audio.Audio(backend=FakeSound())
    b.start()
    b.play_music("no such tune")
    assert b.music_playing is None


def test_a_machine_with_no_sound_just_has_no_music():
    a = audio.Audio(backend=None)
    a.play_music("menu")
    assert a.music_playing is None and not a.enabled
