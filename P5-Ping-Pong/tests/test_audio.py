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
