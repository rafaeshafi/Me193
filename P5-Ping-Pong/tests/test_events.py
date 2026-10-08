"""events.py is the contract every parallel module is built against."""

import dataclasses

import pytest

from pingpong import events

# Golden field lists: changing one is a deliberate, reviewed contract change.
EXPECTED_FIELDS = {
    "ImuSample": ["t_ns", "g", "a", "src"],
    "GestureEvent": ["t_ns", "gesture"],
    "SwingEvent": ["kind", "t_ns", "w_pk", "dur_ms", "n_reversals", "axis_unit",
                   "net_rot_unit", "a_lin_unit", "clipped", "feat", "src", "g_dps", "net_rot_deg"],
    "PaddlePose": ["t_scene_ns", "u", "v", "conf", "hand", "raw"],
    "TagEvent": ["role", "value", "t_ns"],
    "GateResult": ["name", "passed", "note"],
    "Verdict": ["kind", "q_pos", "e_s", "d_min_sw", "gates", "contact_ns"],
    "ShotParams": ["v_out", "T", "S", "A", "aim_a", "q_total", "fault", "label"],
    "HapticCmd": ["name", "strength", "fire_at_ns"],
    "GameEvent": ["kind", "t_ns", "data"],
}


@pytest.mark.parametrize("name", sorted(EXPECTED_FIELDS))
def test_event_has_exactly_the_contracted_fields(name):
    cls = getattr(events, name)
    assert [f.name for f in dataclasses.fields(cls)] == EXPECTED_FIELDS[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_FIELDS))
def test_events_are_frozen_so_threads_can_share_them(name):
    cls = getattr(events, name)
    assert cls.__dataclass_params__.frozen is True


def test_imu_sample_defaults_to_hub_source_and_raw_counts():
    s = events.ImuSample(t_ns=1, g=(1, 2, 3), a=(4, 5, 6))
    assert s.src == "hub"
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.t_ns = 2


def test_game_event_data_defaults_to_an_empty_mapping():
    e = events.GameEvent(kind="start", t_ns=0)
    assert e.data == {}
