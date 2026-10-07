"""HubLink: the only module that talks to the Double Motor's BLE notifications.

Uses the library's own notification classes (via FakeDoubleMotor.emit) so the
real parser is exercised, with a fake device and a fake clock.
"""

import legoeducation as le
import pytest

from pingpong import hub as hubmod
from pingpong.clock import FakeClock
from pingpong.sources_fake import FakeDoubleMotor

CARD = {"card_color": le.LEGO_COLOR_GREEN, "card_serial": "0997"}


def make(fail_connect=False):
    clock = FakeClock(start_ns=1_000_000_000)
    dev = FakeDoubleMotor(fail_connect=fail_connect)
    link = hubmod.HubLink(dev, notify_ms=15, clock=clock, card=CARD)
    return link, dev, clock


def test_connect_requests_15ms_notifications_for_the_configured_card():
    link, dev, _ = make()
    link.connect()
    name, kwargs = dev.calls[-1]
    assert name == "connect"
    assert kwargs["device_notification_delay"] == 15
    assert kwargs["card_color"] == le.LEGO_COLOR_GREEN
    assert kwargs["card_serial"] == "0997"


def test_connect_raises_when_the_hub_is_not_found():
    link, _, _ = make(fail_connect=True)
    with pytest.raises(ConnectionError):
        link.connect()


def test_notification_becomes_a_raw_count_imu_sample_stamped_on_arrival():
    link, dev, clock = make()
    link.connect()
    clock.advance_s(0.5)
    dev.emit(imu=(11, 22, 1000, -40, 50, 600))
    sample = link.imu.get_nowait()
    assert sample.t_ns == clock.now_ns()
    assert sample.a == (11, 22, 1000)
    assert sample.g == (-40, 50, 600)
    assert sample.src == "hub"


def test_gesture_notifications_go_to_their_own_queue():
    link, dev, _ = make()
    link.connect()
    dev.emit(gesture=le.MOTION_GESTURE_SHAKE)
    event = link.gestures.get_nowait()
    assert event.gesture == le.MOTION_GESTURE_SHAKE
    assert event.t_ns > 0
    assert link.imu.empty()


def test_no_gesture_marker_is_not_queued():
    link, dev, _ = make()
    link.connect()
    dev.emit(gesture=le.MOTION_GESTURE_NO_GESTURE)
    assert link.gestures.empty()


def test_callback_never_calls_the_device():
    # Sync library calls inside the notification callback raise RuntimeError
    # on the real library, so the callback may only parse and enqueue.
    link, dev, _ = make()
    link.connect()
    before = len(dev.calls)
    dev.emit(imu=(0, 0, 1000, 0, 0, 0), gesture=le.MOTION_GESTURE_TAPPED)
    assert len(dev.calls) == before


def test_stale_ms_tracks_the_clock_since_the_last_notification():
    link, dev, clock = make()
    link.connect()
    assert link.stale_ms() is None
    dev.emit(imu=(0, 0, 1000, 0, 0, 0))
    clock.advance_s(0.3)
    assert link.stale_ms() == pytest.approx(300.0)
    assert link.is_stale(250.0) is True
    assert link.is_stale(350.0) is False


def test_close_stops_the_motors_then_disconnects_and_is_idempotent():
    link, dev, _ = make()
    link.connect()
    link.close()
    link.close()
    names = [c[0] for c in dev.calls]
    assert names[-2:] == ["motor_stop", "disconnect"]
    assert names.count("disconnect") == 1
    assert dev.calls[-2][1]["motor"] == le.MOTOR_BOTH


def test_close_still_disconnects_when_motor_stop_fails():
    link, dev, _ = make()
    link.connect()
    dev.fail_on = {"motor_stop"}
    link.close()
    assert [c[0] for c in dev.calls][-1] == "disconnect"


def test_reconnect_repasses_the_notify_delay():
    # connect() defaults to 100 ms; forgetting the delay on reconnect would
    # silently drop the stream to 10 Hz.
    link, dev, _ = make()
    link.connect()
    dev.connected = False
    link.reconnect()
    assert dev.calls[-1][1]["device_notification_delay"] == 15


def test_battery_percent_is_none_until_the_hub_has_reported():
    link, dev, _ = make()
    link.connect()
    assert link.battery_pct() is None
    dev.set_battery(87)
    assert link.battery_pct() == 87


def test_card_kwargs_normalise_colour_name_and_zero_pad_the_serial():
    assert hubmod.card_kwargs("green", 997) == {"card_color": le.LEGO_COLOR_GREEN, "card_serial": "0997"}
    assert hubmod.card_kwargs("Orange", "1129") == {"card_color": le.LEGO_COLOR_ORANGE, "card_serial": "1129"}
    with pytest.raises(ValueError):
        hubmod.card_kwargs("chartreuse", "0001")
    with pytest.raises(ValueError):
        hubmod.card_kwargs("green", "12345")


def test_a_battery_level_of_zero_is_the_librarys_not_yet_known_value_not_an_empty_battery():
    from pingpong.hub import HubLink
    from pingpong.sources_fake import FakeDoubleMotor

    dev = FakeDoubleMotor()
    link = HubLink(dev)
    dev.set_battery(0)
    assert link.battery_pct() is None
    dev.set_battery(63)
    assert link.battery_pct() == 63


def test_a_packet_the_parser_cannot_read_is_counted_and_the_stream_carries_on():
    link, dev, clock = make()
    link.connect()
    clock.advance_s(0.2)
    link._on_notify(object())                                  # not a notification at all (a library or firmware surprise)
    assert link.n_parse_errors == 1 and link.last_rx_ns == clock.now_ns()      # the hub is still talking
    dev.emit(imu=(1, 2, 1000, 3, 4, 5))
    assert link.imu.get_nowait().g == (3, 4, 5) and link.n_parse_errors == 1


def test_the_failed_pending_responses_at_disconnect_do_not_print_asyncios_never_retrieved_noise(caplog):
    import asyncio
    import gc
    import logging

    loop = asyncio.new_event_loop()
    try:
        with caplog.at_level(logging.ERROR, logger="asyncio"):
            expected = loop.create_future()                     # what the library does to every pending response
            expected.set_exception(ConnectionError("Device disconnected unexpectedly"))
            del expected
            gc.collect()
            assert [r for r in caplog.records if r.name == "asyncio"] == []
            other = loop.create_future()                        # any other forgotten error must still be reported
            other.set_exception(ValueError("something else broke"))
            del other
            gc.collect()
            assert any("something else broke" in r.getMessage() for r in caplog.records)
    finally:
        loop.close()
