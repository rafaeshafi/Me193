"""The fakes enforce the REAL library signatures, so a misnamed argument fails in the tests, not on the hardware."""

import pytest

from pingpong.sources_fake import FakeDoubleMotor, FakeMqttClient


def test_the_fake_hub_accepts_exactly_what_the_real_double_motor_accepts():
    dev = FakeDoubleMotor()
    dev.connect(card_color=1, card_serial="1131", device_notification_delay=15)
    dev.motor_run_for_time(60, direction=0, motor=1, speed=100, blocking=False)
    dev.beep(pattern=0, frequency=880, count=2, blocking=False)
    dev.light_color(3, pattern=1, blocking=False)
    dev.motor_stop(motor=2, blocking=False)
    dev.begin_batch()
    dev.end_batch(blocking=False)
    dev.cancel_batch()
    dev.disconnect()


@pytest.mark.parametrize("call", [
    lambda d: d.beep(volume=3),                                   # no such parameter
    lambda d: d.beep(880),                                        # frequency is keyword-only
    lambda d: d.motor_run_for_time(duration_ms=60),               # the argument is time_ms
    lambda d: d.motor_run_for_time(60, power=100),
    lambda d: d.light_color(color=3, colour=4),
    lambda d: d.motor_stop(both=True),
    lambda d: d.connect(color="red", serial="1131"),              # the real names are card_color / card_serial
    lambda d: d.end_batch(True, False),
])
def test_a_call_the_real_hub_would_reject_fails_loudly_in_the_fake(call):
    with pytest.raises(TypeError):
        call(FakeDoubleMotor())


def test_the_fake_broker_client_accepts_exactly_what_paho_accepts():
    client = FakeMqttClient()
    client.will_set("t/status", "offline", qos=1, retain=True)
    client.reconnect_delay_set(1, 30)
    client.connect_async("test.mosquitto.org", 1883, 30)
    client.loop_start()
    client.publish("t/x", "1.0", qos=1, retain=True)
    client.subscribe("t/#", qos=1)
    client.disconnect()
    client.loop_stop()


@pytest.mark.parametrize("call", [
    lambda c: c.publish("t", "1.0", quality=1),
    lambda c: c.will_set("t", message="offline"),
    lambda c: c.connect_async(hostname="x"),
    lambda c: c.subscribe("t", qos=1, retain=True),
    lambda c: c.reconnect_delay_set(minimum=1),
])
def test_a_call_paho_would_reject_fails_loudly_in_the_fake(call):
    with pytest.raises(TypeError):
        call(FakeMqttClient())
