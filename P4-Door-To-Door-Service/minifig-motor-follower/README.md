# 🚗 Minifig Motor Follower

Receives the green LEGO minifig's position over MQTT from the laptop tracker
(`laptop-minifig-tracker/live_tracker.py`) and drives the car toward it with two
DC motors via Cytron Maker Drive, steering so the minifig stays centered. Also
shows the minifig's position as a dot on the LED matrix.

## Running

1. On the laptop: `cd laptop-minifig-tracker && python live_tracker.py`
2. On the board: start this app.

## Wiring

Maker Drive uses sign-magnitude (PWM_PWM) control: two inputs per channel,
no DIR pin. Speed is bit-banged with digitalWrite (not analogWrite), so no
pin needs to support hardware PWM.

| Maker Drive input | Board pin |
| --- | --- |
| M1A (left motor) | 3 |
| M1B (left motor) | 11 |
| M2A (right motor) | 4 |
| M2B (right motor) | 7 |

M2 was originally on 9/10; those pins got correct values in the serial log
but never produced motor movement, so M2 was moved to 4/7 to test whether the
fault followed the Maker Drive channel (it didn't move) or the MCU pins.

Also connect the Maker Drive GND to a board GND — without a common ground the
PWM signals have no reference and the motors won't turn.
