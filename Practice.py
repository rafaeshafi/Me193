import time

import legoeducation as le

m = le.SingleMotor()
m.connect(card_color=le.LEGO_COLOR_YELLOW, card_serial=994)
print("connected:", m.connected)

if m.connected:
    m.motor_set_speed(50)
    m.motor_run_for_time(2000)  # spin for 2 s
    time.sleep(0.5)
    m.motor_stop()
    m.disconnect()
