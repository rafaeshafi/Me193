from display import payload, wave_byte, wave_bytes


def test_payload_keys_and_agreement():
    p = payload("BIKE", "STAND", cad=72, pk=1.13, prompt="BIKE 12s", st="REC")
    assert p == "act=BIKE;m1=STAND;agree=0;cad=72;pk=1.1;prompt=BIKE 12s;st=REC"
    assert "agree=1" in payload("WALK", "WALK", cad=110, pk=1.6)
    assert payload("---", "---", cad=None, pk=None) == "act=---;m1=---;agree=1;cad=--;pk=--;prompt=;st="


def test_payload_is_short_and_sanitised():
    p = payload("NO IMU", "---", None, None, prompt="x;y=z" * 10, st="a=b;c")
    assert ";" not in p.split("prompt=")[1].split(";st=")[0]
    assert len(p) < 120


def test_wave_bytes_map_minus4_to_plus8_g():
    assert wave_byte(-4.0) == 0 and wave_byte(8.0) == 255
    assert wave_byte(1.0) == 106
    assert wave_byte(-9.0) == 0 and wave_byte(20.0) == 255
    assert wave_bytes([-4.0, 1.0, 8.0]) == bytes([0, 106, 255])
