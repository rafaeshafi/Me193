from types import SimpleNamespace

from tools import scan_hubs

LEGO_COMPANY_ID = 0x0397
FW_GREEN, FW_ORANGE = 6, 8  # firmware colour codes (rpc_message.LEGO_COLOR_*)


def adv(fw_color, serial, rssi=-50, group=0x02, device=0x01, name="LEGO Hub"):
    data = bytes([group, device, fw_color, serial & 0xFF, serial >> 8])
    return SimpleNamespace(manufacturer_data={LEGO_COMPANY_ID: data}, rssi=rssi,
                           local_name=name, service_uuids=[])


def test_row_decodes_card_colour_and_zero_padded_serial():
    row = scan_hubs.row_from_advertisement("AA:BB", adv(FW_GREEN, 997))
    assert row["color"] == "green"
    assert row["serial"] == "0997"
    assert row["rssi"] == -50
    assert row["address"] == "AA:BB"


def test_non_lego_advertisements_are_ignored():
    other = SimpleNamespace(manufacturer_data={0x004C: b"\x01\x02"}, rssi=-40, local_name="x",
                            service_uuids=[])
    assert scan_hubs.row_from_advertisement("CC:DD", other) is None


def test_rows_sort_strongest_signal_first_and_dedupe_by_address():
    rows = scan_hubs.merge_rows([
        scan_hubs.row_from_advertisement("A", adv(FW_GREEN, 997, rssi=-80)),
        scan_hubs.row_from_advertisement("B", adv(FW_ORANGE, 1129, rssi=-45)),
        scan_hubs.row_from_advertisement("A", adv(FW_GREEN, 997, rssi=-60)),
    ])
    assert [r["address"] for r in rows] == ["B", "A"]
    assert rows[1]["rssi"] == -60  # best (strongest) sighting of A is kept


def test_table_marks_the_nearest_hub_as_probably_yours():
    rows = scan_hubs.merge_rows([
        scan_hubs.row_from_advertisement("B", adv(FW_ORANGE, 1129, rssi=-45)),
        scan_hubs.row_from_advertisement("A", adv(FW_GREEN, 997, rssi=-80)),
    ])
    text = scan_hubs.format_rows(rows)
    lines = [l for l in text.splitlines() if "orange" in l or "green" in l]
    assert "<- nearest" in lines[0] and "orange" in lines[0] and "1129" in lines[0]
    assert "<- nearest" not in lines[1]


def test_empty_scan_explains_what_to_do():
    assert "wake" in scan_hubs.format_rows([]).lower()


def test_help_works_anywhere_without_touching_bluetooth(capsys):
    import pytest

    with pytest.raises(SystemExit) as stop:
        scan_hubs.main(["--help"])
    assert stop.value.code == 0 and "--secs" in capsys.readouterr().out


def test_an_unknown_option_is_refused_instead_of_starting_a_scan(capsys):
    import pytest

    with pytest.raises(SystemExit) as stop:
        scan_hubs.main(["--sec", "x"])
    assert stop.value.code == 2
