"""List LEGO Education hubs in BLE range (passive scan; connects to nothing).

Usage:
    ./pp scan_hubs [--secs 8]
    ./pp scan_hubs --selftest

Wake your Double Motor with its button, hold it next to the laptop and run
this: the nearest hub (strongest signal) is almost certainly yours; its card
colour + serial are what config_local.json / env_check need.  13-18 LEGO
motors can be in range in a classroom, so the tools never "try all cards".
"""

import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import legoeducation as le  # noqa: E402
from legoeducation.basic_ble import SERVICE_UUID, BasicBLE  # noqa: E402

COLOR_NAMES = {
    le.LEGO_COLOR_GREEN: "green", le.LEGO_COLOR_BLUE: "blue", le.LEGO_COLOR_RED: "red",
    le.LEGO_COLOR_ORANGE: "orange", le.LEGO_COLOR_YELLOW: "yellow", le.LEGO_COLOR_AZURE: "azure",
    le.LEGO_COLOR_PURPLE: "purple", le.LEGO_COLOR_MAGENTA: "magenta",
    le.LEGO_COLOR_WHITE: "white", le.LEGO_COLOR_TEAL: "teal", le.LEGO_COLOR_NOCOLOR: "none",
}


def row_from_advertisement(address, adv):
    """One table row from a BLE advertisement, or None if it is not a LEGO hub."""
    product_id, color, serial = BasicBLE._extract_manufacturer_info(adv)
    if product_id is None:
        return None
    return {
        "address": address,
        "name": getattr(adv, "local_name", None) or "",
        "rssi": getattr(adv, "rssi", None),
        "product": f"0x{product_id:04x}",
        "color": COLOR_NAMES.get(color, f"?{color}"),
        "serial": f"{serial:04d}",
    }


def merge_rows(rows):
    """Dedupe by address (keep the strongest sighting), strongest signal first."""
    best = {}
    for row in rows:
        if row is None:
            continue
        old = best.get(row["address"])
        if old is None or (row["rssi"] or -999) > (old["rssi"] or -999):
            best[row["address"]] = row
    return sorted(best.values(), key=lambda r: r["rssi"] if r["rssi"] is not None else -999, reverse=True)


def format_rows(rows):
    if not rows:
        return ("No LEGO hubs seen. Wake the Double Motor with its button, keep it within a metre "
                "of the laptop, and check Bluetooth permission for this terminal.")
    lines = [f"{'colour':<8} {'serial':<6} {'rssi':>5}  {'product':<8} {'address'}"]
    for i, r in enumerate(rows):
        mark = "  <- nearest" if i == 0 else ""
        lines.append(f"{r['color']:<8} {r['serial']:<6} {str(r['rssi']):>5}  {r['product']:<8} "
                     f"{r['address']}{mark}")
    return "\n".join(lines)


async def scan(seconds):
    from bleak import BleakScanner

    rows = []

    def on_detect(device, adv):
        rows.append(row_from_advertisement(device.address, adv))

    async with BleakScanner(detection_callback=on_detect, service_uuids=[SERVICE_UUID]):
        await asyncio.sleep(seconds)
    return merge_rows(rows)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if "--selftest" in argv:
        from types import SimpleNamespace

        data = bytes([0x02, 0x01, 6, 997 & 0xFF, 997 >> 8])
        adv = SimpleNamespace(manufacturer_data={0x0397: data}, rssi=-42, local_name="LEGO", service_uuids=[])
        rows = merge_rows([row_from_advertisement("AA:BB", adv), row_from_advertisement("CC", SimpleNamespace(
            manufacturer_data={}, rssi=-10, local_name="x", service_uuids=[]))])
        assert len(rows) == 1 and rows[0]["color"] == "green" and rows[0]["serial"] == "0997"
        assert "<- nearest" in format_rows(rows)
        print("scan_hubs selftest OK")
        return 0
    from pingpong import hostcheck

    hostcheck.require_host("Bluetooth")
    secs = float(argv[argv.index("--secs") + 1]) if "--secs" in argv else 8.0
    print(f"scanning {secs:.0f} s (passive; nothing is connected) ...")
    print(format_rows(asyncio.run(scan(secs))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
