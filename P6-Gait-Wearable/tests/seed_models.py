"""Hand-set seed trees for the tests (the files in models/ are the trained ones and change with every session)."""
SEED1 = {
    "features": ["peak_x"],
    "classes": ["RUN", "STAND", "WALK"],
    "root": {"f": "peak_x", "t": 1.15, "l": {"label": "STAND"},
             "r": {"f": "peak_x", "t": 2.2, "l": {"label": "WALK"}, "r": {"label": "RUN"}}},
}
SEED2 = {
    "features": ["tilt", "std_x", "std_perp", "mag_std", "mag_peak", "gyro_rms", "gyro_peak", "dom_freq", "gyro_ratio"],
    "classes": ["BIKE", "RUN", "STAND", "WALK"],
    "root": {"f": "gyro_rms", "t": 40.0, "l": {"label": "STAND"},
             "r": {"f": "tilt", "t": 0.6, "l": {"label": "BIKE"},
                   "r": {"f": "mag_peak", "t": 2.2, "l": {"label": "WALK"}, "r": {"label": "RUN"}}}},
}
