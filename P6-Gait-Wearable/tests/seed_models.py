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


import random


def seed_stepnet(seed=0, channels=12, kernel=5, dilations=(1, 2, 4), pm_pct95=100.0):
    """A StepNet spec with small deterministic random weights: the right shape, no skill (tests and the first deploy)."""
    rng = random.Random(seed)
    layers, cin = [], 6
    for d in dilations:
        layers.append({"w": [[[rng.uniform(-0.3, 0.3) for _ in range(kernel)] for _ in range(cin)] for _ in range(channels)],
                       "b": [rng.uniform(-0.1, 0.1) for _ in range(channels)], "dilation": d, "act": "relu"})
        cin = channels
    layers.append({"w": [[[rng.uniform(-0.3, 0.3)] for _ in range(cin)]], "b": [-1.0], "dilation": 1, "act": "softplus"})
    return {"rate_hz": 25, "decimate": 4, "dt": 0.04, "kernel": kernel,
            "mean": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "std": [0.5, 0.3, 0.3, 100.0, 50.0, 50.0],
            "layers": layers, "meta": {"trained": None, "recordings": 0, "loro": {"pm_pct95": pm_pct95}}}
