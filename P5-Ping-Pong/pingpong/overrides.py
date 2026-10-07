"""--set section.name=value: change the swing detector, the judge or the levels without editing code.

    swing.<field of SwingParams>   e.g. swing.t_pk=150          (the weakest swing that counts, in dps)
    judge.<setting of HitJudge>    e.g. judge.d95_s=0.2         (how long after the window a miss is declared)
    level.<field of Level>         e.g. level.radius_sw=0.8     (applies to every level; tags and keys keep it)
    latency.<field of Latency>     e.g. latency.display_s=0.08  (where the time goes: see pingpong/latency.py)

`./pp play --set ...` tunes a live game; `./pp replay --set ...` asks what the same recorded swings would have
done.  A live session records the settings it ran with, and its replay starts from them.
"""

import dataclasses

from pingpong import latency as latency_mod
from pingpong import levels
from pingpong.swing import SwingParams

JUDGE_SETTINGS = ("t_pk", "d95_s", "min_dur_ms", "max_dur_ms", "max_reversals", "min_conf", "refractory_s",
                  "max_hits_per_s", "contact_lag_s")
SECTIONS = {"swing": tuple(f.name for f in dataclasses.fields(SwingParams)), "judge": JUDGE_SETTINGS,
            "level": tuple(f.name for f in dataclasses.fields(levels.Level)),
            "latency": tuple(f.name for f in dataclasses.fields(latency_mod.Latency))}


def parse(items):
    """["level.late_s=0.4", ...] -> {"level": {"late_s": 0.4}}."""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        section, dot, name = key.partition(".")
        if not (sep and dot and name):
            raise ValueError(f"expected section.name=value, got {item!r}")
        try:
            number = float(value) if "." in value or "e" in value.lower() else int(value)
        except ValueError:
            raise ValueError(f"{item!r}: the value must be a number") from None
        out.setdefault(section, {})[name] = number
    return out


def check(settings):
    for section, entries in settings.items():
        if section not in SECTIONS:
            raise ValueError(f"unknown section {section!r}: choose from {', '.join(SECTIONS)}")
        for name in entries:
            if name not in SECTIONS[section]:
                raise ValueError(f"unknown {section} setting {name!r}: choose from {', '.join(SECTIONS[section])}")
    return settings


def merge(base, extra):
    """Both sets of settings, the later one winning; neither input is changed."""
    out = {section: dict(entries) for section, entries in (base or {}).items()}
    for section, entries in (extra or {}).items():
        out.setdefault(section, {}).update(entries)
    return out


def format_settings(settings):
    return ", ".join(f"{section}.{name}={value}" for section, entries in settings.items()
                     for name, value in entries.items())


def apply(rig, settings):
    """Put the settings into a built LiveRig (the detector, the judge and the game's level)."""
    game = rig.session.game
    if "swing" in settings:
        rig.imu.set_params(dataclasses.replace(rig.imu.detector.p, **settings["swing"]))
        if "t_pk" in settings["swing"] and "t_pk" not in settings.get("judge", {}):
            game.judge.t_pk = settings["swing"]["t_pk"]                  # the judge's J3 follows the detector
    if "latency" in settings:
        lat = dataclasses.replace(rig.session.latency, **settings["latency"])
        rig.session.latency = rig.session.view.latency = lat
        game.judge.contact_lag_s = lat.contact_lag_s                       # the judge follows (unless set by hand below)
    for name, value in settings.get("judge", {}).items():
        setattr(game.judge, name, value)
    if "level" in settings:
        game.level_overrides = dict(settings["level"])
        game.set_level(game.level)
