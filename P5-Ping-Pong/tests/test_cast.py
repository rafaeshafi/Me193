"""The cast: the three opponents (one per level) and an avatar for every player, made from their name."""

import dataclasses

import pytest

from pingpong import cast, levels


def test_every_level_a_card_can_set_has_its_own_opponent():
    assert set(cast.OPPONENTS) == {1, 2, 3}
    names = [cast.OPPONENTS[n].name for n in (1, 2, 3)]
    assert len(set(names)) == 3
    for level in (1, 2, 3):
        assert cast.opponent_for(levels.LEVELS[level]) is cast.OPPONENTS[level]


def test_an_opponent_has_a_name_a_line_to_say_and_a_look_of_its_own():
    looks = [cast.OPPONENTS[n] for n in (1, 2, 3)]
    for look in looks:
        assert look.name and look.tagline and look.hair_style in cast.HAIR_STYLES and look.accessory in cast.ACCESSORIES
    assert len({look.shirt for look in looks}) == 3 and len({look.hair for look in looks}) == 3
    assert len({(look.hair_style, look.accessory) for look in looks}) == 3


def test_stars_say_how_hard_an_opponent_is_and_grow_with_the_level():
    for kind in ("speed", "reach", "spin"):
        stars = [cast.stars(levels.LEVELS[n])[kind] for n in (1, 2, 3)]
        assert all(1 <= s <= 5 for s in stars) and stars == sorted(stars) and stars[0] < stars[2]


def test_a_players_avatar_is_the_same_every_time_and_comes_from_the_name_alone():
    assert cast.player_look("rafae") == cast.player_look("rafae")
    assert cast.player_look("Rafae") == cast.player_look("rafae")                  # the case of a name is not a new person
    assert cast.player_look("rafae").name == "RAFAE"


def test_different_players_look_different():
    looks = {cast.player_look(n) for n in ("rafae", "maya", "omar", "zed", "guest", "ana", "li", "sam", "dee", "kai")}
    assert len(looks) >= 9
    assert len({look.skin for look in looks}) >= 3 and len({look.hair for look in looks}) >= 4
    assert len({look.shirt for look in looks}) >= 4


def test_a_players_avatar_never_wears_an_opponents_shirt_or_gets_a_stranger_style():
    for name in ("rafae", "maya", "omar", "zed", "guest"):
        look = cast.player_look(name)
        assert look.hair_style in cast.HAIR_STYLES and look.accessory in cast.ACCESSORIES
        assert look.shirt not in {o.shirt for o in cast.OPPONENTS.values()}


def test_an_empty_name_still_gets_an_avatar():
    assert cast.player_look("").name == "PLAYER"


def test_looks_are_frozen_so_a_screen_cannot_change_a_character_by_accident():
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast.OPPONENTS[1].name = "Someone"
