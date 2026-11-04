import json

import pytest

from lock import config


def test_load_repo_config():
    cfg = config.load()
    assert cfg["code"] == [2, 5, 3]
    assert len(cfg["roi"]) == 3


@pytest.mark.parametrize("bad", [[1, 1, 2], [0, 1, 2], [1, 2], [1, 2, 6], [1, 2, 3, 4]])
def test_bad_codes_refused(bad):
    with pytest.raises(ValueError):
        config.validate({"code": bad})


def test_bad_roi_refused():
    with pytest.raises(ValueError):
        config.validate({"roi": [0.8, 0.1, 0.5]})


def test_defaults_fill_in(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"code": [1, 2, 3]}))
    cfg = config.load(p)
    assert cfg["hold_frames"] == config.DEFAULTS["hold_frames"]
