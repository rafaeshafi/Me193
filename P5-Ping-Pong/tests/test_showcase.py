"""./pp showcase: the intro as a video and a picture of every screen, to look at without playing."""

import cv2
import pytest

from tools import showcase

PICTURES = {"title", "mode", "mode_online", "online", "online_empty", "wait", "vs_friend", "win_friend", "match_friend", "opponent", "vs", "win",
            "lose", "record", "rally", "match", "countdown"}


def test_it_writes_a_picture_of_every_screen_and_the_intro_as_a_video(tmp_path):
    assert showcase.main(["--out", str(tmp_path), "--size", "192x108", "--fps", "5"]) == 0
    assert {p.stem for p in tmp_path.glob("*.png")} == PICTURES
    for p in tmp_path.glob("*.png"):
        image = cv2.imread(str(p))
        assert image.shape == (108, 192, 3) and image.std() > 10, p.name
    cap = cv2.VideoCapture(str(tmp_path / "intro.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 40 and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 192
    cap.release()


def test_the_video_can_be_left_out(tmp_path):
    assert showcase.main(["--out", str(tmp_path), "--size", "192x108", "--no-video"]) == 0
    assert not (tmp_path / "intro.mp4").exists() and (tmp_path / "title.png").exists()


def test_a_size_that_is_not_width_x_height_is_refused(tmp_path, capsys):
    assert showcase.main(["--out", str(tmp_path), "--size", "big"]) == 2
    assert "WIDTHxHEIGHT" in capsys.readouterr().err


def test_it_checks_itself():
    assert showcase.main(["--selftest"]) == 0
