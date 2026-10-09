"""The game shows its score on the UNO Q's matrix: every state it draws goes to the board link, and --no-board leaves it alone."""

import play

from pingpong import hud


class FakeBoard:
    def __init__(self):
        self.shown, self.started, self.closed = [], False, False

    def show(self, state):
        self.shown.append(state)

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


def test_the_window_loop_gives_the_board_every_state_it_draws():
    from pingpong import fakerig

    rig, board, frames = fakerig.FakeRig(), FakeBoard(), []
    play.run_loop(rig.rig, show=frames.append, wait_key=lambda ms: ord("q") if len(frames) >= 3 else 255, board=board)
    assert len(frames) == 3 and len(board.shown) >= 3
    assert all(isinstance(state, hud.HudState) for state in board.shown)


def test_the_fake_window_loop_gives_the_board_its_states_too():
    session = play.make_fake_session(play.parse_args(["--fake", "--no-intro", "--no-audio"]))
    board, frames = FakeBoard(), []
    play.fake_loop(session, show=frames.append, wait_key=lambda ms: ord("q") if len(frames) >= 3 else 255,
                   mouse_xy=lambda: (640, 500), board=board)
    assert len(board.shown) >= 3


def test_a_bug_in_what_the_board_is_shown_costs_a_frame_and_not_the_game():
    from pingpong import fakerig

    class Broken(FakeBoard):
        def show(self, state):
            raise ValueError("a bug in the matrix")

    rig, frames, said = fakerig.FakeRig(), [], []
    play.run_loop(rig.rig, show=frames.append, wait_key=lambda ms: ord("q") if len(frames) >= 3 else 255, board=Broken(), log=said.append)
    assert len(frames) >= 1 and any("a bug in the matrix" in line for line in said)


def test_the_board_is_on_unless_it_is_turned_off_and_is_started_and_closed_with_the_game():
    assert play.parse_args([]).no_board is False and play.parse_args(["--no-board"]).no_board is True
    made = FakeBoard()
    assert play.open_board(play.parse_args([]), for_game=lambda log: made) is made and made.started
    assert play.open_board(play.parse_args(["--no-board"]), for_game=lambda log: (_ for _ in ()).throw(AssertionError("asked"))) is None
    assert play.open_board(play.parse_args([]), for_game=lambda log: None) is None                     # no adb: no board, no fuss
