"""The Session's side of playing a friend: carrying out what the online screens ask for (open the list, host, join, cancel, leave,
ask for a rematch), turning a pair into a game, and putting the session back as it was afterwards.

While a game with a friend is on, the GameCore has the friend's Remote, a tracker of its own and no publisher: what happens in it can
never reach the score topic or the single-player record, and when it is over the session plays the computer again as before.
"""

from pingpong import levels
from pingpong import online as online_mod
from pingpong.uistate import OnlineView


class OnlineMixin:
    online = None                    # an online.Online, or None when this session cannot play a friend
    opponent_name = ""               # the friend being played ("" against the computer)
    _versus_saved = None             # what the game had before the friend came (rules.GameCore.play_friend)
    _net_target = 7                  # the points a game this player hosts is played to

    def attach_online(self, online):
        self.online, self._net_target = online, self.game.target_points

    def close_online(self):
        """Say goodbye to a friend and leave the list (the player quit)."""
        if self.online is not None:
            self.online.close()

    # --- what the screens ask for ----------------------------------------------------------------------------------------------------
    def _online_action(self, verb, arg):
        online, now = self.online, self.clock.now_ns()
        if online is None:
            if verb == "open":
                self.flow.set_online(OnlineView(status="offline", message="ONLINE PLAY IS NOT SET UP"))
            elif verb in ("host", "join"):
                self.apply(self.flow.net_failed(now))
            return
        if verb == "open":
            online.open()
        elif verb == "close":
            online.close()
        elif verb == "host":
            online.host(arg, self._net_target)
        elif verb == "join":
            online.join(arg)
        elif verb == "cancel":
            online.cancel()
        elif verb == "leave":
            self._end_versus()
            online.leave()
        elif verb == "rematch" and self.game.remote is not None:
            self.game.remote.send_rematch()

    # --- the clock ---------------------------------------------------------------------------------------------------------------------
    def _online_tick(self, now):
        online, flow = self.online, self.flow
        if online is None:
            return
        for event in online.step(now):
            if isinstance(event, online_mod.Paired):
                self._begin_versus(event, now)
            else:
                self.apply(flow.net_failed(now))
        flow.set_online(online.view())
        remote = self.game.remote
        if remote is None:
            return
        rest = self.view.rest(now)
        if rest is not None:
            remote.set_paddle(rest[0])
        if remote.gone and not flow.opponent_gone:
            flow.opponent_left(True)
        if flow.rematch_pending and remote.rematch_seen and self.game.phase == "MATCH_OVER":
            self.apply(flow.start_rematch(now))

    def _begin_versus(self, pairing, now):
        """A friend is found: the game becomes one with them, at the host's pace and to the host's score."""
        game = self.game
        if game.remote is None:
            self._versus_saved = game.play_friend(pairing.remote, pairing.target)
        else:                                                  # (a second pair on the same session: keep what was first set aside)
            game.remote, game.target_points = pairing.remote, pairing.target
        game.set_level(levels.LEVELS[pairing.pace])
        game.set_mode("match")
        self.opponent_name = pairing.opponent
        self.apply(self.flow.paired(pairing.opponent, pairing.pace, now))

    def _end_versus(self):
        if self._versus_saved is not None:
            self.game.end_friend(self._versus_saved)
        self._versus_saved, self.opponent_name = None, ""

    def _waiting_text(self):
        """The words over a game that waits for the friend (""  when it is not waiting)."""
        if self.game.remote is not None and "opponent" in self.game.pause_reasons:
            return f"WAITING FOR {self.opponent_name}..."
        return ""
