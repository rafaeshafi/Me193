"""PingPong scoreboard entry point on the UNO Q; App Lab runs this inside the app container."""
import logging

from arduino.app_utils import App, Bridge

from scoreboard import PORT, Scoreboard, levels, serve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

board = Scoreboard(draw=lambda frame: Bridge.call("draw", levels(frame), timeout=2))
serve(board, port=PORT)
App.run(user_loop=board.loop)
