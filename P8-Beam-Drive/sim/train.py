"""Train the Beam-Straight task. Same as mjlab's own `train`, but it registers our task first.

    python train.py Beam-Straight --env.scene.num-envs 4096
"""
import sys

import beam_task  # noqa: F401  (registers "Beam-Straight")
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
