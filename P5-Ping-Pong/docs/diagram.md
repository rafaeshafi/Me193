# Architecture (one Mac process, as built)

```
 PLAYER: Double Motor in the dominant fist (the paddle); off hand = AprilTag cards
   | webcam                                   | BLE: IMU in (<= ~66 Hz), haptic commands out
   v                                          v
+---------------------- MAC: one Python process (P1 pins, Python 3.12) ------------------------+
| [V] VisionWorker thread: frame -> MediaPipe pose -> One-Euro -> PaddlePose (camera-lag stamped)|
|     + AprilTag search in the lobby -> TagVoter (4 of 6 frames; START held 0.4 s)               |
| [H] legoeducation loop thread: callback ONLY parses + stamps arrival time -> queue             |
| [I] ImuWorker thread: SwingDetector (signed axis) + ShakeMonitor (FFT, gate J6) -> events      |
|     samples = the hub's gyro, or (hub < 25 Hz / --no-hub) PoseGyro: hand velocity from poses   |
| [A] Actuator thread: the ONLY sender of hub commands (<= 10 writes/s); blank windows -> [I]    |
|                                                                                                |
| MAIN ~60 Hz  LiveRig.pump():  poses -> sensor health (pause / reconnect) -> tags -> swings     |
|                               -> GameCore.tick -> actuator -> HUD                              |
|   GameCore: judge J1-J6 -> shot (speed, spin, aim, fault) -> physics Leg -> CpuPolicy          |
|             (softmax zones, PD paddle, optional Q-learning) -> rules (Survival / Match)        |
|   Session: events -> haptics + audio + HUD + leaderboard summary                               |
|        |                |                  |                    |                            |
|        v                v                  v                    v                            |
|  [Q] paho thread   [S] Store (SQLite)  [R] Recorder (JSONL)  [Au] Audio mixer stream         |
+--------+----------------+------------------+--------------------+----------------------------+
         v                v                  v                    v
 test.mosquitto.org   data/pingpong.db   recordings/<session>/   the speakers
 ME193/Rogers/RafaeShafi = "12.0"
```

Rules: only queues, immutable snapshots and locked scalars cross threads; the game loop never calls BLE,
MQTT or SQLite synchronously (the actuator thread owns BLE writes, paho owns the network); the camera is
opened on the main thread of the terminal that was granted Camera/Bluetooth; teardown runs in a fixed order
(actuator, audio, camera, IMU, recorder, store, MQTT, hub) and every step is guarded on its own.

Offline: `pingpong/fakerig.py` plays a scripted player through the real parser, hub link, swing detector,
vision worker, tag voter and haptics on a simulated clock (`./pp play --selftest`); `pingpong/replay.py`
feeds a recorded session back through the same code.

## Playing a friend (two Macs, each running everything above)

```
 MAYA's Mac                                                                        RAFAE's Mac
 Flow (ONLINE / WAIT screens) -> actions -> Session._online_action                 same
 online.Online: list, host, join, pairing  <-- lobby: ME193-pp/v1/lobby/<CODE> -->  online.Online
        |  (retained: who hosts, pace, points; cleared by withdraw() or by the host's last will)
        v  pairing: guest says join{gid} until the host answers welcome{to=gid, sid} (or busy{to=gid})
 GameCore.remote = versus.Remote  <-- room: .../room/<CODE>/host | .../guest -->    versus.Remote
        |     hit / miss / rematch: numbered, in order, acked by every ping, resent after 2.5 s   (QoS 1)
        |     pos (my paddle, 15 Hz), ping / pong (the PING chip)                                  (QoS 0)
        v                                         test.mosquitto.org or --net-broker
 my ball leaves here: plan_return(end_z = the far end)      their game starts the same ball at them:
 and waits at the far end for their answer                  plan_mirrored (x -> -x, z -> 2.74 - z, side spin turned)
```

No clock is shared: a ball starts flying when its message is read, so each player has the ball's own flight to react in, and the
receiver's game alone decides hit or miss. While a friend's game is on, `GameCore.play_friend` swaps in a tracker of its own and
no publisher, so nothing in it can reach the score topic; `end_friend` puts the single-player ones back. Only `config.OWNER`'s
player ever publishes the score. An in-memory network (`pingpong/loopnet.py`) and `pingpong/friendrig.py` play two whole
pipelines against each other without a broker.

