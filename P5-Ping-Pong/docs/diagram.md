# Architecture (one Mac process)

```
 PLAYER: dominant fist = Double Motor (paddle); off hand = AprilTag cards (or a stand in frame)
   | webcam                                   | BLE: IMU 15 ms notify in, haptic cmds out
   v                                          v
+--------------------- MAC: one Python process (P1 pins, Python 3.12) ---------------------+
| [V] VisionWorker: newest frame, t_read stamp, unmirrored 640x360 -> MediaPipe pose       |
|     -> One-Euro -> immutable PoseSnapshot (tuple swap, no shared deque)                  |
| [T] TagWorker: 960 px gray, LOBBY only -> TagVoter (4 of 6 frames; START held 0.4 s)     |
| [H] legoeducation loop thread: callback ONLY parses + stamps monotonic_ns -> queue       |
| [I] IMU worker: blocks on queue -> SwingDetector -> SwingEvent -> main queue             |
| [A] Actuator: sole sender of hub commands, <=10 writes/s, reports real write times       |
| MAIN ~60 Hz: judge -> shot -> Leg physics -> Opponent -> Rules/Modes -> Haptics -> HUD   |
|      |                      |                       |                                    |
|      v                      v                       v                                    |
| [Q] paho thread      [S] StoreWriter (SQLite)   [L] per-ball recorder (JSONL)            |
+------+----------------------+-----------------------+------------------------------------+
       v                      v                       v
test.mosquitto.org:1883    data/pingpong.db        recordings/<session>/<ball>.jsonl
ME193/Rogers/RafaeShafi = "12.0"
```
