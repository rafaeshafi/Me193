# Runs the fine-tuned YOLOv8 model on a live webcam feed, draws the
# bounding box + centroid of the best detection, and publishes the
# centroid over MQTT for the UNO Q to pick up and drive motors with.
#
# Usage:
#   python live_tracker.py
#   python live_tracker.py --model runs/detect/green_minifig/weights/best.pt --mqtt-host localhost
#
# Press 'q' in the preview window to quit.

import argparse
import json
import time

import cv2
from ultralytics import YOLO
import paho.mqtt.client as mqtt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/detect/green_minifig-2/weights/best.pt")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    p.add_argument("--mqtt-host", default="localhost")
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--topic", default="minifig/centroid")
    p.add_argument("--no-preview", action="store_true", help="Run headless, no cv2 window")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    # Async connect: won't crash if the broker isn't up yet, and keeps
    # retrying in the background so it picks up the connection once it is.
    client.connect_async(args.mqtt_host, args.mqtt_port)
    client.loop_start()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Dropped frame, retrying...")
                time.sleep(0.05)
                continue

            frame_h, frame_w = frame.shape[:2]
            results = model.predict(frame, conf=args.conf, verbose=False)[0]

            best = None
            for box in results.boxes:
                conf = float(box.conf[0])
                if best is None or conf > best["conf"]:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    best = {"conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}

            payload = {"found": False}

            if best is not None:
                cx = (best["x1"] + best["x2"]) / 2
                cy = (best["y1"] + best["y2"]) / 2
                payload = {
                    "found": True,
                    "cx": cx,
                    "cy": cy,
                    "w": best["x2"] - best["x1"],
                    "h": best["y2"] - best["y1"],
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                    # Normalized to [-1, 1], 0 = centered. The UNO Q sketch
                    # consumes these directly, independent of camera resolution.
                    "x_norm": (cx - frame_w / 2) / (frame_w / 2),
                    "y_norm": (cy - frame_h / 2) / (frame_h / 2),
                    "conf": best["conf"],
                }

                if not args.no_preview:
                    cv2.rectangle(
                        frame,
                        (int(best["x1"]), int(best["y1"])),
                        (int(best["x2"]), int(best["y2"])),
                        (0, 255, 0),
                        2,
                    )
                    cv2.drawMarker(
                        frame, (int(cx), int(cy)), (0, 0, 255),
                        markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2,
                    )
                    cv2.putText(
                        frame, f"({cx:.0f}, {cy:.0f}) conf={best['conf']:.2f}",
                        (int(best["x1"]), max(0, int(best["y1"]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )

            client.publish(args.topic, json.dumps(payload))

            if not args.no_preview:
                cv2.imshow("minifig tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
