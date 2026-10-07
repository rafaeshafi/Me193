# Fine-tunes a YOLOv8 detector to find the green LEGO minifigure.
#
# Dataset: dataset/ (Roboflow YOLOv8 export, class "Minifig").
# Drop your Roboflow export in as a "dataset" folder next to this file
# (it must contain data.yaml at its root) before running this.
#
# Training progress and the final weights land under
# runs/detect/green_minifig-2/weights/best.pt -- re-running this overwrites
# that same folder (live_tracker.py's default --model points there).

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).parent / "dataset" / "data.yaml"
EPOCHS = 100  # small dataset (~50-100 images) benefits from more epochs than the usual 50
IMAGE_SIZE = 640
DEVICE = "mps"  # Apple Silicon GPU; falls back to CPU below if unavailable

if not DATA_YAML.exists():
    raise SystemExit(
        f"Dataset not found at {DATA_YAML}.\n"
        "Export your Roboflow project as 'YOLOv8' and unzip it into a 'dataset' "
        "folder next to this script, then run again."
    )

# Start from COCO-pretrained weights and fine-tune -- much faster than
# training from scratch, and works well with a few dozen images.
model = YOLO("yolov8n.pt")

try:
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        name="green_minifig-2",
        exist_ok=True,  # overwrite the same run each time instead of forking green_minifig-3, -4, ...
        device=DEVICE,
    )
except Exception as e:
    print(f"Training on device={DEVICE!r} failed ({e}). Retrying on CPU...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        name="green_minifig-2",
        exist_ok=True,  # overwrite the same run each time instead of forking green_minifig-3, -4, ...
        device="cpu",
    )

print("Done. Weights saved to runs/detect/green_minifig-2/weights/best.pt")
