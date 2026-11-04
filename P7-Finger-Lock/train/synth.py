"""Synthetic hands, so the whole pipeline (dataset, training, export, board engine, movie) runs before the first real
picture is taken. A palm is a filled ellipse at the bottom of the crop and each finger a thick line fanning out of it;
pose, size, brightness and background vary; some noise and a soft shadow. Not a substitute for real data."""
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def hand(n_fingers, rng, size=128):
    """uint8 (size, size, 3) BGR picture of a hand showing n_fingers (0 = fist)."""
    bg = int(rng.integers(60, 140))
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    grad = np.linspace(-20, 20, size)[None, :, None] * rng.uniform(-1, 1)
    img = np.clip(img.astype(np.float32) + grad, 0, 255).astype(np.uint8)
    skin = (int(rng.integers(120, 200)), int(rng.integers(140, 215)), int(rng.integers(170, 240)))  # BGR, reddish-tan
    cx, cy = size // 2 + int(rng.integers(-10, 10)), int(size * 0.68) + int(rng.integers(-8, 8))
    pw, ph = int(size * rng.uniform(0.15, 0.19)), int(size * rng.uniform(0.16, 0.21))
    tilt = float(rng.uniform(-18, 18))
    cv2.ellipse(img, (cx, cy), (pw, ph), tilt, 0, 360, skin, -1)
    thickness = int(size * rng.uniform(0.045, 0.06))
    length = size * rng.uniform(0.34, 0.44)
    spread = rng.uniform(50, 80) if n_fingers > 1 else 0
    angles = np.linspace(-spread / 2, spread / 2, n_fingers) if n_fingers else []
    for a in angles:
        rad = np.deg2rad(a + tilt - 90)
        tip = (int(cx + length * np.cos(rad)), int(cy - ph * 0.6 + length * np.sin(rad)))
        base = (int(cx + pw * 0.5 * np.cos(rad)), int(cy - ph * 0.6 + pw * 0.3 * np.sin(rad)))
        cv2.line(img, base, tip, skin, thickness, cv2.LINE_AA)
        cv2.circle(img, tip, thickness // 2, skin, -1, cv2.LINE_AA)
    noise = rng.normal(0, rng.uniform(2, 8), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if cv2 is not None:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def dataset(per_class=60, seed=0, size=128, sessions=3):
    """-> (images uint8 (N, size, size, 3), labels (N,), session ids (N,))."""
    rng = np.random.default_rng(seed)
    imgs, labels, sess = [], [], []
    for s in range(sessions):
        for label in range(6):
            for _ in range(per_class // sessions):
                imgs.append(hand(label, rng, size))
                labels.append(label)
                sess.append(f"synth{s}")
    return np.stack(imgs), np.asarray(labels), np.asarray(sess)


def write(root, per_class=60, seed=0):
    """Write the synthetic set in the data/raw/<label>/<session>-<n>.png layout."""
    from pathlib import Path
    imgs, labels, sess = dataset(per_class, seed)
    root = Path(root)
    counts = {}
    for img, label, s in zip(imgs, labels, sess):
        d = root / str(label)
        d.mkdir(parents=True, exist_ok=True)
        n = counts[(label, s)] = counts.get((label, s), 0) + 1
        cv2.imwrite(str(d / f"{s}-{n:04d}.png"), img)
    return len(imgs)
