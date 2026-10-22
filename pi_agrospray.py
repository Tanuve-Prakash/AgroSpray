#!/usr/bin/env python3
"""
AgroSpray - Raspberry Pi controller
Loop: capture -> segment leaf -> damage % -> VGG16 classify -> decide -> ESP32 sprays/moves

Examples
  # Real robot (Pi Camera + ESP32 on USB serial)
  python3 pi_agrospray.py --model models/agrospray_vgg16.tflite --classes models/class_names.json --port /dev/ttyUSB0

  # Real robot + send every stop to the dashboard backend (queues offline, retries later)
  python3 pi_agrospray.py --model models/agrospray_vgg16.tflite --classes models/class_names.json \\
      --port /dev/ttyUSB0 --server http://192.168.1.10:8000 --api-key YOUR_KEY

  # Bench test with saved leaf photos, no hardware
  python3 pi_agrospray.py --source ./test_leaves --model models/agrospray_vgg16.keras --classes models/class_names.json --dry-run

  # Vision-only test (no model file): damage % decides everything
  python3 pi_agrospray.py --source ./test_leaves --no-model --dry-run
"""
import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("agrospray")

# --------------------------------------------------------------------------
# Tunables - calibrate these on your own crops and lighting
# --------------------------------------------------------------------------
IMG_SIZE = 224
VGG_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)

HSV_GREEN = ((30, 40, 30), (90, 255, 255))    # healthy tissue
HSV_LESION = ((5, 40, 30), (30, 255, 255))    # yellow / brown tissue
MIN_LEAF_FRAC = 0.03       # ignore green blobs smaller than 3% of the frame
DEAD_BROWN_FRAC = 0.15     # no green at all but this much brown -> dead crop

HEALTHY_MAX = 5.0          # damage % below this = healthy
HEALTHY_OVERRIDE = 25.0    # classifier says healthy and damage below this = healthy
# (tier name, upper damage %, dose ml/L at lower edge, dose ml/L at upper edge)
TIERS = [
    ("MILD", 20.0, 5.0, 10.0),
    ("MODERATE", 45.0, 10.0, 20.0),
    ("SEVERE", 80.0, 20.0, 30.0),
]
MAX_DOSE = 30.0            # ml/L at the top of the SEVERE band
MAX_SPRAY_MS = 2500        # pump on-time that delivers MAX_DOSE (calibrate!)


# --------------------------------------------------------------------------
# Vision: leaf segmentation + damage %
# --------------------------------------------------------------------------
@dataclass
class LeafResult:
    leaf_found: bool
    damage_pct: float
    leaf_frac: float
    dead_crop: bool
    overlay: np.ndarray


def analyse_leaf(bgr: np.ndarray, dead_by_colour: bool = False) -> LeafResult:
    """Segment the leaf from green tissue, fill it, and measure the share of
    yellow/brown pixels inside it (the damage % health index)."""
    h, w = bgr.shape[:2]
    scale = 480.0 / max(h, w)
    small = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr.copy()
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, *HSV_GREEN)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    lesion_colour = cv2.inRange(hsv, *HSV_LESION)
    total = green.size

    # Closing bridges lesion holes so the whole leaf outline becomes one region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    region = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = [c for c in contours if cv2.contourArea(c) > MIN_LEAF_FRAC * total]

    leaf = np.zeros_like(region)
    cv2.drawContours(leaf, keep, -1, 255, thickness=cv2.FILLED)
    leaf_area = cv2.countNonZero(leaf)
    overlay = small.copy()

    if leaf_area == 0:
        # Brown soil looks like a dead leaf, so this is opt-in (--dead-by-colour).
        # Either way nothing is sprayed; only the LCD message differs.
        dead = dead_by_colour and cv2.countNonZero(lesion_colour) / total > DEAD_BROWN_FRAC
        return LeafResult(False, 100.0 if dead else 0.0, 0.0, dead, overlay)

    lesion = cv2.bitwise_and(cv2.bitwise_and(leaf, lesion_colour), cv2.bitwise_not(green))
    damage = 100.0 * cv2.countNonZero(lesion) / leaf_area
    overlay[lesion > 0] = (0, 0, 255)
    cv2.drawContours(overlay, keep, -1, (0, 255, 0), 2)
    return LeafResult(True, damage, leaf_area / total, damage >= TIERS[-1][1], overlay)


# --------------------------------------------------------------------------
# Classifier (VGG16): .tflite (recommended on the Pi) or .keras
# --------------------------------------------------------------------------
class Classifier:
    def __init__(self, model_path: str, classes_path: str):
        self.names = json.loads(Path(classes_path).read_text())
        self.tflite = model_path.endswith(".tflite")
        if self.tflite:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter
            self.interp = Interpreter(model_path=model_path, num_threads=4)
            self.interp.allocate_tensors()
            self.inp = self.interp.get_input_details()[0]
            self.out = self.interp.get_output_details()[0]
        else:
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)

    def predict(self, bgr: np.ndarray):
        x = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        x = (x - VGG_MEAN_BGR)[None, ...]          # same as keras vgg16.preprocess_input
        if self.tflite:
            self.interp.set_tensor(self.inp["index"], x)
            self.interp.invoke()
            probs = self.interp.get_tensor(self.out["index"])[0]
        else:
            probs = self.model.predict(x, verbose=0)[0]
        i = int(np.argmax(probs))
        return self.names[i], float(probs[i])


# --------------------------------------------------------------------------
# Decision: severity tier -> dose -> spray time
# --------------------------------------------------------------------------
@dataclass
class Decision:
    state: str        # HEALTHY | MILD | MODERATE | SEVERE | DEAD | NOLEAF
    dose_ml_l: float
    spray_ms: int
    reason: str


def tier_for(damage: float):
    lo = HEALTHY_MAX
    for name, hi, d_lo, d_hi in TIERS:
        if damage < hi:
            t = min(max((damage - lo) / (hi - lo), 0.0), 1.0)
            return name, d_lo + t * (d_hi - d_lo)
        lo = hi
    return "DEAD", 0.0


def decide(label, conf, leaf: LeafResult, min_conf: float) -> Decision:
    if leaf.dead_crop:
        return Decision("DEAD", 0.0, 0, "dead crop, skipped to avoid waste")
    if not leaf.leaf_found:
        return Decision("NOLEAF", 0.0, 0, "no leaf in frame")

    trusted = label is not None and conf >= min_conf
    healthy_pred = trusted and "healthy" in label.lower()
    diseased_pred = trusted and not healthy_pred
    damage = leaf.damage_pct

    if healthy_pred and damage < HEALTHY_OVERRIDE:
        return Decision("HEALTHY", 0.0, 0, "classifier healthy")
    if diseased_pred:
        damage = max(damage, HEALTHY_MAX)          # diseased => at least mild dose
    if damage < HEALTHY_MAX:
        return Decision("HEALTHY", 0.0, 0, "damage below threshold")

    state, dose = tier_for(damage)
    if state == "DEAD":
        return Decision("DEAD", 0.0, 0, "damage above dead threshold")
    spray_ms = int(round(dose / MAX_DOSE * MAX_SPRAY_MS))
    return Decision(state, dose, spray_ms, f"damage {damage:.1f}%")


# --------------------------------------------------------------------------
# Hardware: camera + ESP32 link
# --------------------------------------------------------------------------
class Camera:
    """Pi Camera (picamera2) -> USB webcam -> folder of images."""

    def __init__(self, source: str):
        self.files, self.cam, self.cap = None, None, None
        if source != "camera":
            self.files = sorted(p for p in Path(source).iterdir()
                                if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if not self.files:
                sys.exit(f"No images found in {source}")
            self.idx = 0
            return
        try:
            from picamera2 import Picamera2
            self.cam = Picamera2()
            self.cam.configure(self.cam.create_still_configuration(
                main={"size": (1024, 768), "format": "RGB888"}))   # numpy order is BGR
            self.cam.start()
            time.sleep(1.0)
        except Exception as e:
            log.warning("picamera2 unavailable (%s), trying USB camera", e)
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                sys.exit("No camera found")

    def grab(self):
        if self.files is not None:
            if self.idx >= len(self.files):
                return None, None
            p = self.files[self.idx]
            self.idx += 1
            return cv2.imread(str(p)), p.name
        if self.cam is not None:
            return self.cam.capture_array(), "frame"
        ok, frame = self.cap.read()
        return (frame if ok else None), "frame"

    def close(self):
        if self.cam is not None:
            self.cam.stop()
        if self.cap is not None:
            self.cap.release()


class ESP32Link:
    def __init__(self, port: str, baud: int, dry_run: bool):
        self.dry = dry_run
        if dry_run:
            return
        import serial
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)                      # ESP32 resets when the port opens
        self.ser.reset_input_buffer()
        self.ser.write(b"PING\n")
        if b"PONG" not in self.ser.readline():
            log.warning("ESP32 did not answer PING - check port/baud/wiring")

    def act(self, state: str, spray_ms: int, timeout: float = 25.0) -> bool:
        cmd = f"ACT,{state},{spray_ms}\n"
        if self.dry:
            log.info("[dry-run] -> ESP32: %s", cmd.strip())
            return True
        self.ser.reset_input_buffer()
        self.ser.write(cmd.encode())
        end = time.time() + timeout
        while time.time() < end:
            if self.ser.readline().strip() == b"DONE":
                return True
        log.error("ESP32 timed out on %s", cmd.strip())
        return False

    def stop(self):
        if not self.dry:
            self.ser.write(b"STOP\n")


class Uploader:
    """Sends each stop to the dashboard backend. If the network is down the stop is
    queued in an outbox file and retried on the next stop, so the robot never waits."""

    def __init__(self, url: str, api_key: str, robot_id: str, outbox: Path):
        self.url, self.key, self.robot_id = url.rstrip("/"), api_key, robot_id
        self.outbox = outbox
        self.pending = []
        if outbox.exists():
            self.pending = [json.loads(l) for l in outbox.read_text().splitlines() if l.strip()]

    def send(self, record: dict, image_path: Path):
        record["robot_id"] = self.robot_id
        self.pending.append({"record": record, "image": str(image_path)})
        self.flush()

    def flush(self):
        import requests
        while self.pending:
            item = self.pending[0]
            img = Path(item["image"])
            fh = open(img, "rb") if img.exists() else None
            try:
                files = {"image": (img.name, fh, "image/jpeg")} if fh else None
                r = requests.post(self.url + "/api/stops", data={"data": json.dumps(item["record"])},
                                  files=files, headers={"X-API-Key": self.key}, timeout=5)
                r.raise_for_status()
            except Exception as e:
                log.warning("dashboard unreachable (%s) - %d stop(s) queued", e, len(self.pending))
                self._save()
                return
            finally:
                if fh:
                    fh.close()
            self.pending.pop(0)
        self._save()

    def _save(self):
        self.outbox.write_text("".join(json.dumps(i) + "\n" for i in self.pending))


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="AgroSpray Pi controller")
    ap.add_argument("--source", default="camera", help="'camera' or a folder of leaf images")
    ap.add_argument("--model", default="models/agrospray_vgg16.tflite")
    ap.add_argument("--classes", default="models/class_names.json")
    ap.add_argument("--no-model", action="store_true", help="skip the CNN, use damage %% only")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dry-run", action="store_true", help="do not talk to the ESP32")
    ap.add_argument("--frames", type=int, default=3, help="frames per stop (voting)")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--dead-by-colour", action="store_true",
                    help="treat a frame with no green but lots of brown as dead crop (soil can fool this)")
    ap.add_argument("--max-stops", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--server", default="", help="dashboard backend URL, e.g. http://192.168.1.10:8000")
    ap.add_argument("--api-key", default="agrospray-dev-key", help="must match AGROSPRAY_KEY on the server")
    ap.add_argument("--robot-id", default="robot-1")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log_dir = Path(args.log_dir)
    (log_dir / "frames").mkdir(parents=True, exist_ok=True)

    clf = None if args.no_model else Classifier(args.model, args.classes)
    cam = Camera(args.source)
    link = ESP32Link(args.port, args.baud, args.dry_run)
    uploader = (Uploader(args.server, args.api_key, args.robot_id, log_dir / "outbox.jsonl")
                if args.server else None)
    n_frames = 1 if args.source != "camera" else max(1, args.frames)

    stats = Counter()
    spray_total_ms = 0
    csv_path = log_dir / "stops.csv"
    new_file = not csv_path.exists()
    fh = open(csv_path, "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["time", "source", "label", "conf", "damage_pct", "state", "dose_ml_l", "spray_ms"])

    try:
        while not args.max_stops or stats["stops"] < args.max_stops:
            frames, name = [], None
            for _ in range(n_frames):
                f, name = cam.grab()
                if f is None:
                    break
                frames.append(f)
            if not frames:
                log.info("No more frames - finished")
                break

            results = [analyse_leaf(f, args.dead_by_colour) for f in frames]
            preds = [clf.predict(f) for f in frames] if clf else []
            leaf = sorted(results, key=lambda r: r.damage_pct)[len(results) // 2]  # median
            label, conf = None, 0.0
            if preds:
                label = Counter(p[0] for p in preds).most_common(1)[0][0]
                conf = float(np.mean([p[1] for p in preds if p[0] == label]))

            d = decide(label, conf, leaf, args.min_conf)
            log.info("%s | %s (%.2f) | damage %.1f%% | %s -> %.1f ml/L, spray %d ms",
                     name, label, conf, leaf.damage_pct, d.state, d.dose_ml_l, d.spray_ms)

            idx = results.index(leaf)
            img_path = log_dir / "frames" / f"{int(time.time())}_{stats['stops']:05d}_{d.state}.jpg"
            cv2.imwrite(str(img_path), results[idx].overlay)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, label, f"{conf:.3f}",
                             f"{leaf.damage_pct:.2f}", d.state, f"{d.dose_ml_l:.1f}", d.spray_ms])
            fh.flush()

            link.act(d.state, d.spray_ms)          # act first, upload after: the robot never waits on Wi-Fi
            if uploader:
                uploader.send({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "source": name, "label": label,
                               "conf": round(conf, 3), "damage_pct": round(leaf.damage_pct, 2),
                               "state": d.state, "dose_ml_l": round(d.dose_ml_l, 1),
                               "spray_ms": d.spray_ms, "max_spray_ms": MAX_SPRAY_MS}, img_path)
            stats["stops"] += 1
            stats["sprayed" if d.spray_ms else "skipped"] += 1
            spray_total_ms += d.spray_ms
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        link.stop()
        cam.close()
        fh.close()
        stops = stats["stops"]
        if stops:
            baseline = stops * MAX_SPRAY_MS
            saving = 100.0 * (1 - spray_total_ms / baseline)
            log.info("Summary: %d stops, %d sprayed, %d skipped, pump %.1f s; "
                     "%.0f%% less than max-dose blanket spraying of every stop",
                     stops, stats["sprayed"], stats["skipped"], spray_total_ms / 1000, saving)


if __name__ == "__main__":
    main()
