# AgroSpray

AI plant disease detection and targeted pesticide-spraying robot (Team Omega, Nexus SIT Hackathon).

Closed loop: **capture -> segment leaf -> damage % -> VGG16 -> decide -> spray -> move -> report to dashboard**

```
  Pi Camera --> Raspberry Pi (OpenCV + VGG16 TFLite) --serial--> ESP32 --> L298N motors, pump, LEDs, LCD
                      |
                      +--- Wi-Fi (queued offline) ---> Flask backend (SQLite) ---> Farmer dashboard (browser)
```

| Path | Runs on | Job |
|---|---|---|
| `train_vgg16.py` | laptop / Colab | Train VGG16 on 6 classes; export `.keras`, `.tflite`, `class_names.json` |
| `pi_agrospray.py` | Raspberry Pi | Camera, leaf segmentation, damage %, classification, severity decision, ESP32 commands, CSV log, upload to dashboard |
| `esp32_controller/esp32_controller.ino` | ESP32 | L298N drive, pump, green/red LEDs, 16x2 I2C LCD |
| `backend/app.py` | laptop / Pi / cloud VM | REST API + SQLite storage + serves the dashboard |
| `dashboard/index.html` | any browser | Live farmer dashboard: counts, pesticide saving, status bars, damage trend, crop table, recent leaf photos, CSV export |
| `test_leaves/` | - | 6 synthetic leaf images for a quick no-hardware test |

## Quick try (no hardware, no model)

```
pip install -r requirements.txt            # numpy, opencv, pyserial, requests
pip install -r backend/requirements.txt    # flask
export AGROSPRAY_KEY=mysecret              # Windows PowerShell: $env:AGROSPRAY_KEY="mysecret"
python3 backend/app.py                     # dashboard at http://localhost:8000
```
In a second terminal:
```
python3 pi_agrospray.py --source test_leaves --no-model --dry-run --server http://localhost:8000 --api-key mysecret
```
Open http://localhost:8000 and the 6 stops appear.

## 1. Train
Put images in `dataset/<class_name>/` (see `dataset/README.md`), then:
```
python3 train_vgg16.py --data dataset --out models
```
Copy `models/agrospray_vgg16.tflite` and `models/class_names.json` to the Pi.

## 2. Flash the ESP32
Arduino IDE -> install "LiquidCrystal I2C" (Frank de Brabander) -> board "ESP32 Dev Module" -> upload.

| Part | ESP32 pin |
|---|---|
| L298N ENA / IN1 / IN2 | 25 / 26 / 27 |
| L298N ENB / IN3 / IN4 | 17 / 14 / 16 |
| Pump MOSFET or relay | 23 |
| Green LED / Red LED (with 220 ohm) | 32 / 18 |
| LCD SDA / SCL | 21 / 22 |
| Pi link | ESP32 USB cable to a Pi USB port |

Common ground everywhere. Power the pump and motors from the battery side, not the ESP32 5V pin; add a flyback diode across the pump if you drive it with a bare MOSFET.
Serial test from the Arduino serial monitor: `ACT,SEVERE,1000`.

## 3. Run the robot
```
sudo apt install -y python3-picamera2
pip install -r requirements.txt
python3 pi_agrospray.py --model models/agrospray_vgg16.tflite --classes models/class_names.json \
    --port /dev/ttyUSB0 --server http://<dashboard-ip>:8000 --api-key <same secret>
```
The robot acts first and uploads after, so poor Wi-Fi never delays spraying. If the dashboard is unreachable, stops are queued in `logs/outbox.jsonl` and sent on a later stop.

## 4. Dashboard server
```
export AGROSPRAY_KEY=<secret>
python3 backend/app.py          # PORT=8000 by default
```
API: `POST /api/stops` (needs `X-API-Key`), `GET /api/stops?limit=50`, `GET /api/summary`, `GET /api/export.csv`.
For use outside your own network, put it behind HTTPS (for example a reverse proxy) and use a strong key; the built-in Flask server is for demos and small deployments.

## Calibrate before trusting it
- **Segmentation thresholds** (`HSV_GREEN`, `HSV_LESION` in `pi_agrospray.py`) are generic. Tune on betel, Malabar spinach and potato leaves in your field lighting. Brown soil can look like lesion, so keep the leaf large in the frame.
- **Severity bands** (`TIERS`) use your 5-10 / 10-20 / 20-30 ml/L bands; the damage-% cut-offs (5/20/45/80) are starting guesses.
- **Dose -> spray time:** one pump varies volume, not concentration, so `MAX_SPRAY_MS` maps the top dose to pump time. Measure your nozzle's ml/s and set it.
- `MOVE_MS` and `DRIVE_PWM` in the sketch depend on your chassis.
- Follow the pesticide label and wear protection; test with plain water first.

## Testing status
Tested as software: vision and decision logic on synthetic leaves, the Flask API, the dashboard JavaScript syntax, and the robot-to-backend upload including the offline queue. Not yet run on real hardware, with a trained model, or in a browser by me; the first run on your setup is the real test.
