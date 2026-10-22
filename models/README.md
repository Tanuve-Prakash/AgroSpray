# models

`python3 train_vgg16.py --data dataset --out models` writes three files here:

- `agrospray_vgg16.keras` - full Keras model
- `agrospray_vgg16.tflite` - small model for the Raspberry Pi
- `class_names.json` - class order used by the model

Copy the `.tflite` and `class_names.json` to the same `models/` folder on the Pi.
Model files are ignored by git (GitHub rejects files over 100 MB); host them on Drive or Releases and link here.
