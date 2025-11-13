Voici le code Markdown complet pour le fichier `README.md` en anglais, centré sur la présentation et les commandes.

```markdown
# Deep Learning Segmentation Project

This project implements segmentation models (Encoder-Decoder CNN, U-Net) on two datasets located at `/dtu/datasets1/02516`.

## 1. Datasets

### 1.1. PhC Cell Segmentation (e.g., PhC-U373)
* **Contents**: Phase contrast microscopy images (grayscale or RGB) + binary masks (cell vs background).
* **Approximate size** (variant dependent): 100–400 images.
* No predefined train/validation/test split (a split must be created).

### 1.2. Retinal Vessel Segmentation (e.g., DRIVE/STARE)
* **Contents**: High-resolution color retina images + two types of masks:
    1.  Vessel segmentation mask (this is the one to use for training).
    2.  Field-of-view (FOV) mask separating valid circular image regions from black borders.
* **Typical DRIVE setup**: 40 images (20 train official / 20 test official).
* **Important**: Non-FOV pixels must be excluded when computing metrics.

---

## 2. Suggested Directory Layout

```

.
├── README.md
├── requirements.txt
├── Exercises.pdf
├── train.py
├── predict.py
├── measure.py
├── configs/
│   └── default.yaml
├── splits/
│   ├── phc\_train.txt
│   ├── phc\_val.txt
│   ├── phc\_test.txt
│   ├── retina\_train.txt
│   ├── retina\_val.txt
│   ├── retina\_test.txt
├── lib/
│   ├── datasets/
│   │   ├── phc\_dataset.py
│   │   └── retina\_dataset.py
│   ├── model/
│   │   ├── EncDecModel.py
│   │   ├── UNetModel.py
│   │   ├── DilatedNetModel.py
│   ├── losses.py
│   ├── metrics.py
│   └── utils.py
└── scripts/
├── create\_splits.py
└── visualize\_batch.py

````

---

## 3. Execution Examples

```bash
# Generate splits for the datasets
python scripts/create_splits.py --dataset phc --root /dtu/datasets1/02516/PhC --output splits/
python scripts/create_splits.py --dataset retina --root /dtu/datasets1/02516/Retina --output splits/

# Train U-Net with Dice loss on PhC
python train.py --config configs/default.yaml --model unet --loss dice --dataset phc

# Predict on the PhC test split
python predict.py --checkpoint checkpoints/unet_phc_best.pt --dataset phc --model unet --out outputs/phc/test/

# Measure performance on PhC
python measure.py --pred outputs/phc/test/ --gt /dtu/datasets1/02516/PhC/masks/

# Measure performance on Retina (using the FOV mask)
python measure.py --pred outputs/retina/test/ --gt /dtu/datasets1/02516/Retina/vessels/ --fov /dtu/datasets1/02516/Retina/fov/
````

```
```
