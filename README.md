# Deep Learning Segmentation Project

## Folder layout
```
.
├── train.py            # Training script
├── predict.py          # Inference on test split
├── measure.py          # Compute Dice/IoU/Accuracy/Sensitivity/Specificity
├── scripts/
│   └── create_splits.py  # Create split files from dataset folders (writes filepaths)
├── splits/             # Generated split lists (*.txt with filepaths)
├── configs/
│   └── default.yaml      # Hyperparameters + dataset roots
├── lib/
│   ├── datasets/
│   │   ├── retina_dataset.py  # DRIVE layout support (training/test; 1st_manual; mask as FOV)
│   │   └── phc_dataset.py     # Generic images/masks layout (filepaths-based)
│   ├── model/                 # EncDec (baseline), UNet (optional)
│   ├── losses.py
│   ├── metrics.py
│   └── utils.py (optional)
└── requirements.txt
```

## Quick start
1) Install
```
pip install -r requirements.txt
```

2) Create splits (DRIVE official layout)
```
python scripts/create_splits.py --dataset retina --root /dtu/datasets1/02516/DRIVE --output splits/
# Optional (if you also use PhC):
python scripts/create_splits.py --dataset phc --root /dtu/datasets1/02516/phc_data --output splits/ \
  --images_dir images --masks_dir masks
```

3) Update config
- Set `retina_root: /dtu/datasets1/02516/DRIVE`
- (Optional) set `phc_root` if using PhC

4) Train (baseline EncDec)
```
python train.py --config configs/default.yaml --model encdec --dataset retina --loss bce
```

5) Predict on test split
```
python predict.py --checkpoint checkpoints/encdec_retina_best.pt --dataset retina --model encdec --out outputs/retina/test/
```

6) Measure (for DRIVE include FOV to ignore black border)
```
python measure.py --pred outputs/retina/test/ \
  --gt /dtu/datasets1/02516/DRIVE/test/1st_manual/ \
  --fov /dtu/datasets1/02516/DRIVE/test/mask/
```

Notes
- Splits contain filepaths to images; datasets derive vessel and FOV masks from DRIVE naming rules.
- Baseline simple model (EncDec) is preserved.
- If your PhC folder differs, pass `--images_dir` and `--masks_dir` to the split script.
