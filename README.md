# Deep Learning Segmentation Project

Minimal instructions for collaborators.

## Dataset Assumptions

Both datasets already have train/test folders:
- DRIVE: /DRIVE/training/... and /DRIVE/test/...
- PhC: /phc_data/train/images/*.jpg and /phc_data/train/labels/, plus /phc_data/test/images /phc_data/test/labels

We build splits as:
- From `train` folder: 80% → training, 20% → validation
- From `test` folder: random 20% subset → test split (to keep evaluation light)

Split files contain absolute image paths (one per line).

## Structure
```
.
├── train.py
├── predict.py
├── measure.py
├── scripts/
│   └── create_splits.py
├── splits/
├── configs/
│   └── default.yaml
├── lib/
│   ├── datasets/
│   │   ├── retina_dataset.py
│   │   └── phc_dataset.py
│   ├── model/
│   │   └── EncDecModel.py
│   ├── losses.py
│   ├── metrics.py
└── requirements.txt
```

## Create Splits

```
python scripts/create_splits.py --dataset retina --root /dtu/datasets1/02516/DRIVE --output splits/
python scripts/create_splits.py --dataset phc --root /dtu/datasets1/02516/phc_data --output splits/
```

## Train (baseline EncDec)

```
python train.py --config configs/default.yaml --model encdec --dataset retina --loss bce
python train.py --config configs/default.yaml --model encdec --dataset phc --loss bce
```

## Predict

```
python predict.py --config configs/default.yaml --model encdec --dataset retina --checkpoint checkpoints/encdec_retina_best.pt --out outputs/retina/test/
python predict.py --config configs/default.yaml --model encdec --dataset phc --checkpoint checkpoints/encdec_phc_best.pt --out outputs/phc/test/
```
To visualize : 
python predict.py --config configs/default.yaml --model encdec --dataset retina --checkpoint checkpoints/encdec_retina_best.pt --out outputs/retina/test/ --show --num_show 5
python predict.py --config configs/default.yaml --model encdec --dataset phc --checkpoint checkpoints/encdec_phc_best.pt --out outputs/phc/test/ --show --num_show 5

## Measure

DRIVE (use FOV):
```
python measure.py --pred outputs/retina/test/ \
  --gt /dtu/datasets1/02516/DRIVE/test/1st_manual/ \
  --fov /dtu/datasets1/02516/DRIVE/test/mask/
```

PhC:
```
python measure.py --pred outputs/phc/test/ \
  --gt /dtu/datasets1/02516/phc_data/test/labels/
```

## Notes

- Test subset is a random 20% of official test images (reproducible via seed).
- Labels for PhC are derived by replacing `/images/` with `/labels/` and keeping filename.
- Keep the EncDec model as baseline for loss ablation.
