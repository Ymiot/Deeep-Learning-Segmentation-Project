# Deep Learning Segmentation Project

This project implements segmentation models (simple encoder–decoder CNN, U-Net, and optionally a dilated network) on two datasets located at `/dtu/datasets1/02516`.

## 1. Datasets

### 1.1. PhC Cell Segmentation (e.g., PhC-U373)
- Typical contents: phase contrast microscopy images (grayscale or RGB) + binary masks (cell vs background).
- Approximate size (variant dependent): 100–400 images.
- No predefined train/validation/test split.
- Class imbalance may occur (background dominates).
- Action: create a reproducible split (e.g. 70% train / 15% validation / 15% test).

### 1.2. Retinal Vessel Segmentation (e.g., DRIVE/STARE style)
- Contents: high-resolution color retina images + two types of masks:
  1. Vessel segmentation mask (this is the one to use for training).
  2. Field-of-view (FOV) mask separating valid circular image regions from black borders.
- Typical DRIVE setup: 40 images (20 train official / 20 test official). If official split exists:
  - Option A: Use official train set; sub-split into train/validation. Use the official test set only at the end.
  - Option B: If structure differs, re-split entire set (e.g. 60/20/20).
- Important: Exclude non-FOV pixels when computing metrics (mask them out).

### Recommended Split Strategy
- Fix a random seed (e.g. 42).
- Use `train_test_split` or custom index shuffling.
- Save split indices into `splits/` for reproducibility (`phc_train.txt`, etc.).

## 2. Goals

1. Implement custom PyTorch `Dataset` and `DataLoader` for both datasets.
2. Implement a simple encoder–decoder segmentation network.
3. Implement and train a U-Net (adapt image size carefully to balance resolution vs GPU memory).
4. (Optional) Implement a DilatedNet for comparison.
5. Compute metrics: Dice, Intersection over Union (IoU/Jaccard), Accuracy, Sensitivity (Recall), Specificity.
6. Perform an ablation over loss functions:
   - Binary Cross Entropy (BCE)
   - Dice Loss
   - Focal Loss
   - Weighted BCE (positive class weight)
   - (Optional) BCE + Total Variation regularization
7. Report performance on both datasets (train/validation/test splits).

## 3. Suggested Directory Layout

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
│   ├── phc_train.txt
│   ├── phc_val.txt
│   ├── phc_test.txt
│   ├── retina_train.txt
│   ├── retina_val.txt
│   ├── retina_test.txt
├── lib/
│   ├── datasets/
│   │   ├── phc_dataset.py
│   │   └── retina_dataset.py
│   ├── model/
│   │   ├── EncDecModel.py
│   │   ├── UNetModel.py
│   │   ├── DilatedNetModel.py
│   ├── losses.py
│   ├── metrics.py
│   └── utils.py
└── scripts/
    ├── create_splits.py
    └── visualize_batch.py
```

## 4. Image Resolution & GPU Memory

- Suggested default size: `256x256` (or `512x512` if GPU has enough memory).
- Large retina images may require downscaling or patch extraction.
- Adjust `batch_size` depending on VRAM (e.g., 4–8 for large inputs).

## 5. Models

### 5.1. Simple Encoder–Decoder
- Down path: Conv blocks + MaxPool.
- Up path: Upsampling (bilinear or transposed conv).
- Output: Single-channel logits → Sigmoid during loss.

### 5.2. U-Net
- Multiple downsampling stages (4–5).
- Skip connections from encoder to decoder.
- Each block: two Conv(3×3) + BatchNorm + ReLU.
- Upsampling: ConvTranspose2d or UpSample + Conv.

### 5.3. DilatedNet (Optional)
- Use dilated convolutions to enlarge receptive field without pooling.

## 6. Loss Functions

| Loss | Advantages | Disadvantages |
|------|------------|---------------|
| BCE | Stable, standard | Sensitive to imbalance |
| Dice | Direct overlap optimization | Can be unstable for tiny objects |
| Focal | Handles class imbalance | Requires tuning γ, α |
| Weighted BCE | Simple weighting | Selecting weights is heuristic |
| BCE + TV | Smoother boundaries | Extra hyperparameter; runtime cost |

## 7. Metrics

Using pixel-level counts: TP, FP, TN, FN after thresholding (e.g. `pred > 0.5`).

- Dice: `2TP / (2TP + FP + FN)` (good overlap indicator; forgiving for small false positives).
- IoU: `TP / (TP + FP + FN)` (stricter than Dice).
- Accuracy: `(TP + TN) / total` (misleading under severe imbalance).
- Sensitivity (Recall): `TP / (TP + FN)` (missed positives cost).
- Specificity: `TN / (TN + FP)` (false alarm control).

For retina, apply metrics only inside FOV.

## 8. Training Flow

1. Load config (image size, batch size, learning rate).
2. Build datasets using stored index splits.
3. Train with chosen loss; monitor validation Dice/IoU.
4. Apply early stopping or save best checkpoint.
5. Test: write predicted masks under `outputs/{dataset}/test/`.
6. Measure metrics with `measure.py`.

## 9. Ablation Study

Pseudo loop:
```
for loss in ["bce","dice","focal","bce_weighted"]:
    init model
    train epochs
    evaluate on validation
    log metrics
```
Interpret which loss handles imbalance (often Focal or Dice outperform BCE on thin vessels).

## 10. Reproducibility

- Fix seeds (`torch`, `numpy`, `random`).
- Save split files once.
- Log environment (CUDA version, GPU model).
- Store metrics in CSV and/or TensorBoard.

## 11. Execution Examples

```
# Generate splits
python scripts/create_splits.py --dataset phc --root /dtu/datasets1/02516/PhC --output splits/
python scripts/create_splits.py --dataset retina --root /dtu/datasets1/02516/Retina --output splits/

# Train U-Net with Dice loss
python train.py --config configs/default.yaml --model unet --loss dice --dataset phc

# Predict on test split
python predict.py --checkpoint checkpoints/unet_phc_best.pt --dataset phc --model unet --out outputs/phc/test/

# Measure performance
python measure.py --pred outputs/phc/test/ --gt /dtu/datasets1/02516/PhC/masks/

# Retina (with FOV)
python measure.py --pred outputs/retina/test/ --gt /dtu/datasets1/02516/Retina/vessels/ --fov /dtu/datasets1/02516/Retina/fov/
```

## 12. Important Notes

- Ensure output normalization consistency (use BCEWithLogitsLoss → raw logits; apply Sigmoid only for metrics).
- For retina: zero out predictions outside FOV before metric computation or ignore them via masking.
- Keep batch sizes low for high-res retina images to prevent GPU memory exhaustion.

## 13. Licensing & Data Usage

Respect each dataset’s original license (e.g., DRIVE). Do not redistribute raw data in this repo—reference official sources.

---

Adapt as needed if the actual dataset folder structure differs. Let me know if you want TensorBoard integration, patch-based inference, or mixed precision training added.