# Deep Learning Segmentation Project

This repository provides a framework for semantic image segmentation using deep learning.  
It is designed for experiments on two biomedical datasets (PhC cell segmentation and retinal vessel segmentation) located at `/dtu/datasets1/02516`.

## Directory Structure

```
.
├── train.py            # Model training script
├── predict.py          # Inference/prediction script
├── measure.py          # Compute segmentation metrics
├── scripts/
│   └── create_splits.py  # Script to create train/val/test splits
├── splits/             # Folder with split indices (generated)
├── configs/
│   └── default.yaml      # Main config file (paths, hyperparams)
├── lib/
│   ├── model/            # Model architectures (EncDec, UNet, ...)
│   ├── losses.py         # Loss functions
│   ├── metrics.py        # Metric functions
│   └── datasets/         # Custom PyTorch Dataset classes
└── requirements.txt      # Python dependencies
```

## Quick Start

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Create dataset splits**  
   (*Run once for each dataset. Adjust `--root` as needed.*)
   ```
   python scripts/create_splits.py --dataset phc --root /dtu/datasets1/02516/PhC --output splits/
   python scripts/create_splits.py --dataset retina --root /dtu/datasets1/02516/Retina --output splits/
   ```

3. **Edit `configs/default.yaml`**  
   - Update dataset root paths if necessary.

4. **Train a model**
   ```
   python train.py --config configs/default.yaml --model encdec --dataset phc --loss bce
   ```

5. **Predict segmentations on test set**
   ```
   python predict.py --checkpoint checkpoints/encdec_phc_best.pt --dataset phc --model encdec --out outputs/phc/test/
   ```

6. **Evaluate metrics**
   ```
   python measure.py --pred outputs/phc/test/ --gt /dtu/datasets1/02516/PhC/masks/
   ```

## Notes

- The default model is a simple encoder–decoder CNN based on the project's requirements.
- Dataset-specific details (preprocessing, mask handling, FOV for retina) are handled by the relevant dataloader.
- All random splits are reproducible via a fixed seed in the config.
- Add further models, losses, or augmentations as needed using the modular structure.

For any questions about extending or running the code, please refer to comments in the scripts or contact the project maintainers.
