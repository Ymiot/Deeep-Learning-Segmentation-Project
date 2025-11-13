import os, argparse
import numpy as np
from PIL import Image
from lib.metrics import compute_all

def load_mask(path):
    return (np.array(Image.open(path)) > 127).astype(np.uint8)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=str, required=True, help="Directory containing predicted masks")
    p.add_argument("--gt", type=str, required=True, help="Directory containing ground truth masks")
    p.add_argument("--fov", type=str, default=None, help="Directory containing FOV masks (retina only)")
    return p.parse_args()

def main():
    args = parse_args()
    pred_files = sorted([f for f in os.listdir(args.pred) if f.endswith(".png")])
    metrics_all = []
    for f in pred_files:
        pred = load_mask(os.path.join(args.pred, f))
        gt = load_mask(os.path.join(args.gt, f))
        mask = None
        if args.fov:
            fov = load_mask(os.path.join(args.fov, f))
            mask = (fov == 1)
        import torch
        pred_t = torch.tensor(pred[None, ...], dtype=torch.float32)
        gt_t = torch.tensor(gt[None, ...], dtype=torch.float32)
        m = compute_all(pred_t, gt_t, mask=mask)
        metrics_all.append({k: float(v) for k,v in m.items()})
    summary = {k: np.mean([x[k] for x in metrics_all]) for k in metrics_all[0].keys()}
    print("Average metrics:")
    for k,v in summary.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()