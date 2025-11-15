import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet
from lib.datasets.phc_dataset import PhCDataset
from lib.datasets.retina_dataset import RetinaDataset
from lib.metrics import compute_all

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--model", type=str, choices=["encdec","unet"], default="encdec")
    p.add_argument("--dataset", type=str, choices=["phc","retina"], default="retina")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/test/")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--show", action="store_true", help="Display a few predictions with matplotlib")
    p.add_argument("--num_show", type=int, default=5, help="Number of images to show")
    return p.parse_args()

def load_paths(path):
    triplets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = [p.strip() for p in line.split(",")]
                triplets.append(parts)
    return triplets

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    size = cfg["image_size"]
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor()
    ])

    os.makedirs(args.out, exist_ok=True)

    # Load dataset
    if args.dataset == "phc":
        test_paths = load_paths("splits/phc_test.txt")
        test_ds = PhCDataset(test_paths, transform=transform)
        is_retina = False
    else:
        test_paths = load_paths("splits/retina_test.txt")
        test_ds = RetinaDataset(test_paths, cfg["retina_root"], transform)
        is_retina = True

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Load model
    if args.model == "encdec":
        model = EncDec()
    else:
        model = UNet(in_channels=3, out_channels=1, base=cfg["unet_base"], depth=cfg["unet_depth"])

    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device)
    model.eval()

    all_metrics = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if is_retina:
                imgs = batch["image"].to(args.device)
                masks = batch["mask"].to(args.device)
                fov = batch["fov"].to(args.device)
            else:
                imgs, masks = batch
                imgs = imgs.to(args.device)
                masks = masks.to(args.device)
                fov = None

            logits = model(imgs)
            probs = torch.sigmoid(logits)

            for i in range(probs.shape[0]):
                pred_sel = probs[i]
                mask_sel = masks[i]

                if fov is not None:
                    fov_sel = fov[i]
                    m = compute_all(pred_sel, mask_sel, mask=(fov_sel > 0))
                else:
                    m = compute_all(pred_sel, mask_sel, mask=None)
                all_metrics.append({k: v.item() for k,v in m.items()})

                # Save prediction
                pred_img = (pred_sel.cpu().numpy() * 255).astype(np.uint8)
                if pred_img.ndim == 2:
                    pred_img = pred_img.squeeze()
                pred_img_pil = Image.fromarray(pred_img)
                img_name = f"pred_{idx*probs.shape[0]+i}.png"
                pred_img_pil.save(os.path.join(args.out, img_name))

                # Optional: show images
                if args.show and idx < args.num_show:
                    img_display = imgs[i].cpu().permute(1,2,0).numpy()
                    mask_display = mask_sel.cpu().numpy().squeeze()
                    pred_display = pred_sel.cpu().numpy().squeeze()
                    fig, axes = plt.subplots(1,3, figsize=(12,4))
                    axes[0].imshow(img_display)
                    axes[0].set_title("Input image")
                    axes[0].axis('off')
                    axes[1].imshow(mask_display, cmap='gray')
                    axes[1].set_title("Ground truth")
                    axes[1].axis('off')
                    axes[2].imshow(pred_display, cmap='gray')
                    axes[2].set_title("Prediction")
                    axes[2].axis('off')
                    plt.show()

    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    print("=== Test set metrics ===")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"Predictions saved to {args.out}")

if __name__ == "__main__":
    main()
