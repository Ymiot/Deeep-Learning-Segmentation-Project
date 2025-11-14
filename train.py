import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet
from lib.losses import BCELoss, DiceLoss, FocalLoss, BCELoss_TotalVariation
from lib.metrics import compute_all
from lib.datasets.phc_dataset import PhCDataset
from lib.datasets.retina_dataset import RetinaDataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--model", type=str, choices=["encdec","unet"], default="encdec")
    p.add_argument("--loss", type=str, choices=["bce","dice","focal","bce_weighted","bce_tv"], default="bce")
    p.add_argument("--dataset", type=str, choices=["phc","retina"], default="retina")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def get_loss(name):
    if name == "bce": return BCELoss()
    if name == "dice": return DiceLoss()
    if name == "focal": return FocalLoss()
    if name == "bce_weighted": return BCELoss(weight_pos=3.0)
    if name == "bce_tv": return BCELoss_TotalVariation(tv_weight=1e-4)
    raise ValueError(name)

def load_paths(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

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

    if args.dataset == "phc":
        train_paths = load_paths("splits/phc_train.txt")
        val_paths = load_paths("splits/phc_val.txt")
        test_paths = load_paths("splits/phc_test.txt")
        img_root = os.path.join(cfg["phc_root"], "images")
        mask_root = os.path.join(cfg["phc_root"], "masks")
        train_ds = PhCDataset(train_paths, img_root, mask_root, transform)
        val_ds = PhCDataset(val_paths, img_root, mask_root, transform)
        test_ds = PhCDataset(test_paths, img_root, mask_root, transform)
        is_retina = False
    else:
        train_paths = load_paths("splits/retina_train.txt")
        val_paths = load_paths("splits/retina_val.txt")
        test_paths = load_paths("splits/retina_test.txt")
        train_ds = RetinaDataset(train_paths, cfg["retina_root"], transform)
        val_ds = RetinaDataset(val_paths, cfg["retina_root"], transform)
        test_ds = RetinaDataset(test_paths, cfg["retina_root"], transform)
        is_retina = True

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    if args.model == "encdec":
        model = EncDec(in_channels=3)
    else:
        model = UNet(in_channels=3, out_channels=1, base=cfg["unet_base"], depth=cfg["unet_depth"])

    model = model.to(args.device)
    loss_fn = get_loss(args.loss)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val_dice = 0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
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
            loss = loss_fn(logits, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
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
                    mask_sel = masks[i]
                    pred_sel = probs[i]
                    fov_sel = fov[i] if (is_retina and fov is not None) else None
                    m = compute_all(pred_sel, mask_sel, mask=(fov_sel == 1) if fov_sel is not None else None)
                    val_metrics.append({k: v.item() for k,v in m.items()})

        avg_val = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0].keys()}
        dice_val = avg_val["dice"]
        print(f"Epoch {epoch+1}/{cfg['epochs']} TrainLoss={epoch_loss/len(train_loader):.4f} "
              f"ValDice={dice_val:.4f} ValIoU={avg_val['iou']:.4f}")

        if dice_val > best_val_dice:
            best_val_dice = dice_val
            torch.save(model.state_dict(), f"checkpoints/{args.model}_{args.dataset}_best.pt")
            print("  -> New best model saved.")

    print("Training complete.")

if __name__ == "__main__":
    main()
