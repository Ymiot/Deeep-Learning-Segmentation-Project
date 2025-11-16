import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

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

    # -------------------------
    # Dataset
    # -------------------------
    if args.dataset == "phc":
        train_paths = load_paths("splits/phc_train.txt")
        val_paths = load_paths("splits/phc_val.txt")
        test_paths = load_paths("splits/phc_test.txt")
        train_ds = PhCDataset(train_paths, transform=transform)
        val_ds = PhCDataset(val_paths, transform=transform)
        test_ds = PhCDataset(test_paths, transform=transform)
        is_retina = False
    else:
        train_paths = load_paths("splits/retina_train.txt")
        val_paths = load_paths("splits/retina_val.txt")
        test_paths = load_paths("splits/retina_test.txt")
        # train_ds = RetinaDataset(train_paths, cfg["retina_root"], transform)
        # val_ds = RetinaDataset(val_paths, cfg["retina_root"], transform)
        # test_ds = RetinaDataset(test_paths, cfg["retina_root"], transform)
        train_ds = RetinaDataset(train_paths, cfg["retina_root"], transform, image_size=cfg["image_size"])
        val_ds = RetinaDataset(val_paths, cfg["retina_root"], transform, image_size=cfg["image_size"])
        test_ds = RetinaDataset(test_paths, cfg["retina_root"], transform, image_size=cfg["image_size"])
        is_retina = True

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

    # -------------------------
    # Model & Optimizer
    # -------------------------
    if args.model == "encdec":
        model = EncDec()
    else:
        model = UNet(in_channels=3, out_channels=1, base=cfg["unet_base"], depth=cfg["unet_depth"])
    model = model.to(args.device)
    loss_fn = get_loss(args.loss)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val_dice = 0
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # -------------------------
    # History for curves
    # -------------------------
    train_loss_history = []
    val_loss_history = []
    train_metrics_history = {}
    val_metrics_history = {}

    for epoch in range(cfg["epochs"]):
        # -------------------------
        # Training
        # -------------------------
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

        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # -------------------------
        # Metrics on TRAIN
        # -------------------------
        model.eval()
        train_metrics = []
        with torch.no_grad():
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
                probs = torch.sigmoid(logits)
                for i in range(probs.shape[0]):
                    pred_sel = probs[i]
                    mask_sel = masks[i]
                    if fov is not None:
                        fov_sel = fov[i]
                        m = compute_all(pred_sel, mask_sel, mask=(fov_sel > 0))
                    else:
                        m = compute_all(pred_sel, mask_sel, mask=None)
                    train_metrics.append({k: v.item() for k, v in m.items()})

        avg_train = {k: np.mean([m[k] for m in train_metrics]) for k in train_metrics[0].keys()}
        # Init metrics history dicts
        if epoch == 0:
            for k in avg_train.keys():
                train_metrics_history[k] = []
                val_metrics_history[k] = []
        for k in avg_train.keys():
            train_metrics_history[k].append(avg_train[k])

        # -------------------------
        # Metrics on VAL
        # -------------------------
        val_metrics = []
        val_epoch_loss = 0
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
                loss = loss_fn(logits, masks)
                val_epoch_loss += loss.item()
                probs = torch.sigmoid(logits)
                for i in range(probs.shape[0]):
                    pred_sel = probs[i]
                    mask_sel = masks[i]
                    if fov is not None:
                        fov_sel = fov[i]
                        m = compute_all(pred_sel, mask_sel, mask=(fov_sel > 0))
                    else:
                        m = compute_all(pred_sel, mask_sel, mask=None)
                    val_metrics.append({k: v.item() for k, v in m.items()})

        avg_val = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0].keys()}
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        for k in avg_val.keys():
            val_metrics_history[k].append(avg_val[k])

        train_str = " ".join([f"{k}={v:.4f}" for k, v in avg_train.items()])
        val_str = " ".join([f"{k}={v:.4f}" for k, v in avg_val.items()])
        print(f"Epoch {epoch+1}/{cfg['epochs']} TrainLoss={avg_train_loss:.4f} ValLoss={avg_val_loss:.4f} "
              f"TRAIN: {train_str} VAL: {val_str}")

        # -------------------------
        # Save best model
        # -------------------------
        if avg_val["dice"] > best_val_dice:
            best_val_dice = avg_val["dice"]
            torch.save(model.state_dict(), f"checkpoints/{args.model}_{args.dataset}_best.pt")
            print("  -> New best model saved.")

    # -------------------------
    # Plot Loss Curves
    # -------------------------
    plt.figure()
    plt.plot(range(1, cfg["epochs"]+1), train_loss_history, label="Train Loss")
    plt.plot(range(1, cfg["epochs"]+1), val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/loss_curves.png")
    plt.show()

    # -------------------------
    # Plot Metrics Curves
    # -------------------------
    for metric in train_metrics_history.keys():
        plt.figure()
        plt.plot(range(1, cfg["epochs"]+1), train_metrics_history[metric], label=f"Train {metric}")
        plt.plot(range(1, cfg["epochs"]+1), val_metrics_history[metric], label=f"Val {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"Train vs Val {metric}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/{metric}_curves.png")
        plt.show()

    print("Training complete.")

if __name__ == "__main__":
    main()
