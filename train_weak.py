import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import json

from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet
from lib.losses import ClickSupervisionLoss
from lib.metrics import compute_all
from lib.datasets.phc_dataset_clicks import PhCDatasetClicks
from lib.datasets.phc_dataset import PhCDataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--model", type=str, choices=["encdec","unet"], default="encdec")
    p.add_argument("--num_pos_clicks", type=int, default=10, help="Number of positive clicks")
    p.add_argument("--num_neg_clicks", type=int, default=10, help="Number of negative clicks")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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

    # -------------------------
    # Dataset with Clicks
    # -------------------------
    train_paths = load_paths("splits/phc_train.txt")
    val_paths = load_paths("splits/phc_val.txt")
    
    # Training dataset uses clicks (weak supervision)
    train_ds = PhCDatasetClicks(
        train_paths, 
        transform=transform, 
        image_size=size,
        num_pos_clicks=args.num_pos_clicks,
        num_neg_clicks=args.num_neg_clicks,
        sample_strategy='random'
    )
    
    # Validation dataset uses full masks for proper evaluation
    val_ds = PhCDataset(val_paths, transform=transform, image_size=size)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # -------------------------
    # Model & Optimizer
    # -------------------------
    if args.model == "encdec":
        model = EncDec()
    else:
        model = UNet(in_channels=3, out_channels=1, base=cfg["unet_base"], depth=cfg["unet_depth"])
    model = model.to(args.device)
    
    # Use click supervision loss for training
    loss_fn = ClickSupervisionLoss()
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val_dice = 0
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots/weak_supervision", exist_ok=True)

    # -------------------------
    # History for curves
    # -------------------------
    train_loss_history = []
    val_loss_history = []
    val_metrics_history = {}

    print(f"Training with weak supervision: {args.num_pos_clicks} positive clicks, {args.num_neg_clicks} negative clicks")
    print(f"Model: {args.model}, Device: {args.device}")
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    for epoch in range(cfg["epochs"]):
        # -------------------------
        # Training with clicks
        # -------------------------
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            imgs, point_masks = batch
            imgs = imgs.to(args.device)
            point_masks = point_masks.to(args.device)

            logits = model(imgs)
            loss = loss_fn(logits, point_masks)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # -------------------------
        # Validation with full masks
        # -------------------------
        model.eval()
        val_metrics = []
        val_epoch_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                imgs, masks = batch
                imgs = imgs.to(args.device)
                masks = masks.to(args.device)

                logits = model(imgs)
                probs = torch.sigmoid(logits)
                
                # Compute metrics on full masks
                for i in range(probs.shape[0]):
                    pred_sel = probs[i]
                    mask_sel = masks[i]
                    m = compute_all(pred_sel, mask_sel, mask=None)
                    val_metrics.append({k: v.item() for k, v in m.items()})

        avg_val = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0].keys()}
        
        # Init metrics history dicts
        if epoch == 0:
            for k in avg_val.keys():
                val_metrics_history[k] = []
        for k in avg_val.keys():
            val_metrics_history[k].append(avg_val[k])

        val_str = " ".join([f"{k}={v:.4f}" for k, v in avg_val.items()])
        print(f"Epoch {epoch+1}/{cfg['epochs']} TrainLoss={avg_train_loss:.4f} VAL: {val_str}")

        # -------------------------
        # Save best model
        # -------------------------
        if avg_val["dice"] > best_val_dice:
            best_val_dice = avg_val["dice"]
            checkpoint_name = f"checkpoints/{args.model}_phc_weak_{args.num_pos_clicks}pos_{args.num_neg_clicks}neg_best.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_val_dice': best_val_dice,
                'train_loss_history': train_loss_history,
                'val_metrics_history': val_metrics_history,
                'config': {
                    'model': args.model,
                    'num_pos_clicks': args.num_pos_clicks,
                    'num_neg_clicks': args.num_neg_clicks,
                    'image_size': size,
                    'batch_size': cfg['batch_size'],
                    'lr': cfg['lr'],
                    'epochs': cfg['epochs']
                }
            }, checkpoint_name)
            print(f"  -> New best model saved. Dice: {best_val_dice:.4f}")

    # -------------------------
    # Save training history to JSON
    # -------------------------
    history_file = f"checkpoints/{args.model}_phc_weak_{args.num_pos_clicks}pos_{args.num_neg_clicks}neg_history.json"
    history_data = {
        'train_loss_history': train_loss_history,
        'val_metrics_history': val_metrics_history,
        'best_val_dice': best_val_dice,
        'config': {
            'model': args.model,
            'num_pos_clicks': args.num_pos_clicks,
            'num_neg_clicks': args.num_neg_clicks,
            'image_size': size,
            'batch_size': cfg['batch_size'],
            'lr': cfg['lr'],
            'epochs': cfg['epochs']
        }
    }
    with open(history_file, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"Training history saved to: {history_file}")

    # -------------------------
    # Plot Loss Curve
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, cfg["epochs"]+1), train_loss_history, label="Train Loss (Click Supervision)", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Weak Supervision Training Loss ({args.num_pos_clicks}+{args.num_neg_clicks} clicks)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plot_name = f"plots/weak_supervision/loss_{args.model}_{args.num_pos_clicks}pos_{args.num_neg_clicks}neg.png"
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to: {plot_name}")

    # -------------------------
    # Plot Metrics Curves
    # -------------------------
    for metric in val_metrics_history.keys():
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, cfg["epochs"]+1), val_metrics_history[metric], label=f"Val {metric}", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f"Validation {metric.capitalize()} (Weak Supervision)", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plot_name = f"plots/weak_supervision/{metric}_{args.model}_{args.num_pos_clicks}pos_{args.num_neg_clicks}neg.png"
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"Metric plots saved to: plots/weak_supervision/")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Model saved to: checkpoints/{args.model}_phc_weak_{args.num_pos_clicks}pos_{args.num_neg_clicks}neg_best.pt")
    print(f"History saved to: {history_file}")
    print("="*60)

if __name__ == "__main__":
    main()
