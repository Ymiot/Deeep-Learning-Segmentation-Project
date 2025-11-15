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
    train_metrics_history = {}
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
        # Compute training metrics (for tracking)
        # -------------------------
        model.eval()
        train_metrics = []
        with torch.no_grad():
            for batch in train_loader:
                imgs, point_masks = batch
                imgs = imgs.to(args.device)
                point_masks = point_masks.to(args.device)
                
                logits = model(imgs)
                probs = torch.sigmoid(logits)
                
                # Compute metrics only on clicked pixels for training
                for i in range(probs.shape[0]):
                    pred_sel = probs[i]
                    mask_sel = point_masks[i]
                    # Convert point_mask to binary (0 and 1 only) for metric computation
                    valid_mask = (mask_sel != -1)
                    if valid_mask.sum() > 0:
                        m = compute_all(pred_sel, mask_sel.clamp(0, 1), mask=valid_mask.squeeze())
                        train_metrics.append({k: v.item() for k, v in m.items()})
        
        avg_train = {k: np.mean([m[k] for m in train_metrics]) for k in train_metrics[0].keys()}

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
                train_metrics_history[k] = []
                val_metrics_history[k] = []
        for k in avg_val.keys():
            train_metrics_history[k].append(avg_train[k])
            val_metrics_history[k].append(avg_val[k])

        # Regular epoch logging
        val_str = " ".join([f"{k}={v:.4f}" for k, v in avg_val.items()])
        print(f"Epoch {epoch+1}/{cfg['epochs']} TrainLoss={avg_train_loss:.4f} VAL: {val_str}")
        
        # Detailed progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\n{'='*70}")
            print(f"Progress Summary - Epoch {epoch+1}/{cfg['epochs']}")
            print(f"{'='*70}")
            print(f"Training Loss:   {avg_train_loss:.4f}")
            print(f"\nTraining Metrics (on clicks):")
            for k, v in avg_train.items():
                print(f"  {k.capitalize():15s}: {v:.4f}")
            print(f"\nValidation Metrics (on full masks):")
            for k, v in avg_val.items():
                print(f"  {k.capitalize():15s}: {v:.4f}")
            print(f"\nBest Val Dice so far: {best_val_dice:.4f}")
            print(f"{'='*70}\n")

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
                'train_metrics_history': train_metrics_history,
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
        'train_metrics_history': train_metrics_history,
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
    # Plot Combined Loss Curve (Train only - no val loss computed)
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, cfg["epochs"]+1), train_loss_history, label="Train Loss", linewidth=2, color='blue')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Training Loss - Weak Supervision ({args.num_pos_clicks}+{args.num_neg_clicks} clicks)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plot_name = f"plots/weak_supervision/loss_{args.model}_{args.num_pos_clicks}pos_{args.num_neg_clicks}neg.png"
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to: {plot_name}")

    # -------------------------
    # Plot Combined Metrics (Train vs Val)
    # -------------------------
    for metric in val_metrics_history.keys():
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, cfg["epochs"]+1), train_metrics_history[metric], 
                label=f"Train {metric}", linewidth=2, linestyle='--', alpha=0.7)
        plt.plot(range(1, cfg["epochs"]+1), val_metrics_history[metric], 
                label=f"Val {metric}", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f"{metric.capitalize()} - Train vs Val (Weak Supervision)", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plot_name = f"plots/weak_supervision/{metric}_{args.model}_{args.num_pos_clicks}pos_{args.num_neg_clicks}neg.png"
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"Metric plots saved to: plots/weak_supervision/")
    
    # -------------------------
    # Plot Combined Overview (Loss + Accuracy)
    # -------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss subplot
    ax1.plot(range(1, cfg["epochs"]+1), train_loss_history, label="Train Loss", linewidth=2, color='blue')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy subplot
    ax2.plot(range(1, cfg["epochs"]+1), train_metrics_history['accuracy'], 
            label="Train Accuracy", linewidth=2, linestyle='--', alpha=0.7)
    ax2.plot(range(1, cfg["epochs"]+1), val_metrics_history['accuracy'], 
            label="Val Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy - Train vs Val", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Training Overview - {args.model.upper()} ({args.num_pos_clicks}+{args.num_neg_clicks} clicks)", 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plot_name = f"plots/weak_supervision/overview_{args.model}_{args.num_pos_clicks}pos_{args.num_neg_clicks}neg.png"
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Overview plot saved to: {plot_name}")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Model saved to: checkpoints/{args.model}_phc_weak_{args.num_pos_clicks}pos_{args.num_neg_clicks}neg_best.pt")
    print(f"History saved to: {history_file}")
    print("="*60)

if __name__ == "__main__":
    main()
