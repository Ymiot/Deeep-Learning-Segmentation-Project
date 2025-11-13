import argparse
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from lib.datasets.phc_dataset import PhCDataset
from lib.datasets.retina_dataset import RetinaDataset
import torchvision.transforms as transforms
import yaml

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", choices=["phc","retina"], default="phc")
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor()
    ])
    if args.dataset == "phc":
        idxs = [int(x.strip()) for x in open("splits/phc_train.txt").readlines()][:8]
        ds = PhCDataset(cfg["phc_root"], idxs, transform)
        imgs = [ds[i][0] for i in range(len(ds))]
    else:
        idxs = [int(x.strip()) for x in open("splits/retina_train.txt").readlines()][:8]
        ds = RetinaDataset(cfg["retina_root"], idxs, transform)
        imgs = [ds[i]["image"] for i in range(len(ds))]
    grid = make_grid(torch.stack(imgs), nrow=4)
    plt.figure(figsize=(8,8))
    plt.imshow(grid.permute(1,2,0))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()