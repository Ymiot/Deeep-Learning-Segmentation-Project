import os, argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from lib.model.UNetModel import UNet
from lib.model.EncDecModel import EncDec
from lib.datasets.phc_dataset import PhCDataset
from lib.datasets.retina_dataset import RetinaDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

def save_mask(array, path):
    im_arr = (array*255).astype(np.uint8)
    Image.fromarray(im_arr).save(path)

def load_paths(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, choices=["encdec","unet"], default="encdec")
    p.add_argument("--dataset", type=str, choices=["phc","retina"], default="retina")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    size = cfg["image_size"]
    transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])

    if args.dataset == "phc":
        test_paths = load_paths("splits/phc_test.txt")
        test_ds = PhCDataset(test_paths, transform=transform)
        is_retina = False
    else:
        test_paths = load_paths("splits/retina_test.txt")
        test_ds = RetinaDataset(test_paths, cfg["retina_root"], transform)
        is_retina = True

    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    if args.model == "encdec":
        model = EncDec(in_channels=3)
    else:
        model = UNet(in_channels=3, out_channels=1, base=cfg["unet_base"], depth=cfg["unet_depth"])
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device)
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            if is_retina:
                img = batch["image"].to(args.device)
            else:
                img, _ = batch
                img = img.to(args.device)
            logits = model(img)
            probs = torch.sigmoid(logits)[0,0].cpu().numpy()
            mask = (probs > 0.5).astype(np.uint8)
            save_mask(mask, os.path.join(args.out, f"mask_{i:04d}.png"))

    print("Prediction finished.")

if __name__ == "__main__":
    main()
