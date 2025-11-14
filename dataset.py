# retina_dataset.py / phc_dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class RetinaDataset(Dataset):
    def __init__(self, split_file, transform=None):
        self.transform = transform
        self.samples = []

        with open(split_file, "r") as f:
            for line in f:
                # Supprime les espaces et \n, puis split par ','
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) != 3:
                    raise ValueError(f"Expected 3 paths per line, got {len(parts)}: {line}")
                img_path, mask_path, label_path = parts
                self.samples.append((img_path, mask_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label_path = self.samples[idx]

        # Vérification rapide
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label not found: {label_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        label = Image.open(label_path).convert("L")

        if self.transform:
            img, mask, label = self.transform(img, mask, label)

        return img, mask, label


class PHCDataset(Dataset):
    def __init__(self, split_file, transform=None):
        self.transform = transform
        self.samples = []

        with open(split_file, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) != 2:
                    raise ValueError(f"Expected 2 paths per line (image, label), got {len(parts)}: {line}")
                img_path, label_path = parts
                # PHC a un seul mask = label
                mask_path = label_path
                self.samples.append((img_path, mask_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label_path = self.samples[idx]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label not found: {label_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        label = Image.open(label_path).convert("L")

        if self.transform:
            img, mask, label = self.transform(img, mask, label)

        return img, mask, label
