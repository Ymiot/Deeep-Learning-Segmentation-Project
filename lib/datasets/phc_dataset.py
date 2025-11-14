import torch
from torch.utils.data import Dataset
from PIL import Image

class PhCDataset(Dataset):
    def __init__(self, lines, transform=None):
        self.samples = []
        for line in lines:
            paths = line.strip().split(",")
            if len(paths) != 2:
                raise ValueError(f"Expected 2 paths per line, got {paths}")
            img_path, mask_path = paths
            self.samples.append((img_path, mask_path))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
