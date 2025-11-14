import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class RetinaDataset(Dataset):
    """Dataset pour DRIVE/retina segmentation."""
    def __init__(self, file_list, root=None, transform=None):
        """
        file_list: liste de lignes de type:
            img_path, mask_path, label_path
        root: racine du dataset (optionnel)
        """
        self.samples = []
        for line in file_list:
            if len(line) != 3:
                raise ValueError(f"Expected 3 paths per line, got {len(line)}: {line}")
            if root:
                line = [os.path.join(root, os.path.relpath(p, start=root)) for p in line]
            self.samples.append(line)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        fov = Image.open(label_path).convert("L")  # DRIVE uses mask also as FOV

        if self.transform:
            img = self.transform(img)
            mask = transforms.Resize((128, 128))(mask)
            fov = transforms.Resize((128, 128))(fov)
            mask = transforms.ToTensor()(mask)
            fov = transforms.ToTensor()(fov)

        return {"image": img, "mask": mask, "fov": fov}
