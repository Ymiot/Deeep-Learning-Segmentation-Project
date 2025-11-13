import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class RetinaDataset(Dataset):
    """
    Retinal vessel dataset with vessel masks and FOV masks.
    Directory structure expected:
      root/images/*.png
      root/vessels/*.png     (vessel segmentation)
      root/fov/*.png         (field-of-view mask)
    """
    def __init__(self, root, split_indices, transform=None, use_fov=True):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.vessel_dir = os.path.join(root, "vessels")
        self.fov_dir = os.path.join(root, "fov")
        self.transform = transform
        self.use_fov = use_fov
        all_imgs = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        self.images = [all_imgs[i] for i in split_indices]

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        fname = os.path.basename(img_path)
        vessel_path = os.path.join(self.vessel_dir, fname)
        fov_path = os.path.join(self.fov_dir, fname)

        image = Image.open(img_path).convert("RGB")
        vessel = Image.open(vessel_path).convert("L")
        fov = Image.open(fov_path).convert("L")

        if self.transform:
            image = self.transform(image)
            vessel = self.transform(vessel)
            fov = self.transform(fov)

        vessel = (vessel > 0.5).float()
        fov = (fov > 0.5).float()

        return {"image": image, "mask": vessel, "fov": fov}