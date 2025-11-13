import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class PhCDataset(Dataset):
    """
    Dataset for Phase Contrast (PhC) cell images.
    Expects directory:
      root/images/*.png|jpg
      root/masks/*.png|jpg
    Masks should be binary or convertible to binary (non-zero = foreground).
    """
    def __init__(self, root, split_indices, transform=None):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.transform = transform
        all_imgs = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        self.images = [all_imgs[i] for i in split_indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        fname = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, fname)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = (mask > 0.5).float()  # binarize
        return image, mask