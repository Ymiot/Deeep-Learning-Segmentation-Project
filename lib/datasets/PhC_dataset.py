import os
from PIL import Image
from torch.utils.data import Dataset

class PhCDataset(Dataset):
    """
    phc_data layout:
      root/train/images/*.jpg
      root/train/labels/*.jpg (or same extension)
      root/test/images/*.jpg
      root/test/labels/*.jpg
    Split files contain absolute image paths pointing into train/images or test/images.
    Label path is derived by replacing '/images/' with '/labels/'.
    """
    def __init__(self, filepaths, transform=None):
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def _label_path(self, img_path):
        return img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label_path = self._label_path(img_path)
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Label not found for image {img_path}: {label_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path).convert("L")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        mask = (mask > 0.5).float()
        return img, mask
