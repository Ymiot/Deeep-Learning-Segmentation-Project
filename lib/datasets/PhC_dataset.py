import os
from PIL import Image
from torch.utils.data import Dataset

class PhCDataset(Dataset):
    """
    Generic PhC dataset with filepaths split lists.
    Expects images under some images_dir; masks under masks_dir with same filenames.
    Example:
      root/images/xxx.png
      root/masks/xxx.png
    Split files contain absolute image paths.
    """
    def __init__(self, filepaths, images_root, masks_root, transform=None):
        self.filepaths = filepaths
        self.images_root = images_root
        self.masks_root = masks_root
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def _mask_path_from_image(self, img_path):
        # Replace images_root with masks_root and keep filename
        fname = os.path.basename(img_path)
        return os.path.join(self.masks_root, fname)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        mask_path = self._mask_path_from_image(img_path)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask not found for {img_path}: {mask_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        mask = (mask > 0.5).float()
        return img, mask
