import os
from PIL import Image
from torch.utils.data import Dataset

class RetinaDataset(Dataset):
    """
    DRIVE layout:
      root/training/images/*.tif
      root/training/1st_manual/XX_manual1.gif
      root/training/mask/XX_training_mask.gif
      root/test/images/*.tif
      root/test/1st_manual/XX_manual1.gif
      root/test/mask/XX_test_mask.gif
    Split files contain absolute image paths inside training/images or test/images.
    Test subset is sampled (20% by script).
    """
    def __init__(self, filepaths, root, transform=None):
        self.filepaths = filepaths
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    @staticmethod
    def _drive_id(fname):
        return os.path.basename(fname)[:2]

    @staticmethod
    def _is_training(path):
        return f"{os.sep}training{os.sep}" in path

    def _paths(self, img_path):
        is_train = self._is_training(img_path)
        two = self._drive_id(img_path)
        split = "training" if is_train else "test"
        vessel_dir = os.path.join(self.root, split, "1st_manual")
        fov_dir = os.path.join(self.root, split, "mask")
        vessel = os.path.join(vessel_dir, f"{two}_manual1.gif")
        suffix = "training" if is_train else "test"
        fov = os.path.join(fov_dir, f"{two}_{suffix}_mask.gif")
        if not os.path.isfile(vessel):
            raise FileNotFoundError(f"Vessel mask missing: {vessel}")
        if not os.path.isfile(fov):
            raise FileNotFoundError(f"FOV mask missing: {fov}")
        return vessel, fov

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        img = Image.open(img_path).convert("RGB")
        vessel_path, fov_path = self._paths(img_path)
        vessel = Image.open(vessel_path).convert("L")
        fov = Image.open(fov_path).convert("L")

        if self.transform:
            img = self.transform(img)
            vessel = self.transform(vessel)
            fov = self.transform(fov)

        vessel = (vessel > 0.5).float()
        fov = (fov > 0.5).float()
        return {"image": img, "mask": vessel, "fov": fov, "image_path": img_path}
