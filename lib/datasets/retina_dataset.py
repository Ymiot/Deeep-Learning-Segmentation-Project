import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class RetinaDataset(Dataset):
    """
    DRIVE-compatible dataset:
      root/training/images/*.tif
      root/training/1st_manual/*.gif     (vessel labels)
      root/training/mask/*.gif           (FOV masks)
      root/test/images/*.tif
      root/test/1st_manual/*.gif
      root/test/mask/*.gif

    Split files contain absolute paths to images (one per line).
    We derive corresponding vessel + FOV paths from DRIVE naming rules.
    """
    def __init__(self, filepaths, root, transform=None):
        self.filepaths = filepaths
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    @staticmethod
    def _drive_id_from_filename(fname):
        # E.g. 01_training.tif or 01_test.tif -> "01"
        # Assumes filenames start with two digits
        return os.path.basename(fname)[:2]

    @staticmethod
    def _split_from_path(path):
        return "training" if f"{os.sep}training{os.sep}" in path else "test"

    def _derive_paths(self, img_path):
        split = self._split_from_path(img_path)
        two = self._drive_id_from_filename(img_path)

        vessel_dir = os.path.join(self.root, split, "1st_manual")
        fov_dir = os.path.join(self.root, split, "mask")

        # Vessel labels follow: 01_manual1.gif
        vessel_name = f"{two}_manual1.gif"
        vessel_path = os.path.join(vessel_dir, vessel_name)

        # FOV follows: 01_training_mask.gif or 01_test_mask.gif
        suffix = "training" if split == "training" else "test"
        fov_name = f"{two}_{suffix}_mask.gif"
        fov_path = os.path.join(fov_dir, fov_name)

        if not os.path.isfile(vessel_path):
            raise FileNotFoundError(f"Vessel mask not found: {vessel_path}")
        if not os.path.isfile(fov_path):
            raise FileNotFoundError(f"FOV mask not found: {fov_path}")
        return vessel_path, fov_path

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        img = Image.open(img_path).convert("RGB")
        vessel_path, fov_path = self._derive_paths(img_path)
        vessel = Image.open(vessel_path).convert("L")  # binary vessel label
        fov = Image.open(fov_path).convert("L")        # binary FOV mask

        if self.transform:
            img = self.transform(img)
            vessel = self.transform(vessel)
            fov = self.transform(fov)

        vessel = (vessel > 0.5).float()
        fov = (fov > 0.5).float()
        return {"image": img, "mask": vessel, "fov": fov, "image_path": img_path}
