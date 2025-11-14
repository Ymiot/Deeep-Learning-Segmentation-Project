import torch
from torch.utils.data import Dataset
from PIL import Image

class PhCDataset(Dataset):
    """Dataset pour PHC segmentation."""
    def __init__(self, file_list, transform=None):
        """
        file_list: liste de chemins de type ['/path/img.jpg', '/path/label.png']
        """
        self.samples = []
        for line in file_list:
            parts = line.split(",")
            if len(parts) != 2:
                raise ValueError(f"Expected 2 paths per line, got {len(parts)}: {line}")
            self.samples.append((parts[0], parts[1]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # mask grayscale

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
