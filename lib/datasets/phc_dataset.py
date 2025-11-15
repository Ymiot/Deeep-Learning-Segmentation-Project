import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class PhCDataset(Dataset):
    """Dataset pour PHC segmentation."""

    def __init__(self, file_list, transform=None, image_size=128):
        """
        file_list: liste de triplets [img_path, mask_path]
        transform: transformations appliquées à l'image
        image_size: taille des images / masques redimensionnés
        """
        self.samples = []
        for line in file_list:
            if len(line) != 2:
                raise ValueError(f"Expected 2 paths per line, got {len(line)}: {line}")
            self.samples.append(line)

        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # mask grayscale

        # transformation de l'image uniquement
        if self.transform:
            img = self.transform(img)

        # redimensionner mask
        mask = transforms.Resize((self.image_size, self.image_size))(mask)
        mask = transforms.ToTensor()(mask)

        return img, mask
