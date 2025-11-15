import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class RetinaDataset(Dataset):
    """Dataset pour DRIVE / segmentation rétinienne."""

    def __init__(self, file_list, root=None, transform=None, image_size=128):
        """
        file_list: liste de lignes de type:
            img_path, mask_path, label_path
        root: racine du dataset (optionnel)
        transform: transformations appliquées à l'image
        image_size: taille des images / masques redimensionnés
        """
        self.samples = []
        for line in file_list:
            if len(line) != 3:
                raise ValueError(f"Expected 3 paths per line, got {len(line)}: {line}")
            if root:
                # construction du chemin complet
                line = [os.path.join(root, os.path.relpath(p, start=root)) for p in line]
            self.samples.append(line)

        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label_path = self.samples[idx]

        # --- Chargement des fichiers ---
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # segmentation des vaisseaux

        # FOV : zone visible de la rétine
        if os.path.exists(label_path):
            fov = Image.open(label_path).convert("L")
            # convertir fov en binaire 0/1 pour zone valide
            fov = fov.point(lambda p: 255 if p > 0 else 0)
        else:
            # Si pas de FOV réel, considérer toute l'image comme valide
            fov = Image.new("L", mask.size, 255)

        # --- Transformations ---
        if self.transform:
            img = self.transform(img)

        # Redimensionner mask et fov
        mask = transforms.Resize((self.image_size, self.image_size))(mask)
        fov  = transforms.Resize((self.image_size, self.image_size))(fov)

        mask = transforms.ToTensor()(mask)
        fov  = transforms.ToTensor()(fov)

        return {"image": img, "mask": mask, "fov": fov}
