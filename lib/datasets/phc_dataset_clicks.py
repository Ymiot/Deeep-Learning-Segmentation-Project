import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class PhCDatasetClicks(Dataset):
    """Dataset for PhC segmentation with click-based supervision."""

    def __init__(self, file_list, transform=None, image_size=128, 
                 num_pos_clicks=10, num_neg_clicks=10, sample_strategy='random'):
        """
        file_list: liste de triplets [img_path, mask_path]
        transform: transformations appliquées à l'image
        image_size: taille des images / masques redimensionnés
        num_pos_clicks: number of positive clicks to sample
        num_neg_clicks: number of negative clicks to sample
        sample_strategy: sampling strategy ('random', etc.)
        """
        self.samples = []
        for line in file_list:
            if len(line) != 2:
                raise ValueError(f"Expected 2 paths per line, got {len(line)}: {line}")
            self.samples.append(line)

        self.transform = transform
        self.image_size = image_size
        self.num_pos_clicks = num_pos_clicks
        self.num_neg_clicks = num_neg_clicks
        self.sample_strategy = sample_strategy

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
        
        # Generate point_mask by sampling clicks from the full mask
        point_mask = self._generate_clicks(mask)

        return img, point_mask
    
    def _generate_clicks(self, mask):
        """
        Generate a point mask by sampling clicks from the full segmentation mask.
        
        Args:
            mask: Full segmentation mask tensor of shape [1, H, W]
        
        Returns:
            point_mask: Tensor of shape [1, H, W] with:
                - 1 at positive click locations
                - 0 at negative click locations
                - -1 everywhere else (don't care)
        """
        # Convert mask to binary (threshold at 0.5)
        mask_np = mask.squeeze().numpy()  # [H, W]
        binary_mask = (mask_np > 0.5).astype(np.uint8)
        
        # Find coordinates of positive and negative pixels
        pos_coords = np.argwhere(binary_mask == 1)  # Shape: [N_pos, 2]
        neg_coords = np.argwhere(binary_mask == 0)  # Shape: [N_neg, 2]
        
        # Initialize point_mask with "don't care" value (-1)
        point_mask = np.full_like(mask_np, -1.0, dtype=np.float32)
        
        # Sample clicks based on strategy
        if self.sample_strategy == 'random':
            # Sample positive clicks
            if len(pos_coords) > 0:
                num_pos_to_sample = min(self.num_pos_clicks, len(pos_coords))
                pos_indices = np.random.choice(len(pos_coords), size=num_pos_to_sample, replace=False)
                sampled_pos = pos_coords[pos_indices]
                point_mask[sampled_pos[:, 0], sampled_pos[:, 1]] = 1.0
            
            # Sample negative clicks
            if len(neg_coords) > 0:
                num_neg_to_sample = min(self.num_neg_clicks, len(neg_coords))
                neg_indices = np.random.choice(len(neg_coords), size=num_neg_to_sample, replace=False)
                sampled_neg = neg_coords[neg_indices]
                point_mask[sampled_neg[:, 0], sampled_neg[:, 1]] = 0.0
        
        # Convert back to tensor and add channel dimension
        point_mask = torch.from_numpy(point_mask).unsqueeze(0)  # [1, H, W]
        
        return point_mask
