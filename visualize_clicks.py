import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

from lib.datasets.phc_dataset_clicks import PhCDatasetClicks

def load_paths(path):
    triplets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = [p.strip() for p in line.split(",")]
                triplets.append(parts)
    return triplets

def visualize_clicks(dataset, indices, output_dir):
    """Visualize images with positive and negative clicks marked."""
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in indices:
        img, point_mask = dataset[idx]
        
        # Convert tensors to numpy for visualization
        img_np = img.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        point_mask_np = point_mask.squeeze().numpy()  # [1, H, W] -> [H, W]
        
        # Find click locations
        pos_clicks = np.argwhere(point_mask_np == 1)  # Positive clicks
        neg_clicks = np.argwhere(point_mask_np == 0)  # Negative clicks
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left: Image with clicks
        axes[0].imshow(img_np)
        axes[0].scatter(pos_clicks[:, 1], pos_clicks[:, 0], 
                       c='lime', s=100, marker='o', edgecolors='white', 
                       linewidths=2, label=f'Positive ({len(pos_clicks)})')
        axes[0].scatter(neg_clicks[:, 1], neg_clicks[:, 0], 
                       c='red', s=100, marker='x', linewidths=2, 
                       label=f'Negative ({len(neg_clicks)})')
        axes[0].set_title('Image with Click Annotations', fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].axis('off')
        
        # Right: Point mask visualization
        # Create RGB visualization: white=background, green=pos, red=neg
        point_viz = np.ones((*point_mask_np.shape, 3))  # White background
        point_viz[point_mask_np == 1] = [0, 1, 0]  # Green for positive
        point_viz[point_mask_np == 0] = [1, 0, 0]  # Red for negative
        
        axes[1].imshow(point_viz)
        axes[1].set_title('Point Mask Visualization', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle(f'Sample {idx} - Click-based Weak Supervision', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'clicks_sample_{idx}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
        print(f"  - Positive clicks: {len(pos_clicks)}")
        print(f"  - Negative clicks: {len(neg_clicks)}")

def main():
    parser = argparse.ArgumentParser(description='Visualize click annotations from PhC dataset')
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                       help='Path to config file')
    parser.add_argument('--split', type=str, default='splits/phc_train.txt',
                       help='Path to split file')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of samples to visualize')
    parser.add_argument('--num_pos_clicks', type=int, default=10,
                       help='Number of positive clicks')
    parser.add_argument('--num_neg_clicks', type=int, default=10,
                       help='Number of negative clicks')
    parser.add_argument('--output', type=str, default='visualizations/clicks',
                       help='Output directory for visualizations')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                       help='Specific indices to visualize (optional)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Set seed for reproducibility
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    
    # Setup transforms
    size = cfg['image_size']
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    
    # Load dataset
    paths = load_paths(args.split)
    dataset = PhCDatasetClicks(
        paths,
        transform=transform,
        image_size=size,
        num_pos_clicks=args.num_pos_clicks,
        num_neg_clicks=args.num_neg_clicks,
        sample_strategy='random'
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Configuration: {args.num_pos_clicks} positive + {args.num_neg_clicks} negative clicks")
    
    # Select indices to visualize
    if args.indices:
        indices = args.indices
    else:
        # Random sampling
        indices = np.random.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
    
    print(f"\nVisualizing samples: {list(indices)}")
    
    # Visualize
    visualize_clicks(dataset, indices, args.output)
    
    print(f"\nAll visualizations saved to: {args.output}/")

if __name__ == '__main__':
    main()
