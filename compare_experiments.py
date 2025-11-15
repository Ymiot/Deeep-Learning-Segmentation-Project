#!/usr/bin/env python3
"""
Compare multiple weak supervision training experiments.
Reads all checkpoint history files and creates comparison plots.
"""

import os
import json
import argparse
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Compare weak supervision experiments')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                       help='Directory containing checkpoint files')
    parser.add_argument('--output_dir', type=str, default='plots/comparisons',
                       help='Output directory for comparison plots')
    parser.add_argument('--model', type=str, default='unet',
                       help='Model type to compare (unet or encdec)')
    parser.add_argument('--pattern', type=str, default=None,
                       help='Custom pattern to filter checkpoints (e.g., "*10pos*")')
    return parser.parse_args()

def load_history_files(checkpoints_dir, model_name, pattern=None):
    """Load all history JSON files matching the criteria."""
    if pattern:
        search_pattern = os.path.join(checkpoints_dir, pattern)
    else:
        search_pattern = os.path.join(checkpoints_dir, f"{model_name}_phc_weak_*_history.json")
    
    history_files = glob.glob(search_pattern)
    
    experiments = []
    for file_path in sorted(history_files):
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract click configuration from filename
        filename = os.path.basename(file_path)
        # Expected format: {model}_phc_weak_{pos}pos_{neg}neg_history.json
        parts = filename.replace('_history.json', '').split('_')
        
        # Find pos and neg clicks
        pos_clicks = None
        neg_clicks = None
        for i, part in enumerate(parts):
            if part.endswith('pos') and i > 0:
                pos_clicks = int(parts[i].replace('pos', ''))
            if part.endswith('neg') and i > 0:
                neg_clicks = int(parts[i].replace('neg', ''))
        
        if pos_clicks is not None and neg_clicks is not None:
            experiments.append({
                'name': f"{pos_clicks}+{neg_clicks} clicks",
                'pos_clicks': pos_clicks,
                'neg_clicks': neg_clicks,
                'total_clicks': pos_clicks + neg_clicks,
                'data': data,
                'file': file_path
            })
    
    # Sort by total clicks
    experiments.sort(key=lambda x: x['total_clicks'])
    
    return experiments

def plot_metric_comparison(experiments, metric, output_path, plot_type='val'):
    """Create comparison plot for a specific metric."""
    plt.figure(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(experiments)))
    
    for exp, color in zip(experiments, colors):
        data = exp['data']
        
        if plot_type == 'val':
            if metric in data.get('val_metrics_history', {}):
                values = data['val_metrics_history'][metric]
                epochs = range(1, len(values) + 1)
                plt.plot(epochs, values, label=exp['name'], 
                        linewidth=2.5, color=color, marker='o', 
                        markersize=3, markevery=max(1, len(epochs)//10))
        elif plot_type == 'train':
            if metric in data.get('train_metrics_history', {}):
                values = data['train_metrics_history'][metric]
                epochs = range(1, len(values) + 1)
                plt.plot(epochs, values, label=exp['name'], 
                        linewidth=2.5, color=color, marker='o', 
                        markersize=3, markevery=max(1, len(epochs)//10))
        elif plot_type == 'both':
            # Plot both train and val
            if metric in data.get('train_metrics_history', {}):
                train_values = data['train_metrics_history'][metric]
                epochs = range(1, len(train_values) + 1)
                plt.plot(epochs, train_values, label=f"{exp['name']} (train)", 
                        linewidth=2, color=color, linestyle='--', alpha=0.6)
            if metric in data.get('val_metrics_history', {}):
                val_values = data['val_metrics_history'][metric]
                epochs = range(1, len(val_values) + 1)
                plt.plot(epochs, val_values, label=f"{exp['name']} (val)", 
                        linewidth=2.5, color=color)
    
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel(metric.capitalize(), fontsize=13, fontweight='bold')
    
    if plot_type == 'val':
        title = f'Validation {metric.capitalize()} - Comparison Across Click Configurations'
    elif plot_type == 'train':
        title = f'Training {metric.capitalize()} - Comparison Across Click Configurations'
    else:
        title = f'{metric.capitalize()} - Train vs Val Across Click Configurations'
    
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_loss_comparison(experiments, output_path):
    """Create comparison plot for training loss."""
    plt.figure(figsize=(12, 7))
    
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(experiments)))
    
    for exp, color in zip(experiments, colors):
        data = exp['data']
        
        if 'train_loss_history' in data:
            losses = data['train_loss_history']
            epochs = range(1, len(losses) + 1)
            plt.plot(epochs, losses, label=exp['name'], 
                    linewidth=2.5, color=color, marker='o', 
                    markersize=3, markevery=max(1, len(epochs)//10))
    
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Loss', fontsize=13, fontweight='bold')
    plt.title('Training Loss - Comparison Across Click Configurations', 
             fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_best_dice_bar(experiments, output_path):
    """Create bar plot comparing best validation Dice scores."""
    plt.figure(figsize=(10, 6))
    
    names = [exp['name'] for exp in experiments]
    dice_scores = [exp['data'].get('best_val_dice', 0) for exp in experiments]
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(experiments)))
    bars = plt.bar(range(len(names)), dice_scores, color=colors, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, dice_scores)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Click Configuration', fontsize=13, fontweight='bold')
    plt.ylabel('Best Validation Dice Score', fontsize=13, fontweight='bold')
    plt.title('Best Validation Dice - Comparison Across Click Configurations', 
             fontsize=15, fontweight='bold', pad=20)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_summary_table(experiments, output_path):
    """Create a text summary of all experiments."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WEAK SUPERVISION EXPERIMENTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Header
        f.write(f"{'Configuration':<20} {'Best Dice':<12} {'Final Train Loss':<18} {'Epochs':<8}\n")
        f.write("-"*80 + "\n")
        
        for exp in experiments:
            data = exp['data']
            config = exp['name']
            best_dice = data.get('best_val_dice', 0)
            final_loss = data.get('train_loss_history', [0])[-1] if data.get('train_loss_history') else 0
            epochs = data.get('config', {}).get('epochs', 0)
            
            f.write(f"{config:<20} {best_dice:<12.4f} {final_loss:<18.4f} {epochs:<8}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("\nDetailed Metrics (at best epoch):\n")
        f.write("-"*80 + "\n\n")
        
        for exp in experiments:
            data = exp['data']
            f.write(f"\n{exp['name']}:\n")
            f.write(f"  Best Validation Dice: {data.get('best_val_dice', 0):.4f}\n")
            
            # Get metrics at the best epoch (last values in history)
            if 'val_metrics_history' in data:
                f.write("  Final Validation Metrics:\n")
                for metric, values in data['val_metrics_history'].items():
                    if values:
                        f.write(f"    {metric.capitalize():<15s}: {values[-1]:.4f}\n")
    
    print(f"Saved: {output_path}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all experiments
    print(f"Loading experiments from: {args.checkpoints_dir}")
    experiments = load_history_files(args.checkpoints_dir, args.model, args.pattern)
    
    if not experiments:
        print(f"No experiments found matching pattern: {args.model}_phc_weak_*_history.json")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp['name']} (from {os.path.basename(exp['file'])})")
    
    print(f"\nGenerating comparison plots...")
    
    # Plot training loss
    plot_loss_comparison(experiments, 
                        os.path.join(args.output_dir, f'{args.model}_loss_comparison.png'))
    
    # Plot best Dice bar chart
    plot_best_dice_bar(experiments,
                      os.path.join(args.output_dir, f'{args.model}_best_dice_comparison.png'))
    
    # Get all available metrics from first experiment
    if experiments and 'val_metrics_history' in experiments[0]['data']:
        metrics = list(experiments[0]['data']['val_metrics_history'].keys())
        
        # Plot each metric (validation)
        for metric in metrics:
            output_path = os.path.join(args.output_dir, f'{args.model}_{metric}_val_comparison.png')
            plot_metric_comparison(experiments, metric, output_path, plot_type='val')
        
        # Optional: Create combined train/val plots
        # Uncomment if you want these as well
        # for metric in metrics:
        #     output_path = os.path.join(args.output_dir, f'{args.model}_{metric}_trainval_comparison.png')
        #     plot_metric_comparison(experiments, metric, output_path, plot_type='both')
    
    # Create summary table
    summary_path = os.path.join(args.output_dir, f'{args.model}_summary.txt')
    create_summary_table(experiments, summary_path)
    
    print(f"\n{'='*60}")
    print("Comparison complete!")
    print(f"All plots saved to: {args.output_dir}/")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
