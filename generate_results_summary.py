#!/usr/bin/env python3
"""
Generate a comprehensive markdown summary of all weak supervision experiments.
Creates tables, statistics, and embeds plots for easy viewing.
"""

import os
import json
import argparse
import glob
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Generate markdown summary of experiments')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                       help='Directory containing checkpoint files')
    parser.add_argument('--plots_dir', type=str, default='plots/weak_supervision',
                       help='Directory containing plot subdirectories')
    parser.add_argument('--output', type=str, default='RESULTS.md',
                       help='Output markdown file')
    parser.add_argument('--model', type=str, default='unet',
                       help='Model type to summarize (unet or encdec)')
    return parser.parse_args()

def load_history_files(checkpoints_dir, model_name):
    """Load all history JSON files for the specified model."""
    search_pattern = os.path.join(checkpoints_dir, f"{model_name}_phc_weak_*_history.json")
    history_files = glob.glob(search_pattern)
    
    experiments = []
    for file_path in sorted(history_files):
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract click configuration from filename
        filename = os.path.basename(file_path)
        parts = filename.replace('_history.json', '').split('_')
        
        pos_clicks = None
        neg_clicks = None
        for i, part in enumerate(parts):
            if part.endswith('pos'):
                pos_clicks = int(part.replace('pos', ''))
            if part.endswith('neg'):
                neg_clicks = int(part.replace('neg', ''))
        
        if pos_clicks is not None and neg_clicks is not None:
            experiments.append({
                'pos_clicks': pos_clicks,
                'neg_clicks': neg_clicks,
                'total_clicks': pos_clicks + neg_clicks,
                'data': data,
                'file': file_path
            })
    
    experiments.sort(key=lambda x: x['total_clicks'])
    return experiments

def format_metric_value(value, precision=4):
    """Format metric value with specified precision."""
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return str(value)

def create_summary_table(experiments):
    """Create markdown table with summary metrics."""
    lines = []
    lines.append("| Configuration | Total Clicks | Best Val Dice | Final Train Loss | Final Val Acc | Epochs |")
    lines.append("|--------------|--------------|---------------|------------------|---------------|--------|")
    
    for exp in experiments:
        data = exp['data']
        config = f"{exp['pos_clicks']}+{exp['neg_clicks']}"
        total = exp['total_clicks']
        best_dice = format_metric_value(data.get('best_val_dice', 0))
        
        train_loss = data.get('train_loss_history', [])
        final_loss = format_metric_value(train_loss[-1] if train_loss else 0)
        
        val_acc = data.get('val_metrics_history', {}).get('accuracy', [])
        final_acc = format_metric_value(val_acc[-1] if val_acc else 0)
        
        epochs = data.get('config', {}).get('epochs', 0)
        
        lines.append(f"| {config} | {total} | **{best_dice}** | {final_loss} | {final_acc} | {epochs} |")
    
    return "\n".join(lines)

def create_detailed_metrics_table(experiments):
    """Create detailed metrics table for all experiments."""
    lines = []
    lines.append("| Config | Dice | IoU | Accuracy | Sensitivity | Specificity |")
    lines.append("|--------|------|-----|----------|-------------|-------------|")
    
    for exp in experiments:
        data = exp['data']
        config = f"{exp['pos_clicks']}+{exp['neg_clicks']}"
        
        val_metrics = data.get('val_metrics_history', {})
        
        metrics = {}
        for metric_name in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
            values = val_metrics.get(metric_name, [])
            metrics[metric_name] = format_metric_value(values[-1] if values else 0)
        
        lines.append(f"| {config} | {metrics['dice']} | {metrics['iou']} | "
                    f"{metrics['accuracy']} | {metrics['sensitivity']} | {metrics['specificity']} |")
    
    return "\n".join(lines)

def create_training_convergence_table(experiments):
    """Create table showing training convergence metrics."""
    lines = []
    lines.append("| Config | Initial Loss | Final Loss | Loss Reduction | Epochs to Best |")
    lines.append("|--------|--------------|------------|----------------|----------------|")
    
    for exp in experiments:
        data = exp['data']
        config = f"{exp['pos_clicks']}+{exp['neg_clicks']}"
        
        train_loss = data.get('train_loss_history', [])
        if train_loss:
            initial_loss = format_metric_value(train_loss[0])
            final_loss = format_metric_value(train_loss[-1])
            reduction = format_metric_value((train_loss[0] - train_loss[-1]) / train_loss[0] * 100, 2)
        else:
            initial_loss = final_loss = reduction = "N/A"
        
        # Find epoch where best dice was achieved
        val_dice = data.get('val_metrics_history', {}).get('dice', [])
        best_dice = data.get('best_val_dice', 0)
        epoch_to_best = "N/A"
        if val_dice and best_dice > 0:
            for i, dice in enumerate(val_dice):
                if abs(dice - best_dice) < 1e-6:
                    epoch_to_best = str(i + 1)
                    break
        
        lines.append(f"| {config} | {initial_loss} | {final_loss} | {reduction}% | {epoch_to_best} |")
    
    return "\n".join(lines)

def create_best_worst_comparison(experiments):
    """Create comparison of best and worst performing configurations."""
    if not experiments:
        return "No experiments found."
    
    # Sort by best dice
    sorted_by_dice = sorted(experiments, key=lambda x: x['data'].get('best_val_dice', 0), reverse=True)
    
    best = sorted_by_dice[0]
    worst = sorted_by_dice[-1]
    
    lines = []
    lines.append("| Metric | Best Config | Value | Worst Config | Value | Difference |")
    lines.append("|--------|-------------|-------|--------------|-------|------------|")
    
    best_config = f"{best['pos_clicks']}+{best['neg_clicks']}"
    worst_config = f"{worst['pos_clicks']}+{worst['neg_clicks']}"
    
    # Compare Dice
    best_dice = best['data'].get('best_val_dice', 0)
    worst_dice = worst['data'].get('best_val_dice', 0)
    diff_dice = format_metric_value(best_dice - worst_dice)
    
    lines.append(f"| **Dice Score** | {best_config} | **{format_metric_value(best_dice)}** | "
                f"{worst_config} | {format_metric_value(worst_dice)} | +{diff_dice} |")
    
    # Compare other metrics
    for metric in ['iou', 'accuracy', 'sensitivity', 'specificity']:
        best_val = best['data'].get('val_metrics_history', {}).get(metric, [])
        worst_val = worst['data'].get('val_metrics_history', {}).get(metric, [])
        
        if best_val and worst_val:
            best_v = best_val[-1]
            worst_v = worst_val[-1]
            diff = format_metric_value(best_v - worst_v)
            
            lines.append(f"| {metric.capitalize()} | {best_config} | {format_metric_value(best_v)} | "
                        f"{worst_config} | {format_metric_value(worst_v)} | +{diff} |")
    
    return "\n".join(lines)

def embed_plots(experiments, plots_dir, model_name):
    """Create markdown to embed comparison plots."""
    lines = []
    
    # Check for comparison plots
    comparison_dir = "plots/comparisons"
    if os.path.exists(comparison_dir):
        lines.append("### Comparison Plots")
        lines.append("")
        
        comparison_plots = [
            (f'{model_name}_best_dice_comparison.png', 'Best Validation Dice Comparison'),
            (f'{model_name}_loss_comparison.png', 'Training Loss Comparison'),
            (f'{model_name}_dice_val_comparison.png', 'Validation Dice over Epochs'),
            (f'{model_name}_accuracy_val_comparison.png', 'Validation Accuracy over Epochs'),
        ]
        
        for plot_file, title in comparison_plots:
            plot_path = os.path.join(comparison_dir, plot_file)
            if os.path.exists(plot_path):
                lines.append(f"#### {title}")
                lines.append(f"![{title}]({plot_path})")
                lines.append("")
    
    # Individual experiment plots
    lines.append("### Individual Training Sessions")
    lines.append("")
    
    for exp in experiments:
        config = f"{exp['pos_clicks']}+{exp['neg_clicks']}"
        session_dir = f"{plots_dir}/{model_name}_{exp['pos_clicks']}pos_{exp['neg_clicks']}neg"
        
        if os.path.exists(session_dir):
            lines.append(f"#### Configuration: {config} clicks")
            lines.append("")
            
            # Overview plot
            overview_path = os.path.join(session_dir, "overview.png")
            if os.path.exists(overview_path):
                lines.append(f"![Overview - {config}]({overview_path})")
                lines.append("")
            
            lines.append("<details>")
            lines.append(f"<summary>View detailed metric plots for {config}</summary>")
            lines.append("")
            
            # Individual metrics
            for metric in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
                metric_path = os.path.join(session_dir, f"{metric}.png")
                if os.path.exists(metric_path):
                    lines.append(f"**{metric.capitalize()}**")
                    lines.append(f"![{metric} - {config}]({metric_path})")
                    lines.append("")
            
            lines.append("</details>")
            lines.append("")
    
    return "\n".join(lines)

def generate_markdown(experiments, model_name, plots_dir):
    """Generate complete markdown document."""
    lines = []
    
    # Header
    lines.append("# Weak Supervision Training Results")
    lines.append("")
    lines.append(f"**Model:** {model_name.upper()}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Experiments:** {len(experiments)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("")
    lines.append("1. [Summary Statistics](#summary-statistics)")
    lines.append("2. [Detailed Metrics](#detailed-metrics)")
    lines.append("3. [Training Convergence](#training-convergence)")
    lines.append("4. [Best vs Worst Comparison](#best-vs-worst-comparison)")
    lines.append("5. [Visualizations](#visualizations)")
    lines.append("6. [Configuration Details](#configuration-details)")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Summary Statistics
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(create_summary_table(experiments))
    lines.append("")
    lines.append("**Legend:**")
    lines.append("- **Configuration**: Number of positive + negative clicks")
    lines.append("- **Best Val Dice**: Highest validation Dice score achieved")
    lines.append("- **Final Train Loss**: Training loss at final epoch")
    lines.append("- **Final Val Acc**: Validation accuracy at final epoch")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Detailed Metrics
    lines.append("## Detailed Metrics")
    lines.append("")
    lines.append("Final validation metrics for all configurations:")
    lines.append("")
    lines.append(create_detailed_metrics_table(experiments))
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Training Convergence
    lines.append("## Training Convergence")
    lines.append("")
    lines.append(create_training_convergence_table(experiments))
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Best vs Worst
    lines.append("## Best vs Worst Comparison")
    lines.append("")
    lines.append(create_best_worst_comparison(experiments))
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Visualizations
    lines.append("## Visualizations")
    lines.append("")
    lines.append(embed_plots(experiments, plots_dir, model_name))
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Configuration Details
    lines.append("## Configuration Details")
    lines.append("")
    
    for exp in experiments:
        config_name = f"{exp['pos_clicks']}+{exp['neg_clicks']}"
        lines.append(f"### {config_name} clicks")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(exp['data'].get('config', {}), indent=2))
        lines.append("```")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Footer
    lines.append("## Notes")
    lines.append("")
    lines.append("- All metrics are computed on the validation set using full masks")
    lines.append("- Training metrics are computed only on clicked pixels")
    lines.append("- Best model is saved based on validation Dice score")
    lines.append("- Click sampling uses random strategy")
    lines.append("")
    
    return "\n".join(lines)

def main():
    args = parse_args()
    
    print(f"Loading experiments from: {args.checkpoints_dir}")
    experiments = load_history_files(args.checkpoints_dir, args.model)
    
    if not experiments:
        print(f"No experiments found for model: {args.model}")
        return
    
    print(f"Found {len(experiments)} experiments")
    
    print(f"Generating markdown summary...")
    markdown_content = generate_markdown(experiments, args.model, args.plots_dir)
    
    # Write to file
    with open(args.output, 'w') as f:
        f.write(markdown_content)
    
    print(f"\n{'='*60}")
    print(f"Markdown summary created: {args.output}")
    print(f"{'='*60}")
    
    # Print quick summary
    print("\nQuick Summary:")
    print(f"  Configurations tested: {len(experiments)}")
    
    if experiments:
        best = max(experiments, key=lambda x: x['data'].get('best_val_dice', 0))
        print(f"  Best configuration: {best['pos_clicks']}+{best['neg_clicks']} clicks")
        print(f"  Best Dice score: {best['data'].get('best_val_dice', 0):.4f}")

if __name__ == '__main__':
    main()
