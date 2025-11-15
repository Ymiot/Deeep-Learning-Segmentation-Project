#!/bin/bash

# Weak Supervision Training Workflow
# Runs training with different click configurations for comparison

set -e  # Exit on error

# Configuration
MODEL="unet"  # Model to use: "unet" or "encdec"
CONFIG="configs/default.yaml"
DEVICE="cuda"

# Arrays of click configurations to test
# Format: "num_pos_clicks num_neg_clicks"
CLICK_CONFIGS=(
    "5 5"
    "10 10"
    "20 20"
    "30 30"
    "50 50"
    "10 50"
    "50 10"
    "300 300"
    "500 500"
)

# Optional: Uncomment to test different models
# MODELS=("encdec" "unet")

echo "=========================================="
echo "Weak Supervision Training Workflow"
echo "=========================================="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Click configurations: ${#CLICK_CONFIGS[@]}"
echo "=========================================="
echo ""

# Create output directories
mkdir -p checkpoints
mkdir -p plots/weak_supervision
mkdir -p visualizations/clicks
mkdir -p logs

# Log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/weak_training_${MODEL}_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo ""

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting workflow at: $(date)"
echo ""

# Counter for progress
TOTAL=${#CLICK_CONFIGS[@]}
CURRENT=0

# Loop through each click configuration
for config in "${CLICK_CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Parse configuration
    read -r NUM_POS NUM_NEG <<< "$config"
    
    echo "=========================================="
    echo "Configuration [$CURRENT/$TOTAL]"
    echo "Positive clicks: $NUM_POS"
    echo "Negative clicks: $NUM_NEG"
    echo "=========================================="
    
    # Visualize clicks for this configuration (only first time or if needed)
    if [ $CURRENT -eq 1 ]; then
        echo "Generating click visualizations..."
        python visualize_clicks.py \
            --num_pos_clicks $NUM_POS \
            --num_neg_clicks $NUM_NEG \
            --num_samples 4 \
            --output "visualizations/clicks_${NUM_POS}pos_${NUM_NEG}neg" || {
                echo "Warning: Visualization failed, continuing..."
            }
        echo ""
    fi
    
    # Run training
    echo "Starting training..."
    python train_weak.py \
        --config $CONFIG \
        --model $MODEL \
        --num_pos_clicks $NUM_POS \
        --num_neg_clicks $NUM_NEG \
        --device $DEVICE
    
    echo ""
    echo "Training completed for ${NUM_POS}+${NUM_NEG} clicks"
    echo "Checkpoint saved to: checkpoints/${MODEL}_phc_weak_${NUM_POS}pos_${NUM_NEG}neg_best.pt"
    echo ""
    
    # Small delay between runs
    sleep 2
done

echo "=========================================="
echo "All training runs completed!"
echo "=========================================="
echo ""

# Summary of results
echo "Summary of trained models:"
echo ""
for config in "${CLICK_CONFIGS[@]}"; do
    read -r NUM_POS NUM_NEG <<< "$config"
    CHECKPOINT="checkpoints/${MODEL}_phc_weak_${NUM_POS}pos_${NUM_NEG}neg_best.pt"
    HISTORY="checkpoints/${MODEL}_phc_weak_${NUM_POS}pos_${NUM_NEG}neg_history.json"
    
    if [ -f "$CHECKPOINT" ]; then
        SIZE=$(du -h "$CHECKPOINT" | cut -f1)
        echo "✓ ${NUM_POS}+${NUM_NEG} clicks: $CHECKPOINT ($SIZE)"
        
        # Extract best dice if history file exists
        if [ -f "$HISTORY" ]; then
            DICE=$(python -c "import json; print(f\"{json.load(open('$HISTORY'))['best_val_dice']:.4f}\")" 2>/dev/null || echo "N/A")
            echo "  Best Validation Dice: $DICE"
        fi
    else
        echo "✗ ${NUM_POS}+${NUM_NEG} clicks: FAILED"
    fi
done

echo ""
echo "Plots saved to: plots/weak_supervision/"
echo "Checkpoints saved to: checkpoints/"
echo "Training history saved to: checkpoints/*_history.json"
echo ""
echo "Workflow completed at: $(date)"
echo "Total duration: $SECONDS seconds"
