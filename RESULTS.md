# Weak Supervision Training Results

**Model:** UNET
**Generated:** 2025-11-15 23:10:32
**Total Experiments:** 9

---

## Table of Contents

1. [Summary Statistics](#summary-statistics)
2. [Detailed Metrics](#detailed-metrics)
3. [Training Convergence](#training-convergence)
4. [Best vs Worst Comparison](#best-vs-worst-comparison)
5. [Visualizations](#visualizations)
6. [Configuration Details](#configuration-details)

---

## Summary Statistics

| Configuration | Total Clicks | Best Val Dice | Final Train Loss | Final Val Acc | Epochs |
|--------------|--------------|---------------|------------------|---------------|--------|
| 5+5 | 10 | **0.9504** | 0.0566 | 0.9706 | 40 |
| 10+10 | 20 | **0.9566** | 0.0485 | 0.9718 | 40 |
| 20+20 | 40 | **0.9631** | 0.0329 | 0.9762 | 40 |
| 10+50 | 60 | **0.9700** | 0.0323 | 0.9803 | 40 |
| 30+30 | 60 | **0.9619** | 0.0329 | 0.9757 | 40 |
| 50+10 | 60 | **0.9435** | 0.0251 | 0.9647 | 40 |
| 50+50 | 100 | **0.9681** | 0.0357 | 0.9775 | 40 |
| 300+300 | 600 | **0.9656** | 0.0251 | 0.9790 | 40 |
| 500+500 | 1000 | **0.9691** | 0.0248 | 0.9812 | 40 |

**Legend:**
- **Configuration**: Number of positive + negative clicks
- **Best Val Dice**: Highest validation Dice score achieved
- **Final Train Loss**: Training loss at final epoch
- **Final Val Acc**: Validation accuracy at final epoch

---

## Detailed Metrics

Final validation metrics for all configurations:

| Config | Dice | IoU | Accuracy | Sensitivity | Specificity |
|--------|------|-----|----------|-------------|-------------|
| 5+5 | 0.9461 | 0.8997 | 0.9706 | 0.9914 | 0.9631 |
| 10+10 | 0.9475 | 0.9015 | 0.9718 | 0.9886 | 0.9665 |
| 20+20 | 0.9555 | 0.9157 | 0.9762 | 0.9962 | 0.9690 |
| 10+50 | 0.9633 | 0.9294 | 0.9803 | 0.9880 | 0.9774 |
| 30+30 | 0.9554 | 0.9151 | 0.9757 | 0.9932 | 0.9691 |
| 50+10 | 0.9363 | 0.8808 | 0.9647 | 0.9978 | 0.9527 |
| 50+50 | 0.9584 | 0.9205 | 0.9775 | 0.9941 | 0.9714 |
| 300+300 | 0.9609 | 0.9252 | 0.9790 | 0.9937 | 0.9736 |
| 500+500 | 0.9649 | 0.9323 | 0.9812 | 0.9947 | 0.9763 |

---

## Training Convergence

| Config | Initial Loss | Final Loss | Loss Reduction | Epochs to Best |
|--------|--------------|------------|----------------|----------------|
| 5+5 | 0.5334 | 0.0566 | 89.39% | 36 |
| 10+10 | 0.5208 | 0.0485 | 90.68% | 38 |
| 20+20 | 0.4989 | 0.0329 | 93.41% | 29 |
| 10+50 | 0.5460 | 0.0323 | 94.08% | 30 |
| 30+30 | 0.4791 | 0.0329 | 93.13% | 34 |
| 50+10 | 0.3709 | 0.0251 | 93.24% | 29 |
| 50+50 | 0.4787 | 0.0357 | 92.54% | 30 |
| 300+300 | 0.4383 | 0.0251 | 94.27% | 17 |
| 500+500 | 0.4226 | 0.0248 | 94.13% | 33 |

---

## Best vs Worst Comparison

| Metric | Best Config | Value | Worst Config | Value | Difference |
|--------|-------------|-------|--------------|-------|------------|
| **Dice Score** | 10+50 | **0.9700** | 50+10 | 0.9435 | +0.0265 |
| Iou | 10+50 | 0.9294 | 50+10 | 0.8808 | +0.0486 |
| Accuracy | 10+50 | 0.9803 | 50+10 | 0.9647 | +0.0155 |
| Sensitivity | 10+50 | 0.9880 | 50+10 | 0.9978 | +-0.0097 |
| Specificity | 10+50 | 0.9774 | 50+10 | 0.9527 | +0.0247 |

---

## Visualizations

### Comparison Plots

#### Best Validation Dice Comparison
![Best Validation Dice Comparison](plots/comparisons/unet_best_dice_comparison.png)

#### Training Loss Comparison
![Training Loss Comparison](plots/comparisons/unet_loss_comparison.png)

#### Validation Dice over Epochs
![Validation Dice over Epochs](plots/comparisons/unet_dice_val_comparison.png)

#### Validation Accuracy over Epochs
![Validation Accuracy over Epochs](plots/comparisons/unet_accuracy_val_comparison.png)

### Individual Training Sessions

#### Configuration: 5+5 clicks

![Overview - 5+5](plots/weak_supervision/unet_5pos_5neg/overview.png)

<details>
<summary>View detailed metric plots for 5+5</summary>

**Dice**
![dice - 5+5](plots/weak_supervision/unet_5pos_5neg/dice.png)

**Iou**
![iou - 5+5](plots/weak_supervision/unet_5pos_5neg/iou.png)

**Accuracy**
![accuracy - 5+5](plots/weak_supervision/unet_5pos_5neg/accuracy.png)

**Sensitivity**
![sensitivity - 5+5](plots/weak_supervision/unet_5pos_5neg/sensitivity.png)

**Specificity**
![specificity - 5+5](plots/weak_supervision/unet_5pos_5neg/specificity.png)

</details>

#### Configuration: 10+10 clicks

![Overview - 10+10](plots/weak_supervision/unet_10pos_10neg/overview.png)

<details>
<summary>View detailed metric plots for 10+10</summary>

**Dice**
![dice - 10+10](plots/weak_supervision/unet_10pos_10neg/dice.png)

**Iou**
![iou - 10+10](plots/weak_supervision/unet_10pos_10neg/iou.png)

**Accuracy**
![accuracy - 10+10](plots/weak_supervision/unet_10pos_10neg/accuracy.png)

**Sensitivity**
![sensitivity - 10+10](plots/weak_supervision/unet_10pos_10neg/sensitivity.png)

**Specificity**
![specificity - 10+10](plots/weak_supervision/unet_10pos_10neg/specificity.png)

</details>

#### Configuration: 20+20 clicks

![Overview - 20+20](plots/weak_supervision/unet_20pos_20neg/overview.png)

<details>
<summary>View detailed metric plots for 20+20</summary>

**Dice**
![dice - 20+20](plots/weak_supervision/unet_20pos_20neg/dice.png)

**Iou**
![iou - 20+20](plots/weak_supervision/unet_20pos_20neg/iou.png)

**Accuracy**
![accuracy - 20+20](plots/weak_supervision/unet_20pos_20neg/accuracy.png)

**Sensitivity**
![sensitivity - 20+20](plots/weak_supervision/unet_20pos_20neg/sensitivity.png)

**Specificity**
![specificity - 20+20](plots/weak_supervision/unet_20pos_20neg/specificity.png)

</details>

#### Configuration: 10+50 clicks

![Overview - 10+50](plots/weak_supervision/unet_10pos_50neg/overview.png)

<details>
<summary>View detailed metric plots for 10+50</summary>

**Dice**
![dice - 10+50](plots/weak_supervision/unet_10pos_50neg/dice.png)

**Iou**
![iou - 10+50](plots/weak_supervision/unet_10pos_50neg/iou.png)

**Accuracy**
![accuracy - 10+50](plots/weak_supervision/unet_10pos_50neg/accuracy.png)

**Sensitivity**
![sensitivity - 10+50](plots/weak_supervision/unet_10pos_50neg/sensitivity.png)

**Specificity**
![specificity - 10+50](plots/weak_supervision/unet_10pos_50neg/specificity.png)

</details>

#### Configuration: 30+30 clicks

![Overview - 30+30](plots/weak_supervision/unet_30pos_30neg/overview.png)

<details>
<summary>View detailed metric plots for 30+30</summary>

**Dice**
![dice - 30+30](plots/weak_supervision/unet_30pos_30neg/dice.png)

**Iou**
![iou - 30+30](plots/weak_supervision/unet_30pos_30neg/iou.png)

**Accuracy**
![accuracy - 30+30](plots/weak_supervision/unet_30pos_30neg/accuracy.png)

**Sensitivity**
![sensitivity - 30+30](plots/weak_supervision/unet_30pos_30neg/sensitivity.png)

**Specificity**
![specificity - 30+30](plots/weak_supervision/unet_30pos_30neg/specificity.png)

</details>

#### Configuration: 50+10 clicks

![Overview - 50+10](plots/weak_supervision/unet_50pos_10neg/overview.png)

<details>
<summary>View detailed metric plots for 50+10</summary>

**Dice**
![dice - 50+10](plots/weak_supervision/unet_50pos_10neg/dice.png)

**Iou**
![iou - 50+10](plots/weak_supervision/unet_50pos_10neg/iou.png)

**Accuracy**
![accuracy - 50+10](plots/weak_supervision/unet_50pos_10neg/accuracy.png)

**Sensitivity**
![sensitivity - 50+10](plots/weak_supervision/unet_50pos_10neg/sensitivity.png)

**Specificity**
![specificity - 50+10](plots/weak_supervision/unet_50pos_10neg/specificity.png)

</details>

#### Configuration: 50+50 clicks

![Overview - 50+50](plots/weak_supervision/unet_50pos_50neg/overview.png)

<details>
<summary>View detailed metric plots for 50+50</summary>

**Dice**
![dice - 50+50](plots/weak_supervision/unet_50pos_50neg/dice.png)

**Iou**
![iou - 50+50](plots/weak_supervision/unet_50pos_50neg/iou.png)

**Accuracy**
![accuracy - 50+50](plots/weak_supervision/unet_50pos_50neg/accuracy.png)

**Sensitivity**
![sensitivity - 50+50](plots/weak_supervision/unet_50pos_50neg/sensitivity.png)

**Specificity**
![specificity - 50+50](plots/weak_supervision/unet_50pos_50neg/specificity.png)

</details>

#### Configuration: 300+300 clicks

![Overview - 300+300](plots/weak_supervision/unet_300pos_300neg/overview.png)

<details>
<summary>View detailed metric plots for 300+300</summary>

**Dice**
![dice - 300+300](plots/weak_supervision/unet_300pos_300neg/dice.png)

**Iou**
![iou - 300+300](plots/weak_supervision/unet_300pos_300neg/iou.png)

**Accuracy**
![accuracy - 300+300](plots/weak_supervision/unet_300pos_300neg/accuracy.png)

**Sensitivity**
![sensitivity - 300+300](plots/weak_supervision/unet_300pos_300neg/sensitivity.png)

**Specificity**
![specificity - 300+300](plots/weak_supervision/unet_300pos_300neg/specificity.png)

</details>

#### Configuration: 500+500 clicks

![Overview - 500+500](plots/weak_supervision/unet_500pos_500neg/overview.png)

<details>
<summary>View detailed metric plots for 500+500</summary>

**Dice**
![dice - 500+500](plots/weak_supervision/unet_500pos_500neg/dice.png)

**Iou**
![iou - 500+500](plots/weak_supervision/unet_500pos_500neg/iou.png)

**Accuracy**
![accuracy - 500+500](plots/weak_supervision/unet_500pos_500neg/accuracy.png)

**Sensitivity**
![sensitivity - 500+500](plots/weak_supervision/unet_500pos_500neg/sensitivity.png)

**Specificity**
![specificity - 500+500](plots/weak_supervision/unet_500pos_500neg/specificity.png)

</details>


---

## Configuration Details

### 5+5 clicks

```json
{
  "model": "unet",
  "num_pos_clicks": 5,
  "num_neg_clicks": 5,
  "image_size": 256,
  "batch_size": 6,
  "lr": 0.0005,
  "epochs": 40
}
```

### 10+10 clicks

```json
{
  "model": "unet",
  "num_pos_clicks": 10,
  "num_neg_clicks": 10,
  "image_size": 256,
  "batch_size": 6,
  "lr": 0.0005,
  "epochs": 40
}
```

### 20+20 clicks

```json
{
  "model": "unet",
  "num_pos_clicks": 20,
  "num_neg_clicks": 20,
  "image_size": 256,
  "batch_size": 6,
  "lr": 0.0005,
  "epochs": 40
}
```

### 10+50 clicks

```json
{
  "model": "unet",
  "num_pos_clicks": 10,
  "num_neg_clicks": 50,
  "image_size": 256,
  "batch_size": 6,
  "lr": 0.0005,
  "epochs": 40
}
```

### 30+30 clicks

```json
{
  "model": "unet",
  "num_pos_clicks": 30,
  "num_neg_clicks": 30,
  "image_size": 256,
  "batch_size": 6,
  "lr": 0.0005,
  "epochs": 40
}
```

### 50+10 clicks

```json
{
  "model": "unet",
  "num_pos_clicks": 50,
  "num_neg_clicks": 10,
  "image_size": 256,
  "batch_size": 6,
  "lr": 0.0005,
  "epochs": 40
}
```

### 50+50 clicks

```json
{
  "model": "unet",
  "num_pos_clicks": 50,
  "num_neg_clicks": 50,
  "image_size": 256,
  "batch_size": 6,
  "lr": 0.0005,
  "epochs": 40
}
```

### 300+300 clicks

```json
{
  "model": "unet",
  "num_pos_clicks": 300,
  "num_neg_clicks": 300,
  "image_size": 256,
  "batch_size": 6,
  "lr": 0.0005,
  "epochs": 40
}
```

### 500+500 clicks

```json
{
  "model": "unet",
  "num_pos_clicks": 500,
  "num_neg_clicks": 500,
  "image_size": 256,
  "batch_size": 6,
  "lr": 0.0005,
  "epochs": 40
}
```

---

## Notes

- All metrics are computed on the validation set using full masks
- Training metrics are computed only on clicked pixels
- Best model is saved based on validation Dice score
- Click sampling uses random strategy
