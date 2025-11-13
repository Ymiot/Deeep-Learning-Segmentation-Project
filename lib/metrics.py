import torch

def threshold(pred, thr=0.5):
    return (pred > thr).long()

def _tp_fp_fn_tn(pred_bin, target_bin, mask=None):
    if mask is not None:
        pred_bin = pred_bin[mask]
        target_bin = target_bin[mask]
    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()
    tn = ((1 - pred_bin) * (1 - target_bin)).sum()
    return tp, fp, fn, tn

def dice_coefficient(pred, target, eps=1e-7, mask=None):
    pred_bin = threshold(pred)
    target_bin = target.long()
    tp, fp, fn, _ = _tp_fp_fn_tn(pred_bin, target_bin, mask)
    return (2*tp + eps) / (2*tp + fp + fn + eps)

def iou(pred, target, eps=1e-7, mask=None):
    pred_bin = threshold(pred)
    target_bin = target.long()
    tp, fp, fn, _ = _tp_fp_fn_tn(pred_bin, target_bin, mask)
    return (tp + eps) / (tp + fp + fn + eps)

def accuracy(pred, target, eps=1e-7, mask=None):
    pred_bin = threshold(pred)
    target_bin = target.long()
    tp, fp, fn, tn = _tp_fp_fn_tn(pred_bin, target_bin, mask)
    return (tp + tn + eps) / (tp + tn + fp + fn + eps)

def sensitivity(pred, target, eps=1e-7, mask=None):
    pred_bin = threshold(pred)
    target_bin = target.long()
    tp, _, fn, _ = _tp_fp_fn_tn(pred_bin, target_bin, mask)
    return (tp + eps) / (tp + fn + eps)

def specificity(pred, target, eps=1e-7, mask=None):
    pred_bin = threshold(pred)
    target_bin = target.long()
    _, fp, _, tn = _tp_fp_fn_tn(pred_bin, target_bin, mask)
    return (tn + eps) / (tn + fp + eps)

def compute_all(pred, target, mask=None):
    return {
        "dice": dice_coefficient(pred, target, mask=mask),
        "iou": iou(pred, target, mask=mask),
        "accuracy": accuracy(pred, target, mask=mask),
        "sensitivity": sensitivity(pred, target, mask=mask),
        "specificity": specificity(pred, target, mask=mask)
    }