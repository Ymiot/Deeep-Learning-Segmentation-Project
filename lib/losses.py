import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self, weight_pos=None):
        super().__init__()
        self.weight_pos = weight_pos
    def forward(self, logits, targets):
        if self.weight_pos is not None:
            weights = torch.ones_like(targets)
            weights[targets == 1] = self.weight_pos
            return F.binary_cross_entropy_with_logits(logits, targets.float(), weight=weights)
        return F.binary_cross_entropy_with_logits(logits, targets.float())

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice = (2*intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs*targets + (1 - probs)*(1 - targets)
        focal = self.alpha * (1 - pt).pow(self.gamma) * bce
        return focal.mean()

class BCELoss_TotalVariation(nn.Module):
    def __init__(self, tv_weight=1e-4):
        super().__init__()
        self.tv_weight = tv_weight
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        probs = torch.sigmoid(logits)
        dx = torch.abs(probs[:,:,1:,:] - probs[:,:,:-1,:]).mean()
        dy = torch.abs(probs[:,:,:,1:] - probs[:,:,:, :-1]).mean()
        tv = dx + dy
        return bce_loss + self.tv_weight * tv

class ClickSupervisionLoss(nn.Module):
    """
    Loss function for click-based supervision.
    Only computes loss at clicked pixels (where point_mask != -1).
    Ignores all pixels where point_mask == -1.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(self, logits, point_mask):
        """
        Args:
            logits: Model predictions (raw logits), shape [B, 1, H, W]
            point_mask: Sparse annotation with values:
                - 1: positive click
                - 0: negative click
                - -1: don't care (ignored)
        
        Returns:
            loss: BCE loss computed only at clicked pixels
        """
        # Create boolean mask for valid pixels (where point_mask != -1)
        valid_mask = (point_mask != -1)  # [B, 1, H, W]
        
        # Check if there are any valid pixels
        if valid_mask.sum() == 0:
            # No clicks available, return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Select only the valid pixels
        selected_logits = logits[valid_mask]  # [N_valid]
        selected_targets = point_mask[valid_mask]  # [N_valid]
        
        # Ensure targets are float and in [0, 1] range
        selected_targets = selected_targets.float()
        
        # Compute BCE loss on selected pixels
        loss = F.binary_cross_entropy_with_logits(
            selected_logits, 
            selected_targets, 
            reduction='mean'
        )
        
        return loss