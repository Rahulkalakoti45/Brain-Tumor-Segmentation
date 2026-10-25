"""
Segmentation Loss Functions Module.

Module Overview:
----------------
1. What it does:
   - Implements soft Dice Loss and Combined Dice + Binary Cross Entropy (BCE) Loss in PyTorch.

2. Why it is required:
   - Standard Cross Entropy loss treats every pixel equally. In brain MRI, healthy background tissue accounts for >95% of voxels, causing standard CE to get trapped in predicting background.
   - Soft Dice Loss explicitly maximizes region overlap, directly handling severe foreground/background class imbalance.

3. Inputs:
   - Predicted Logits Tensor of shape (B, 1, H, W).
   - Ground Truth Binary Target Tensor of shape (B, 1, H, W).
4. Outputs:
   - Scalar Loss Tensor for backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    """
    Soft Dice Similarity Loss.
    """
    def __init__(self, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Input:
            logits (Tensor): Raw network outputs (before Sigmoid). Shape (B, 1, H, W).
            targets (Tensor): Ground truth binary mask {0, 1}. Shape (B, 1, H, W).
        """
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        cardinality = torch.sum(probs_flat * probs_flat, dim=1) + torch.sum(targets_flat * targets_flat, dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        return dice_loss

class DiceBCELoss(nn.Module):
    """
    Combined Soft Dice Loss + Binary Cross Entropy Loss.
    """
    def __init__(self, dice_weight=1.0, bce_weight=1.0, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = SoftDiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        d_loss = self.dice_loss(logits, targets)
        b_loss = self.bce_loss(logits, targets)
        return self.dice_weight * d_loss + self.bce_weight * b_loss
