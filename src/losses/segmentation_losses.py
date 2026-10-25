"""
Segmentation Loss Functions Module.

Module Overview:
----------------
1. What it does:
   - Implements soft Dice Loss, Binary Cross Entropy (BCE), Combined Dice + BCE Loss, Focal Loss, and Tversky Loss.
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
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        cardinality = torch.sum(probs_flat * probs_flat, dim=1) + torch.sum(targets_flat * targets_flat, dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


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


class FocalLoss(nn.Module):
    """
    Focal Loss for handling severe class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss balancing False Positives and False Negatives.
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        tp = torch.sum(probs_flat * targets_flat, dim=1)
        fp = torch.sum(probs_flat * (1 - targets_flat), dim=1)
        fn = torch.sum((1 - probs_flat) * targets_flat, dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()
