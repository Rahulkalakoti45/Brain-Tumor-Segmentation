"""Losses module initialization."""
from src.losses.segmentation_losses import SoftDiceLoss, DiceBCELoss, FocalLoss, TverskyLoss

__all__ = ["SoftDiceLoss", "DiceBCELoss", "FocalLoss", "TverskyLoss"]
