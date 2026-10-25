"""
Unit Tests for 2D U-Net Model Architecture and Forward Pass.
"""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.unet import UNet2D, DoubleConv
from src.losses.segmentation_losses import SoftDiceLoss, DiceBCELoss


class TestUNetModel(unittest.TestCase):
    def test_double_conv(self):
        block = DoubleConv(4, 16)
        x = torch.randn(2, 4, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_unet_forward(self):
        model = UNet2D(in_channels=4, out_channels=1, base_filters=16, depth=3)
        x = torch.randn(2, 4, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 64, 64))

    def test_loss_functions(self):
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, size=(2, 1, 64, 64)).float()

        dice_loss = SoftDiceLoss()
        l1 = dice_loss(logits, targets)
        self.assertGreaterEqual(l1.item(), 0.0)

        combined_loss = DiceBCELoss()
        l2 = combined_loss(logits, targets)
        self.assertGreaterEqual(l2.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
