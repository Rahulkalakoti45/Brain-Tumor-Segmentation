"""
Unit Tests for Segmentation Metrics (Dice, IoU, HD95, Precision, Sensitivity).
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.segmentation_metrics import calculate_metrics, compute_hd95


class TestSegmentationMetrics(unittest.TestCase):
    def test_perfect_match(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:40, 20:40] = 1.0
        m = calculate_metrics(mask, mask)
        self.assertAlmostEqual(m["dice"], 1.0, places=4)
        self.assertAlmostEqual(m["iou"], 1.0, places=4)
        self.assertAlmostEqual(m["precision"], 1.0, places=4)
        self.assertAlmostEqual(m["sensitivity"], 1.0, places=4)
        self.assertEqual(m["hd95"], 0.0)

    def test_zero_match(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        true = np.zeros((64, 64), dtype=np.float32)
        pred[0:10, 0:10] = 1.0
        true[20:30, 20:30] = 1.0

        m = calculate_metrics(pred, true)
        self.assertLess(m["dice"], 0.1)
        self.assertLess(m["iou"], 0.1)


if __name__ == "__main__":
    unittest.main()
