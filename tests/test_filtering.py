"""
Unit Tests for Bilateral, Black Top-Hat, and Hybrid Filtering Modules.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.filtering.bilateral_filter import BilateralFilterModule
from src.filtering.black_tophat import BlackTopHatFilterModule
from src.filtering.hybrid_filter import HybridMRIFilter


class TestFilteringModule(unittest.TestCase):
    def setUp(self):
        self.img = np.random.uniform(0, 255, size=(64, 64)).astype(np.float32)
        self.mri_4ch = np.random.uniform(0, 255, size=(4, 64, 64)).astype(np.float32)

    def test_bilateral_filter(self):
        b_filter = BilateralFilterModule(d=3)
        res = b_filter.apply(self.img)
        self.assertEqual(res.shape, (64, 64))

    def test_black_tophat(self):
        t_filter = BlackTopHatFilterModule(kernel_size=3)
        res = t_filter.apply(self.img)
        self.assertEqual(res.shape, (64, 64))
        self.assertTrue(np.all(res >= 0.0))

    def test_hybrid_filter(self):
        h_filter = HybridMRIFilter(bilateral_d=3, kernel_size=3)
        res = h_filter(self.mri_4ch)
        self.assertEqual(res.shape, (4, 64, 64))


if __name__ == "__main__":
    unittest.main()
