"""
Unit Tests for Intensity Normalization, Preprocessing Pipeline, and Patch Extraction.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.normalization import zscore_normalize, minmax_normalize, clip_outliers
from src.preprocessing.preprocessing import MRIPreprocessingPipeline
from src.preprocessing.patch_extraction import PatchExtractor


class TestPreprocessingModule(unittest.TestCase):
    def test_zscore_normalize(self):
        arr = np.random.uniform(10, 100, size=(100, 100)).astype(np.float32)
        normed = zscore_normalize(arr)
        self.assertAlmostEqual(float(np.mean(normed)), 0.0, places=4)
        self.assertAlmostEqual(float(np.std(normed)), 1.0, places=4)

    def test_preprocessing_pipeline(self):
        pipeline = MRIPreprocessingPipeline(target_size=(64, 64))
        mri_input = np.random.normal(loc=100, scale=20, size=(4, 128, 128)).astype(np.float32)
        output = pipeline(mri_input)
        self.assertEqual(output.shape, (4, 64, 64))

    def test_patch_extractor(self):
        extractor = PatchExtractor(patch_size=(32, 32), stride=(32, 32))
        slice_img = np.random.uniform(0, 1, size=(4, 64, 64)).astype(np.float32)
        patches = extractor.extract_patches(slice_img)
        self.assertEqual(patches.shape, (4, 4, 32, 32)) # 4 patches, 4 channels, 32x32

        reconstructed = extractor.reconstruct_slice(patches, target_shape=(4, 64, 64))
        self.assertEqual(reconstructed.shape, (4, 64, 64))
        np.testing.assert_allclose(reconstructed, slice_img, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
