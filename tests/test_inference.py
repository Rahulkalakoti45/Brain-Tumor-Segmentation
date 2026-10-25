"""
Unit Tests for Volume-Level Inference Used by the CLI and Web App.
"""

import sys
import os
import unittest
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import MODALITIES, segment_volume


class TestVolumeInference(unittest.TestCase):
    def setUp(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "full_pipeline.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def test_segment_volume_shape_and_binary_output(self):
        rng = np.random.default_rng(0)
        mri = rng.random((len(MODALITIES), 64, 64, 4)).astype(np.float32)
        calls = []

        mask, checkpoint_loaded = segment_volume(
            mri, self.config, checkpoint_path=None, progress_callback=calls.append
        )

        self.assertEqual(mask.shape, (64, 64, 4))
        self.assertTrue(set(np.unique(mask)).issubset({0, 1}))
        self.assertFalse(checkpoint_loaded)
        self.assertEqual(calls[-1], 1.0)

    def test_empty_volume_gives_empty_mask(self):
        mri = np.zeros((len(MODALITIES), 64, 64, 3), dtype=np.float32)

        mask, _ = segment_volume(mri, self.config, checkpoint_path=None)

        self.assertEqual(int(mask.sum()), 0)


if __name__ == "__main__":
    unittest.main()
