"""
Unit Tests for Dataset Loader & Patient Split Manager.
"""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders, generate_patient_splits
from src.data.nifti_loader import NiftiLoader
from src.data.validation import DatasetValidator


class TestDatasetModule(unittest.TestCase):
    def test_nifti_loader_mock(self):
        loader = NiftiLoader()
        mock_data = loader.generate_mock_volume("subject_test")
        self.assertEqual(mock_data["modalities"].shape, (4, 128, 128, 64))
        self.assertEqual(mock_data["mask"].shape, (128, 128, 64))
        self.assertTrue(mock_data["is_mock"])

    def test_dataset_validator(self):
        validator = DatasetValidator(raw_dir="data/raw")
        report = validator.validate()
        self.assertIn("status", report)

    def test_dataloaders(self):
        dataloaders = get_dataloaders(raw_dir="data/raw", batch_size=2)
        self.assertIn("train", dataloaders)
        self.assertIn("val", dataloaders)
        self.assertIn("test", dataloaders)

        inputs, targets, slice_ids = next(iter(dataloaders["train"]))
        self.assertEqual(inputs.shape[0], 2)
        self.assertEqual(inputs.shape[1], 4)
        self.assertEqual(targets.shape[1], 1)


if __name__ == "__main__":
    unittest.main()
