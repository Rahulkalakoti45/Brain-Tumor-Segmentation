"""
Unit Tests for Automatic Prompt Generation and SAM 2 Refinement Module.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prompt_generation.automatic_prompt import AutomaticPromptGenerator
from src.refinement.sam2_refiner import SAM2TinyRefiner


class TestPromptGeneration(unittest.TestCase):
    def test_automatic_prompt_generator(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:40, 20:40] = 1.0

        gen = AutomaticPromptGenerator(mode="bbox")
        res = gen.generate_prompts_single(mask)

        self.assertTrue(res["has_tumor"])
        self.assertEqual(len(res["bboxes"]), 1)
        bbox = res["bboxes"][0]
        self.assertLessEqual(bbox[0], 20) # ymin padded
        self.assertLessEqual(bbox[1], 20) # xmin padded

    def test_sam2_refiner_mock(self):
        refiner = SAM2TinyRefiner(use_mock=True)
        img_slice = np.random.uniform(0, 255, size=(64, 64)).astype(np.float32)
        u_mask = np.zeros((64, 64), dtype=np.float32)
        u_mask[20:40, 20:40] = 1.0

        prompts = {"bboxes": [[15, 15, 45, 45]], "has_tumor": True}
        refined = refiner.refine_single_slice(img_slice, u_mask, prompts)

        self.assertEqual(refined.shape, (64, 64))


if __name__ == "__main__":
    unittest.main()
