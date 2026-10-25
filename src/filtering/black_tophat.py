"""
Morphological Black Top-Hat Filter Module for Dark Feature Isolation.

Module Overview:
----------------
1. What it does:
   - Computes the morphological Black Top-Hat transformation: (Closing(I) - I).
   - Isolates dark structures, necrotic core regions, and subtle boundary indentations in MRI scans.

2. Why it is required:
   - Necrotic tumor centers and hypointense lesions are often darker than surrounding edema or tissue.
   - Black Top-Hat transformation enhances dark structures that are smaller than the structuring element, boosting low-contrast tumor boundaries.

3. Inputs:
   - Single-channel 2D MRI slice (NumPy float32 or uint8 array).
4. Outputs:
   - Morphologically enhanced feature map of identical shape.
"""

import cv2
import numpy as np

class BlackTopHatFilterModule:
    """
    Morphological Black Top-Hat filter module.
    """
    def __init__(self, kernel_size=5, kernel_shape="ellipse"):
        """
        Parameters:
            kernel_size (int): Size of morphological structuring element (e.g., 3, 5, 7).
            kernel_shape (str): 'ellipse', 'rect', or 'cross'.
        """
        self.kernel_size = kernel_size
        self.kernel_shape = kernel_shape
        self.structuring_element = self._get_structuring_element()

    def _get_structuring_element(self):
        shape_map = {
            "ellipse": cv2.MORPH_ELLIPSE,
            "rect": cv2.MORPH_RECT,
            "cross": cv2.MORPH_CROSS
        }
        shape_flag = shape_map.get(self.kernel_shape.lower(), cv2.MORPH_ELLIPSE)
        return cv2.getStructuringElement(shape_flag, (self.kernel_size, self.kernel_size))

    def apply(self, img_slice):
        """
        Applies morphological Black Top-Hat filter to 2D numpy array.

        Input:
            img_slice (np.ndarray): Shape (H, W), dtype float32
        Output:
            np.ndarray: Top-Hat feature map of shape (H, W)
        """
        min_val, max_val = img_slice.min(), img_slice.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(img_slice)

        norm_img = ((img_slice - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        tophat_uint8 = cv2.morphologyEx(norm_img, cv2.MORPH_BLACKHAT, self.structuring_element)

        tophat_float = (tophat_uint8.astype(np.float32) / 255.0) * (max_val - min_val)
        return tophat_float
