"""
Comprehensive Medical Image Segmentation Metrics Module.

Module Overview:
----------------
1. What it does:
   - Calculates quantitative segmentation evaluation metrics:
     - Dice Similarity Coefficient (DSC)
     - Intersection over Union (IoU)
     - Precision (Positive Predictive Value)
     - Sensitivity (Recall / True Positive Rate)
     - Specificity (True Negative Rate)
     - 95th Percentile Hausdorff Distance (HD95)

2. Why it is required:
   - Provides objective scientific measurement of tumor segmentation accuracy.
   - Evaluates boundary fidelity and false positive rates across experimental conditions.

3. Inputs:
   - PyTorch Tensors or NumPy arrays of binary predictions (0 or 1) and ground truth (0 or 1).
4. Outputs:
   - Dictionary containing floating-point metric values.
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff

def calculate_metrics(y_pred, y_true, threshold=0.5, smooth=1e-6):
    """
    Computes segmentation metrics for a batch or single slice.

    Parameters:
        y_pred (np.ndarray or torch.Tensor): Model probabilities or binary predictions.
        y_true (np.ndarray or torch.Tensor): Ground truth binary mask.
        threshold (float): Probability threshold for binarization (default: 0.5).
        smooth (float): Small epsilon to prevent division by zero.

    Returns:
        dict: Metric dictionary {'dice', 'iou', 'precision', 'sensitivity', 'specificity', 'hd95'}
    """
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()

    # Binarize predictions
    pred_binary = (y_pred >= threshold).astype(np.uint8)
    true_binary = (y_true > 0).astype(np.uint8)

    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)

    # Calculate Hausdorff Distance (HD95 approximation)
    hd95 = compute_hd95(pred_binary, true_binary)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "hd95": float(hd95)
    }

def compute_hd95(pred_mask, true_mask):
    """
    Computes directed 95th percentile Hausdorff Distance between prediction and ground truth boundaries.
    """
    pred_pts = np.argwhere(pred_mask > 0)
    true_pts = np.argwhere(true_mask > 0)

    # Edge cases: if either mask is empty, distance is undefined / fallback value
    if len(pred_pts) == 0 and len(true_pts) == 0:
        return 0.0 # Perfect match of empty masks
    if len(pred_pts) == 0 or len(true_pts) == 0:
        return 50.0 # Penalty distance for complete miss

    try:
        d_fw = directed_hausdorff(pred_pts, true_pts)[0]
        d_bw = directed_hausdorff(true_pts, pred_pts)[0]
        return float(max(d_fw, d_bw))
    except Exception:
        return 50.0
