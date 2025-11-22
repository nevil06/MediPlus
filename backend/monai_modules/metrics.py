# MONAI Metrics Module - Medical Image Analysis Evaluation
# Comprehensive metrics for segmentation, classification, and detection tasks

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

try:
    from monai.metrics import (
        DiceMetric,
        HausdorffDistanceMetric,
        SurfaceDistanceMetric,
        MeanIoU,
        ConfusionMatrixMetric,
        ROCAUCMetric,
        CumulativeIterationMetric,
        compute_dice,
        compute_hausdorff_distance,
        compute_average_surface_distance,
        compute_surface_dice,
        compute_iou,
        compute_confusion_matrix_metric,
    )
    MONAI_METRICS_AVAILABLE = True
except ImportError:
    MONAI_METRICS_AVAILABLE = False


class MONAIMetrics:
    """Comprehensive medical imaging metrics using MONAI"""

    def __init__(self):
        self.available = MONAI_METRICS_AVAILABLE

    def check_availability(self):
        """Check if MONAI metrics are available"""
        if not self.available:
            raise ImportError("MONAI metrics not available. Install with: pip install monai")
        return True

    # ==================== SEGMENTATION METRICS ====================

    def get_dice_metric(
        self,
        include_background: bool = True,
        reduction: str = 'mean',
        get_not_nans: bool = True,
        ignore_empty: bool = True,
        num_classes: Optional[int] = None,
    ) -> 'DiceMetric':
        """
        Dice Similarity Coefficient Metric.

        Best for: Standard segmentation evaluation
        Range: [0, 1] where 1 is perfect overlap
        """
        self.check_availability()
        return DiceMetric(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
            ignore_empty=ignore_empty,
            num_classes=num_classes,
        )

    def get_hausdorff_metric(
        self,
        include_background: bool = False,
        percentile: Optional[float] = None,
        directed: bool = False,
        reduction: str = 'mean',
        get_not_nans: bool = True,
    ) -> 'HausdorffDistanceMetric':
        """
        Hausdorff Distance Metric.

        Best for: Evaluating boundary accuracy
        - percentile: Use HD95 (95th percentile) for robustness to outliers
        - Lower is better
        """
        self.check_availability()
        return HausdorffDistanceMetric(
            include_background=include_background,
            percentile=percentile,
            directed=directed,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    def get_surface_distance_metric(
        self,
        include_background: bool = False,
        symmetric: bool = True,
        distance_metric: str = 'euclidean',
        reduction: str = 'mean',
        get_not_nans: bool = True,
    ) -> 'SurfaceDistanceMetric':
        """
        Average Surface Distance Metric.

        Best for: Evaluating surface/boundary accuracy
        Lower is better
        """
        self.check_availability()
        return SurfaceDistanceMetric(
            include_background=include_background,
            symmetric=symmetric,
            distance_metric=distance_metric,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    def get_mean_iou_metric(
        self,
        include_background: bool = True,
        reduction: str = 'mean',
        get_not_nans: bool = True,
        ignore_empty: bool = True,
        num_classes: Optional[int] = None,
    ) -> 'MeanIoU':
        """
        Mean Intersection over Union (Jaccard Index).

        Best for: Standard segmentation evaluation, similar to Dice
        Range: [0, 1] where 1 is perfect overlap
        """
        self.check_availability()
        return MeanIoU(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
            ignore_empty=ignore_empty,
            num_classes=num_classes,
        )

    def get_confusion_matrix_metric(
        self,
        include_background: bool = True,
        metric_name: str = 'f1 score',
        compute_sample: bool = True,
        reduction: str = 'mean',
        get_not_nans: bool = True,
    ) -> 'ConfusionMatrixMetric':
        """
        Confusion Matrix-based Metrics.

        Available metrics: 'sensitivity', 'specificity', 'precision', 'negative predictive value',
        'miss rate', 'fall out', 'false discovery rate', 'false omission rate',
        'prevalence threshold', 'threat score', 'accuracy', 'balanced accuracy',
        'f1 score', 'matthews correlation coefficient', 'fowlkes mallows index',
        'informedness', 'markedness'
        """
        self.check_availability()
        return ConfusionMatrixMetric(
            include_background=include_background,
            metric_name=metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    def get_roc_auc_metric(
        self,
        average: str = 'macro',
    ) -> 'ROCAUCMetric':
        """
        ROC AUC Metric for classification.

        average: 'macro', 'micro', 'weighted', or 'none'
        Range: [0, 1] where 1 is perfect discrimination
        """
        self.check_availability()
        return ROCAUCMetric(average=average)

    # ==================== DIRECT COMPUTATION FUNCTIONS ====================

    def compute_dice(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        include_background: bool = True,
    ) -> torch.Tensor:
        """Compute Dice coefficient directly"""
        self.check_availability()
        return compute_dice(y_pred, y, include_background=include_background)

    def compute_hausdorff(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        include_background: bool = False,
        percentile: float = 95.0,
    ) -> torch.Tensor:
        """Compute Hausdorff Distance directly (HD95 by default)"""
        self.check_availability()
        return compute_hausdorff_distance(
            y_pred, y,
            include_background=include_background,
            percentile=percentile
        )

    def compute_asd(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        include_background: bool = False,
    ) -> torch.Tensor:
        """Compute Average Surface Distance directly"""
        self.check_availability()
        return compute_average_surface_distance(
            y_pred, y,
            include_background=include_background,
            symmetric=True
        )

    def compute_iou(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        include_background: bool = True,
    ) -> torch.Tensor:
        """Compute IoU (Jaccard) directly"""
        self.check_availability()
        return compute_iou(y_pred, y, include_background=include_background)

    def compute_surface_dice(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        class_thresholds: List[float],
        include_background: bool = False,
    ) -> torch.Tensor:
        """
        Compute Surface Dice (Normalized Surface Distance) directly.

        Measures the overlap of surfaces within tolerance thresholds.
        """
        self.check_availability()
        return compute_surface_dice(
            y_pred, y,
            class_thresholds=class_thresholds,
            include_background=include_background
        )


class MetricsAggregator:
    """Aggregate and track metrics during training/validation"""

    def __init__(
        self,
        metrics: Dict[str, Any],
        device: str = 'cpu'
    ):
        """
        Args:
            metrics: Dictionary of metric name to MONAI metric instance
            device: Device for tensor operations
        """
        self.metrics = metrics
        self.device = device
        self.reset()

    def reset(self):
        """Reset all metrics"""
        for metric in self.metrics.values():
            if hasattr(metric, 'reset'):
                metric.reset()

    def update(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor
    ):
        """Update all metrics with new predictions"""
        for metric in self.metrics.values():
            metric(y_pred=y_pred, y=y)

    def compute(self) -> Dict[str, float]:
        """Compute and return all metric values"""
        results = {}
        for name, metric in self.metrics.items():
            value = metric.aggregate()
            if isinstance(value, torch.Tensor):
                value = value.item()
            results[name] = value
        return results


# ==================== CONVENIENCE FUNCTIONS ====================

def get_metrics(
    task: str,
    num_classes: int = 2,
    include_background: bool = True,
) -> Dict[str, Any]:
    """
    Get a comprehensive set of metrics for a specific task.

    Args:
        task: One of:
            - 'binary_segmentation': Binary segmentation metrics
            - 'multi_class_segmentation': Multi-class segmentation
            - 'classification': Classification metrics
            - 'organ_segmentation': Organ segmentation (boundary-focused)
            - 'lesion_detection': Lesion detection (small object focused)

    Returns:
        Dictionary of metric name to MONAI metric instance
    """
    metrics_helper = MONAIMetrics()

    task_metrics = {
        'binary_segmentation': {
            'dice': metrics_helper.get_dice_metric(
                include_background=include_background,
                reduction='mean_batch',
            ),
            'iou': metrics_helper.get_mean_iou_metric(
                include_background=include_background,
                reduction='mean_batch',
            ),
            'sensitivity': metrics_helper.get_confusion_matrix_metric(
                metric_name='sensitivity',
                include_background=include_background,
            ),
            'specificity': metrics_helper.get_confusion_matrix_metric(
                metric_name='specificity',
                include_background=include_background,
            ),
        },

        'multi_class_segmentation': {
            'dice': metrics_helper.get_dice_metric(
                include_background=include_background,
                reduction='mean_batch',
                num_classes=num_classes,
            ),
            'iou': metrics_helper.get_mean_iou_metric(
                include_background=include_background,
                reduction='mean_batch',
                num_classes=num_classes,
            ),
            'accuracy': metrics_helper.get_confusion_matrix_metric(
                metric_name='accuracy',
                include_background=include_background,
            ),
        },

        'classification': {
            'accuracy': metrics_helper.get_confusion_matrix_metric(
                metric_name='accuracy',
                include_background=True,
            ),
            'f1_score': metrics_helper.get_confusion_matrix_metric(
                metric_name='f1 score',
                include_background=True,
            ),
            'precision': metrics_helper.get_confusion_matrix_metric(
                metric_name='precision',
                include_background=True,
            ),
            'recall': metrics_helper.get_confusion_matrix_metric(
                metric_name='sensitivity',
                include_background=True,
            ),
            'roc_auc': metrics_helper.get_roc_auc_metric(average='macro'),
        },

        'organ_segmentation': {
            'dice': metrics_helper.get_dice_metric(
                include_background=False,  # Exclude background for organs
                reduction='mean_batch',
                num_classes=num_classes,
            ),
            'hd95': metrics_helper.get_hausdorff_metric(
                include_background=False,
                percentile=95.0,
            ),
            'asd': metrics_helper.get_surface_distance_metric(
                include_background=False,
                symmetric=True,
            ),
        },

        'lesion_detection': {
            'dice': metrics_helper.get_dice_metric(
                include_background=False,
                reduction='mean_batch',
            ),
            'sensitivity': metrics_helper.get_confusion_matrix_metric(
                metric_name='sensitivity',
                include_background=False,
            ),
            'precision': metrics_helper.get_confusion_matrix_metric(
                metric_name='precision',
                include_background=False,
            ),
            'f1_score': metrics_helper.get_confusion_matrix_metric(
                metric_name='f1 score',
                include_background=False,
            ),
        },
    }

    if task not in task_metrics:
        raise ValueError(f"Unknown task: {task}. Choose from {list(task_metrics.keys())}")

    return task_metrics[task]


# ==================== CUSTOM MEDICAL METRICS ====================

class VolumeMetric:
    """
    Volume-based metrics for medical imaging.

    Computes volume measurements in physical units (mm³).
    """

    def __init__(self, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Args:
            spacing: Voxel spacing in mm (x, y, z)
        """
        self.spacing = spacing
        self.voxel_volume = spacing[0] * spacing[1] * spacing[2]

    def compute_volume(self, segmentation: torch.Tensor) -> float:
        """Compute volume in mm³"""
        voxel_count = segmentation.sum().item()
        return voxel_count * self.voxel_volume

    def compute_volume_error(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """Compute volume difference metrics"""
        pred_volume = self.compute_volume(y_pred)
        true_volume = self.compute_volume(y)

        absolute_error = abs(pred_volume - true_volume)
        relative_error = absolute_error / true_volume if true_volume > 0 else float('inf')

        return {
            'predicted_volume_mm3': pred_volume,
            'true_volume_mm3': true_volume,
            'absolute_error_mm3': absolute_error,
            'relative_error': relative_error,
        }


class CompactnessMetric:
    """
    Compactness/Sphericity metric for shape analysis.

    Measures how compact/spherical a shape is.
    Range: [0, 1] where 1 is a perfect sphere.
    """

    def __init__(self, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        self.spacing = spacing

    def compute(self, segmentation: torch.Tensor) -> float:
        """
        Compute compactness = 36π * V² / S³
        where V is volume and S is surface area.
        """
        import math

        # This is a simplified implementation
        # Full implementation would require surface area computation
        volume = segmentation.sum().item()

        if volume == 0:
            return 0.0

        # Approximate surface area (simplified)
        # For accurate computation, use marching cubes or similar
        # This approximation uses boundary voxels
        from scipy import ndimage
        seg_np = segmentation.cpu().numpy()
        eroded = ndimage.binary_erosion(seg_np)
        boundary = seg_np.astype(float) - eroded.astype(float)
        surface_area = boundary.sum()

        if surface_area == 0:
            return 0.0

        compactness = (36 * math.pi * volume ** 2) / (surface_area ** 3)
        return min(compactness, 1.0)


class FragmentationMetric:
    """
    Fragmentation metric for detecting disconnected regions.

    Lower fragmentation indicates better connected segmentation.
    """

    def compute(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute fragmentation metrics.

        Returns number of connected components in prediction vs ground truth.
        """
        from scipy import ndimage

        pred_np = y_pred.cpu().numpy().astype(bool)
        true_np = y.cpu().numpy().astype(bool)

        _, pred_num_components = ndimage.label(pred_np)
        _, true_num_components = ndimage.label(true_np)

        return {
            'pred_components': pred_num_components,
            'true_components': true_num_components,
            'fragmentation_ratio': pred_num_components / max(true_num_components, 1),
        }


def compute_comprehensive_metrics(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    task: str = 'binary_segmentation',
    spacing: Optional[Tuple[float, ...]] = None,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of metrics for evaluation.

    Args:
        y_pred: Predicted segmentation/classification
        y: Ground truth
        task: Task type for metric selection
        spacing: Voxel spacing for volume metrics (optional)
        num_classes: Number of classes

    Returns:
        Dictionary of metric names to values
    """
    results = {}

    # Get task-specific metrics
    if MONAI_METRICS_AVAILABLE:
        metrics = get_metrics(task, num_classes=num_classes)
        for name, metric in metrics.items():
            try:
                metric(y_pred=y_pred, y=y)
                value = metric.aggregate()
                if isinstance(value, torch.Tensor):
                    value = value.item()
                results[name] = value
            except Exception as e:
                results[name] = float('nan')
                print(f"Warning: Could not compute {name}: {e}")

    # Add volume metrics if spacing provided and segmentation task
    if spacing is not None and 'segmentation' in task:
        volume_metric = VolumeMetric(spacing=spacing)
        volume_results = volume_metric.compute_volume_error(y_pred, y)
        results.update(volume_results)

    return results
