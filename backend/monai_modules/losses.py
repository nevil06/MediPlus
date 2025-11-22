# MONAI Losses Module - Medical-Specific Loss Functions
# Optimized loss functions for medical image analysis tasks

import torch
import torch.nn as nn
from typing import Optional, Sequence, Union, Callable

try:
    from monai.losses import (
        DiceLoss,
        DiceCELoss,
        DiceFocalLoss,
        FocalLoss,
        TverskyLoss,
        GeneralizedDiceLoss,
        GeneralizedWassersteinDiceLoss,
        MaskedLoss,
        ContrastiveLoss,
        LocalNormalizedCrossCorrelationLoss,
        GlobalMutualInformationLoss,
    )
    MONAI_LOSSES_AVAILABLE = True
except ImportError:
    MONAI_LOSSES_AVAILABLE = False


class MONAILosses:
    """Comprehensive medical imaging loss functions using MONAI"""

    def __init__(self):
        self.available = MONAI_LOSSES_AVAILABLE

    def check_availability(self):
        """Check if MONAI losses are available"""
        if not self.available:
            raise ImportError("MONAI losses not available. Install with: pip install monai")
        return True

    # ==================== SEGMENTATION LOSSES ====================

    def get_dice_loss(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = 'mean',
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> nn.Module:
        """
        Dice Loss for segmentation tasks.

        Best for: Binary and multi-class segmentation with class imbalance
        """
        self.check_availability()
        return DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )

    def get_dice_ce_loss(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = 'mean',
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> nn.Module:
        """
        Combined Dice + Cross-Entropy Loss.

        Best for: Most segmentation tasks - combines region-based and
        distribution-based optimization
        """
        self.check_availability()
        return DiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            ce_weight=ce_weight,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )

    def get_dice_focal_loss(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = 'mean',
        gamma: float = 2.0,
        focal_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
    ) -> nn.Module:
        """
        Combined Dice + Focal Loss.

        Best for: Segmentation with severe class imbalance (e.g., small lesions)
        """
        self.check_availability()
        return DiceFocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            gamma=gamma,
            focal_weight=focal_weight,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
        )

    def get_focal_loss(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        use_softmax: bool = False,
    ) -> nn.Module:
        """
        Focal Loss for handling class imbalance.

        Best for: Classification with class imbalance, hard example mining
        """
        self.check_availability()
        return FocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            gamma=gamma,
            alpha=alpha,
            weight=weight,
            reduction=reduction,
            use_softmax=use_softmax,
        )

    def get_tversky_loss(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        alpha: float = 0.5,
        beta: float = 0.5,
        reduction: str = 'mean',
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> nn.Module:
        """
        Tversky Loss for controlling precision/recall trade-off.

        Best for: Segmentation where false positives/negatives have different costs
        - alpha > 0.5: penalize false positives more (higher precision)
        - beta > 0.5: penalize false negatives more (higher recall)
        """
        self.check_availability()
        return TverskyLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            alpha=alpha,
            beta=beta,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )

    def get_generalized_dice_loss(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        w_type: str = 'square',
        reduction: str = 'mean',
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> nn.Module:
        """
        Generalized Dice Loss with class-level weighting.

        Best for: Multi-class segmentation with varying class sizes
        w_type: 'simple', 'square', or 'uniform' weighting
        """
        self.check_availability()
        return GeneralizedDiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            w_type=w_type,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )

    # ==================== REGISTRATION LOSSES ====================

    def get_lncc_loss(
        self,
        spatial_dims: int = 3,
        kernel_size: int = 3,
        kernel_type: str = 'rectangular',
        reduction: str = 'mean',
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
    ) -> nn.Module:
        """
        Local Normalized Cross-Correlation Loss.

        Best for: Image registration, deformable registration
        """
        self.check_availability()
        return LocalNormalizedCrossCorrelationLoss(
            spatial_dims=spatial_dims,
            kernel_size=kernel_size,
            kernel_type=kernel_type,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )

    def get_mutual_information_loss(
        self,
        num_bins: int = 23,
        sigma_ratio: float = 0.5,
        reduction: str = 'mean',
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
    ) -> nn.Module:
        """
        Global Mutual Information Loss.

        Best for: Multi-modal image registration
        """
        self.check_availability()
        return GlobalMutualInformationLoss(
            num_bins=num_bins,
            sigma_ratio=sigma_ratio,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )

    # ==================== UTILITY METHODS ====================

    def get_masked_loss(
        self,
        loss: nn.Module,
    ) -> nn.Module:
        """
        Wrapper to apply mask to any loss function.

        Best for: Ignoring certain regions during training (e.g., unknown labels)
        """
        self.check_availability()
        return MaskedLoss(loss)

    def get_contrastive_loss(
        self,
        temperature: float = 0.5,
    ) -> nn.Module:
        """
        Contrastive Loss for self-supervised learning.

        Best for: Pre-training on unlabeled medical images
        """
        self.check_availability()
        return ContrastiveLoss(temperature=temperature)


class CombinedLoss(nn.Module):
    """Combine multiple losses with custom weights"""

    def __init__(
        self,
        losses: Sequence[nn.Module],
        weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)

        if len(self.losses) != len(self.weights):
            raise ValueError("Number of losses must match number of weights")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(input, target)
        return total_loss


# ==================== PRESET LOSS CONFIGURATIONS ====================

def get_loss_function(
    task: str,
    num_classes: int = 2,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Get a pre-configured loss function for a specific medical imaging task.

    Args:
        task: One of:
            - 'binary_segmentation': Binary segmentation (e.g., tumor vs background)
            - 'multi_class_segmentation': Multi-class segmentation
            - 'imbalanced_segmentation': Segmentation with severe class imbalance
            - 'organ_segmentation': Organ segmentation with multiple classes
            - 'lesion_detection': Small lesion segmentation
            - 'classification': Image classification
            - 'classification_imbalanced': Classification with class imbalance
            - 'registration': Image registration
        num_classes: Number of classes (for multi-class tasks)
        class_weights: Optional class weights for handling imbalance

    Returns:
        Configured loss function
    """
    losses = MONAILosses()

    task_configs = {
        'binary_segmentation': lambda: losses.get_dice_ce_loss(
            include_background=True,
            sigmoid=True,
            lambda_dice=1.0,
            lambda_ce=0.5,
        ),

        'multi_class_segmentation': lambda: losses.get_dice_ce_loss(
            include_background=True,
            softmax=True,
            to_onehot_y=True,
            ce_weight=class_weights,
            lambda_dice=1.0,
            lambda_ce=1.0,
        ),

        'imbalanced_segmentation': lambda: losses.get_dice_focal_loss(
            include_background=True,
            sigmoid=True if num_classes == 2 else False,
            softmax=False if num_classes == 2 else True,
            to_onehot_y=num_classes > 2,
            gamma=2.5,
            lambda_dice=1.0,
            lambda_focal=1.5,
        ),

        'organ_segmentation': lambda: losses.get_generalized_dice_loss(
            include_background=False,  # Exclude background for organ segmentation
            softmax=True,
            to_onehot_y=True,
            w_type='square',
        ),

        'lesion_detection': lambda: losses.get_tversky_loss(
            include_background=True,
            sigmoid=True,
            alpha=0.3,  # Lower alpha = less FP penalty
            beta=0.7,   # Higher beta = more FN penalty (recall-focused)
        ),

        'classification': lambda: FocalLoss(
            gamma=0.0,  # gamma=0 is standard cross-entropy
            weight=class_weights,
            reduction='mean',
            use_softmax=True,
        ) if MONAI_LOSSES_AVAILABLE else nn.CrossEntropyLoss(weight=class_weights),

        'classification_imbalanced': lambda: losses.get_focal_loss(
            gamma=2.0,
            alpha=0.25,
            weight=class_weights,
            use_softmax=True,
        ),

        'registration': lambda: losses.get_lncc_loss(
            spatial_dims=kwargs.get('spatial_dims', 3),
            kernel_size=kwargs.get('kernel_size', 9),
        ),
    }

    if task not in task_configs:
        raise ValueError(f"Unknown task: {task}. Choose from {list(task_configs.keys())}")

    return task_configs[task]()


# ==================== CUSTOM MEDICAL LOSSES ====================

class BoundaryLoss(nn.Module):
    """
    Boundary Loss for segmentation.

    Encourages the predicted boundary to align with the ground truth boundary.
    Useful for accurate boundary delineation in medical imaging.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, dist_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probability map (after softmax/sigmoid)
            dist_map: Pre-computed signed distance transform of ground truth
        """
        multipled = torch.einsum('bchwd,bchwd->bchwd', pred, dist_map)

        if self.reduction == 'mean':
            return multipled.mean()
        elif self.reduction == 'sum':
            return multipled.sum()
        return multipled


class TopologicalLoss(nn.Module):
    """
    Topology-Aware Loss for preserving connectivity.

    Useful for tubular structures (vessels, airways) where
    topology preservation is critical.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_persistent_features: torch.Tensor,
        target_persistent_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes topological loss based on persistent homology features.

        Note: Requires pre-computed persistent homology features.
        """
        # Simplified version - computes L2 distance between persistent features
        return torch.norm(pred_persistent_features - target_persistent_features, p=2)


class HDLoss(nn.Module):
    """
    Hausdorff Distance Loss approximation.

    Penalizes large boundary deviations, useful for medical structures
    where boundary accuracy is critical.
    """

    def __init__(self, reduction: str = 'mean', alpha: float = 2.0):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_dist_map: torch.Tensor,
        target_dist_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted segmentation (binary or probability)
            target: Ground truth segmentation
            pred_dist_map: Distance transform of prediction
            target_dist_map: Distance transform of ground truth
        """
        # HD loss approximation
        term1 = ((pred - target) ** 2 * (target_dist_map ** self.alpha)).mean()
        term2 = ((pred - target) ** 2 * (pred_dist_map ** self.alpha)).mean()

        return term1 + term2
