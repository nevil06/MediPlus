# MONAI Inference Module - Advanced Inference Strategies
# Sliding window inference, test-time augmentation, and model ensembling

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Sequence, Any
from PIL import Image
import io

try:
    from monai.inferers import (
        SlidingWindowInferer,
        SimpleInferer,
        SlidingWindowInfererAdapt,
    )
    from monai.transforms import (
        Compose,
        Activations,
        AsDiscrete,
        KeepLargestConnectedComponent,
        EnsureType,
    )
    from monai.data import decollate_batch
    MONAI_INFERER_AVAILABLE = True
except ImportError:
    MONAI_INFERER_AVAILABLE = False


class MONAIInference:
    """Advanced inference utilities for medical imaging"""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.available = MONAI_INFERER_AVAILABLE

    def check_availability(self):
        """Check if MONAI inferers are available"""
        if not self.available:
            raise ImportError("MONAI inferers not available. Install with: pip install monai")
        return True

    # ==================== SLIDING WINDOW INFERENCE ====================

    def get_sliding_window_inferer(
        self,
        roi_size: Tuple[int, ...],
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        mode: str = 'gaussian',
        sigma_scale: float = 0.125,
        padding_mode: str = 'constant',
        cval: float = 0.0,
        sw_device: Optional[str] = None,
        device: Optional[str] = None,
        progress: bool = False,
    ) -> 'SlidingWindowInferer':
        """
        Get sliding window inferer for processing large images.

        Args:
            roi_size: Size of sliding window (e.g., (96, 96, 96) for 3D)
            sw_batch_size: Number of windows to process at once
            overlap: Overlap between windows (0.0 to 1.0)
            mode: Blending mode ('constant' or 'gaussian')
            sigma_scale: Sigma for Gaussian blending
            padding_mode: Padding mode for border handling
            cval: Constant value for padding
            sw_device: Device for window processing
            device: Device for final output
            progress: Show progress bar

        Returns:
            Configured SlidingWindowInferer
        """
        self.check_availability()

        return SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode=mode,
            sigma_scale=sigma_scale,
            padding_mode=padding_mode,
            cval=cval,
            sw_device=sw_device or self.device,
            device=device or self.device,
            progress=progress,
        )

    def sliding_window_predict(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        roi_size: Tuple[int, ...],
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        mode: str = 'gaussian',
    ) -> torch.Tensor:
        """
        Run sliding window inference on inputs.

        Args:
            model: PyTorch model
            inputs: Input tensor [B, C, H, W, D] or [B, C, H, W]
            roi_size: Size of sliding window
            sw_batch_size: Batch size for window processing
            overlap: Window overlap
            mode: Blending mode

        Returns:
            Model outputs with same spatial dimensions as inputs
        """
        inferer = self.get_sliding_window_inferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode=mode,
        )

        model.eval()
        with torch.no_grad():
            outputs = inferer(inputs, model)

        return outputs

    # ==================== TEST-TIME AUGMENTATION ====================

    def predict_with_tta(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        spatial_dims: int = 3,
        flips: List[int] = None,
        rotations: List[int] = None,
    ) -> torch.Tensor:
        """
        Predict with test-time augmentation (flipping and rotation).

        Args:
            model: PyTorch model
            inputs: Input tensor
            spatial_dims: Number of spatial dimensions (2 or 3)
            flips: List of axes to flip (e.g., [0, 1] for 2D, [0, 1, 2] for 3D)
            rotations: List of rotation amounts (in 90-degree increments)

        Returns:
            Averaged predictions from all augmentations
        """
        if flips is None:
            flips = list(range(spatial_dims))

        model.eval()
        predictions = []

        with torch.no_grad():
            # Original prediction
            pred = model(inputs)
            predictions.append(pred)

            # Flip augmentations
            for flip_axis in flips:
                # Adjust axis for batch and channel dims
                actual_axis = flip_axis + 2

                # Forward with flip
                flipped_input = torch.flip(inputs, [actual_axis])
                flipped_pred = model(flipped_input)

                # Flip prediction back
                unflipped_pred = torch.flip(flipped_pred, [actual_axis])
                predictions.append(unflipped_pred)

            # Rotation augmentations (90-degree rotations for 2D)
            if spatial_dims == 2 and rotations:
                for k in rotations:
                    rotated_input = torch.rot90(inputs, k=k, dims=[2, 3])
                    rotated_pred = model(rotated_input)
                    unrotated_pred = torch.rot90(rotated_pred, k=-k, dims=[2, 3])
                    predictions.append(unrotated_pred)

        # Average all predictions
        avg_pred = torch.stack(predictions).mean(dim=0)
        return avg_pred

    # ==================== MODEL ENSEMBLING ====================

    def predict_ensemble(
        self,
        models: List[nn.Module],
        inputs: torch.Tensor,
        weights: Optional[List[float]] = None,
        method: str = 'average',
    ) -> torch.Tensor:
        """
        Ensemble prediction from multiple models.

        Args:
            models: List of PyTorch models
            inputs: Input tensor
            weights: Optional weights for each model
            method: Ensemble method ('average', 'weighted', 'vote')

        Returns:
            Ensemble prediction
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")

        predictions = []

        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(inputs)
                predictions.append(pred)

        if method == 'average':
            return torch.stack(predictions).mean(dim=0)

        elif method == 'weighted':
            weighted_preds = [w * p for w, p in zip(weights, predictions)]
            return sum(weighted_preds)

        elif method == 'vote':
            # Hard voting (for classification)
            votes = torch.stack([p.argmax(dim=1) for p in predictions])
            return torch.mode(votes, dim=0).values

        else:
            raise ValueError(f"Unknown ensemble method: {method}")


class SlidingWindowProcessor:
    """
    High-level processor for large medical images using sliding windows.

    Handles the complete pipeline: preprocessing, inference, and postprocessing.
    """

    def __init__(
        self,
        model: nn.Module,
        roi_size: Tuple[int, ...],
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        mode: str = 'gaussian',
        device: str = 'cpu',
        num_classes: int = 2,
        threshold: float = 0.5,
    ):
        """
        Initialize the sliding window processor.

        Args:
            model: PyTorch segmentation model
            roi_size: Size of sliding window
            sw_batch_size: Batch size for window processing
            overlap: Window overlap (0.0 to 1.0)
            mode: Blending mode ('constant' or 'gaussian')
            device: Device for computation
            num_classes: Number of output classes
            threshold: Threshold for binary segmentation
        """
        self.model = model.to(device)
        self.device = device
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.num_classes = num_classes
        self.threshold = threshold

        if MONAI_INFERER_AVAILABLE:
            self.inferer = SlidingWindowInferer(
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                mode=mode,
            )
        else:
            self.inferer = None

        # Post-processing transforms
        self._setup_postprocessing()

    def _setup_postprocessing(self):
        """Setup post-processing transforms"""
        if MONAI_INFERER_AVAILABLE:
            if self.num_classes == 2:
                # Binary segmentation
                self.post_transforms = Compose([
                    Activations(sigmoid=True),
                    AsDiscrete(threshold=self.threshold),
                ])
            else:
                # Multi-class segmentation
                self.post_transforms = Compose([
                    Activations(softmax=True),
                    AsDiscrete(argmax=True),
                ])
        else:
            self.post_transforms = None

    def predict(
        self,
        inputs: torch.Tensor,
        return_probabilities: bool = False,
        keep_largest: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Run sliding window inference.

        Args:
            inputs: Input tensor [B, C, H, W, D] or [B, C, H, W]
            return_probabilities: Also return probability maps
            keep_largest: Keep only the largest connected component

        Returns:
            Dictionary with 'segmentation' and optionally 'probabilities'
        """
        if self.inferer is None:
            raise RuntimeError("MONAI inferer not available")

        inputs = inputs.to(self.device)

        self.model.eval()
        with torch.no_grad():
            # Run sliding window inference
            logits = self.inferer(inputs, self.model)

            # Apply activation
            if self.num_classes == 2:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)

            # Get segmentation
            if self.num_classes == 2:
                seg = (probs > self.threshold).float()
            else:
                seg = probs.argmax(dim=1, keepdim=True).float()

        results = {'segmentation': seg}

        if return_probabilities:
            results['probabilities'] = probs

        return results

    def predict_numpy(
        self,
        inputs: np.ndarray,
        return_probabilities: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on numpy array.

        Args:
            inputs: Numpy array [B, C, H, W, D] or [B, C, H, W]
            return_probabilities: Also return probability maps

        Returns:
            Dictionary with numpy arrays
        """
        # Convert to tensor
        tensor_input = torch.from_numpy(inputs).float()

        # Add batch dimension if needed
        if tensor_input.ndim == 3:
            tensor_input = tensor_input.unsqueeze(0).unsqueeze(0)
        elif tensor_input.ndim == 4:
            tensor_input = tensor_input.unsqueeze(0)

        # Run inference
        results = self.predict(tensor_input, return_probabilities)

        # Convert back to numpy
        return {k: v.cpu().numpy() for k, v in results.items()}


class ClassificationInferer:
    """
    Inference utilities for medical image classification.

    Handles single images and batches with optional test-time augmentation.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize classification inferer.

        Args:
            model: PyTorch classification model
            device: Device for computation
            class_names: Optional list of class names
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names

    def predict(
        self,
        inputs: torch.Tensor,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run classification inference.

        Args:
            inputs: Input tensor [B, C, H, W]
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions, probabilities, and class names
        """
        inputs = inputs.to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, min(top_k, probs.shape[1]), dim=1)

        results = {
            'predictions': top_indices.cpu().numpy(),
            'probabilities': top_probs.cpu().numpy(),
            'all_probabilities': probs.cpu().numpy(),
        }

        # Add class names if available
        if self.class_names:
            batch_size = top_indices.shape[0]
            results['class_names'] = [
                [self.class_names[idx] for idx in top_indices[b].cpu().numpy()]
                for b in range(batch_size)
            ]

        return results

    def predict_with_tta(
        self,
        inputs: torch.Tensor,
        top_k: int = 5,
        flips: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Classification with test-time augmentation.

        Args:
            inputs: Input tensor [B, C, H, W]
            top_k: Number of top predictions
            flips: Flip axes for augmentation

        Returns:
            Averaged predictions across augmentations
        """
        if flips is None:
            flips = [0, 1]  # Horizontal and vertical flips

        inputs = inputs.to(self.device)
        all_probs = []

        self.model.eval()
        with torch.no_grad():
            # Original
            logits = self.model(inputs)
            all_probs.append(torch.softmax(logits, dim=1))

            # Flip augmentations
            for flip_axis in flips:
                actual_axis = flip_axis + 2
                flipped = torch.flip(inputs, [actual_axis])
                logits = self.model(flipped)
                all_probs.append(torch.softmax(logits, dim=1))

        # Average probabilities
        avg_probs = torch.stack(all_probs).mean(dim=0)
        top_probs, top_indices = torch.topk(avg_probs, min(top_k, avg_probs.shape[1]), dim=1)

        results = {
            'predictions': top_indices.cpu().numpy(),
            'probabilities': top_probs.cpu().numpy(),
            'all_probabilities': avg_probs.cpu().numpy(),
        }

        if self.class_names:
            batch_size = top_indices.shape[0]
            results['class_names'] = [
                [self.class_names[idx] for idx in top_indices[b].cpu().numpy()]
                for b in range(batch_size)
            ]

        return results

    def predict_from_pil(
        self,
        image: Image.Image,
        transform: Callable,
        top_k: int = 5,
        use_tta: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict from PIL Image.

        Args:
            image: PIL Image
            transform: Preprocessing transform
            top_k: Number of top predictions
            use_tta: Use test-time augmentation

        Returns:
            Prediction results
        """
        # Preprocess
        img_array = np.array(image)
        tensor = transform(img_array)

        # Add batch dimension
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # Predict
        if use_tta:
            return self.predict_with_tta(tensor, top_k=top_k)
        else:
            return self.predict(tensor, top_k=top_k)

    def predict_from_bytes(
        self,
        image_bytes: bytes,
        transform: Callable,
        top_k: int = 5,
        use_tta: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict from image bytes.

        Args:
            image_bytes: Raw image bytes
            transform: Preprocessing transform
            top_k: Number of top predictions
            use_tta: Use test-time augmentation

        Returns:
            Prediction results
        """
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return self.predict_from_pil(image, transform, top_k, use_tta)
