"""
Medical Image Segmentation Service
Provides organ, lesion, and structure segmentation using MONAI

Supports:
- Organ segmentation (liver, spleen, kidneys, etc.)
- Tumor/lesion segmentation
- Lung segmentation in chest X-rays
- Cardiac structure segmentation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import base64
from io import BytesIO
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from monai_modules.transforms import MONAITransforms
    from monai_modules.networks import MONAINetworks, get_network
    from monai_modules.inference import SlidingWindowProcessor
    from monai_modules.losses import get_loss_function
    from monai_modules.metrics import get_metrics
    MONAI_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"MONAI modules import error: {e}")
    MONAI_MODULES_AVAILABLE = False

try:
    from services.ai_vision_service import ai_enhancement_service
    AI_ENHANCEMENT_AVAILABLE = True
except ImportError:
    AI_ENHANCEMENT_AVAILABLE = False


class MedicalSegmentationService:
    """
    Advanced medical image segmentation service using MONAI

    Features:
    - Multiple segmentation architectures (UNet, SegResNet, UNETR, SwinUNETR)
    - Sliding window inference for large images
    - Multi-organ segmentation
    - Post-processing with connected component analysis
    """

    # Supported segmentation tasks
    SEGMENTATION_TASKS = {
        'lung_2d': {
            'name': 'Lung Segmentation (2D)',
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 3,  # Background, Left Lung, Right Lung
            'roi_size': (256, 256),
            'classes': ['Background', 'Left Lung', 'Right Lung']
        },
        'cardiac_2d': {
            'name': 'Cardiac Segmentation (2D)',
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 4,
            'roi_size': (224, 224),
            'classes': ['Background', 'Left Ventricle', 'Right Ventricle', 'Myocardium']
        },
        'organ_3d': {
            'name': 'Multi-Organ Segmentation (3D)',
            'spatial_dims': 3,
            'in_channels': 1,
            'out_channels': 14,
            'roi_size': (96, 96, 96),
            'classes': [
                'Background', 'Spleen', 'Right Kidney', 'Left Kidney',
                'Gallbladder', 'Esophagus', 'Liver', 'Stomach',
                'Aorta', 'Inferior Vena Cava', 'Portal Vein',
                'Pancreas', 'Right Adrenal', 'Left Adrenal'
            ]
        },
        'liver_tumor': {
            'name': 'Liver and Tumor Segmentation',
            'spatial_dims': 3,
            'in_channels': 1,
            'out_channels': 3,
            'roi_size': (128, 128, 128),
            'classes': ['Background', 'Liver', 'Tumor']
        },
        'brain_tumor': {
            'name': 'Brain Tumor Segmentation',
            'spatial_dims': 3,
            'in_channels': 4,  # T1, T1ce, T2, FLAIR
            'out_channels': 4,
            'roi_size': (128, 128, 128),
            'classes': ['Background', 'Necrotic Core', 'Edema', 'Enhancing Tumor']
        }
    }

    def __init__(
        self,
        task: str = 'lung_2d',
        architecture: str = 'unet',
        device: str = 'auto'
    ):
        """
        Initialize segmentation service

        Args:
            task: Segmentation task from SEGMENTATION_TASKS
            architecture: Network architecture ('unet', 'segresnet', 'unetr', 'swin_unetr')
            device: Computation device ('auto', 'cpu', 'cuda')
        """
        self.task = task
        self.architecture = architecture
        self.device = self._select_device(device)

        self.task_config = self.SEGMENTATION_TASKS.get(task)
        if self.task_config is None:
            raise ValueError(f"Unknown task: {task}. Choose from {list(self.SEGMENTATION_TASKS.keys())}")

        self.model = None
        self.transforms = None
        self.processor = None
        self._initialize()

    def _select_device(self, device: str) -> str:
        if device == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return 'cuda'
            return 'cpu'
        return device

    def _initialize(self):
        """Initialize model and transforms"""
        if not TORCH_AVAILABLE or not MONAI_MODULES_AVAILABLE:
            print("Required dependencies not available")
            return

        config = self.task_config

        # Get appropriate network
        try:
            networks = MONAINetworks(device=self.device)

            if self.architecture == 'unet':
                self.model = networks.get_unet(
                    spatial_dims=config['spatial_dims'],
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2
                )
            elif self.architecture == 'segresnet':
                self.model = networks.get_segresnet(
                    spatial_dims=config['spatial_dims'],
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels']
                )
            elif self.architecture == 'attention_unet':
                self.model = networks.get_attention_unet(
                    spatial_dims=config['spatial_dims'],
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels']
                )
            elif self.architecture == 'unetr':
                self.model = networks.get_unetr(
                    img_size=config['roi_size'],
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    spatial_dims=config['spatial_dims']
                )
            elif self.architecture == 'swin_unetr':
                self.model = networks.get_swin_unetr(
                    img_size=config['roi_size'],
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    spatial_dims=config['spatial_dims']
                )
            else:
                # Default to UNet
                self.model = networks.get_basic_unet(
                    spatial_dims=config['spatial_dims'],
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels']
                )

            self.model.eval()

            # Create sliding window processor for 3D tasks
            if config['spatial_dims'] == 3:
                self.processor = SlidingWindowProcessor(
                    model=self.model,
                    roi_size=config['roi_size'],
                    sw_batch_size=4,
                    overlap=0.5,
                    device=self.device,
                    num_classes=config['out_channels']
                )

        except Exception as e:
            print(f"Model initialization error: {e}")
            self.model = None

        # Setup transforms
        transforms = MONAITransforms(device=self.device)
        if config['spatial_dims'] == 2:
            if self.task == 'lung_2d':
                self.transforms = transforms.get_chest_xray_transforms(
                    spatial_size=config['roi_size'][:2],
                    is_grayscale=True  # Segmentation uses grayscale images
                )
            elif self.task == 'cardiac_2d':
                self.transforms = transforms.get_chest_xray_transforms(
                    spatial_size=config['roi_size'][:2],
                    is_grayscale=True  # Segmentation uses grayscale images
                )
        else:
            self.transforms = transforms.get_ct_scan_transforms(
                spatial_size=config['roi_size']
            )

    def segment(
        self,
        image_data: str,
        return_overlay: bool = True,
        return_masks: bool = True,
        enhance_with_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Segment medical image

        Args:
            image_data: Base64 encoded image
            return_overlay: Return colored overlay visualization
            return_masks: Return individual class masks
            enhance_with_ai: Use Groq AI for enhanced explanations

        Returns:
            Segmentation results with masks, statistics, and AI analysis
        """
        if self.model is None:
            return self._fallback_result()

        try:
            # Decode image
            image_array = self._decode_image(image_data)
            if image_array is None:
                return self._fallback_result()

            # Run segmentation
            results = self._run_segmentation(image_array)

            # Add visualization
            if return_overlay:
                results['overlay'] = self._create_overlay(
                    image_array,
                    results['segmentation']
                )

            # Add individual masks
            if return_masks:
                results['masks'] = self._extract_masks(results['segmentation'])

            # Compute statistics
            results['statistics'] = self._compute_statistics(results['segmentation'])

            # AI enhancement
            if enhance_with_ai and AI_ENHANCEMENT_AVAILABLE:
                results['ai_analysis'] = ai_enhancement_service.enhance_segmentation_analysis(
                    statistics=results['statistics'],
                    task=self.task
                )

            return results

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _run_segmentation(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Run the segmentation model"""
        config = self.task_config

        # Preprocess
        if self.transforms is not None:
            tensor = self.transforms(image_array)
        else:
            tensor = torch.from_numpy(image_array).float()

        # Add batch dimension
        if tensor.ndim == config['spatial_dims'] + 1:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        # Run inference
        self.model.eval()
        with torch.no_grad():
            if config['spatial_dims'] == 3 and self.processor is not None:
                # Use sliding window for 3D
                outputs = self.processor.predict(tensor)
                segmentation = outputs['segmentation']
            else:
                # Direct inference for 2D
                logits = self.model(tensor)
                if config['out_channels'] == 1:
                    segmentation = (torch.sigmoid(logits) > 0.5).float()
                else:
                    segmentation = logits.argmax(dim=1, keepdim=True).float()

        return {
            'success': True,
            'segmentation': segmentation.cpu().numpy(),
            'classes': config['classes'],
            'task': self.task,
            'architecture': self.architecture
        }

    def _compute_statistics(self, segmentation: np.ndarray) -> Dict[str, Any]:
        """Compute segmentation statistics"""
        config = self.task_config
        stats = {}

        # Remove batch dimension if present
        if segmentation.ndim > config['spatial_dims']:
            seg = segmentation[0, 0]
        else:
            seg = segmentation

        # Compute per-class statistics
        total_pixels = seg.size
        for class_idx, class_name in enumerate(config['classes']):
            class_mask = (seg == class_idx)
            pixel_count = class_mask.sum()

            stats[class_name] = {
                'pixel_count': int(pixel_count),
                'percentage': float(pixel_count / total_pixels * 100),
                'present': pixel_count > 0
            }

        # For 3D volumes, compute volume statistics
        if config['spatial_dims'] == 3:
            # Assuming 1mm isotropic spacing by default
            voxel_volume_mm3 = 1.0
            for class_name in config['classes']:
                if class_name != 'Background':
                    stats[class_name]['volume_mm3'] = stats[class_name]['pixel_count'] * voxel_volume_mm3

        return stats

    def _create_overlay(
        self,
        original: np.ndarray,
        segmentation: np.ndarray,
        alpha: float = 0.5
    ) -> str:
        """Create colored overlay visualization"""
        config = self.task_config

        # Color map for different classes
        colors = [
            [0, 0, 0],        # Background - black
            [255, 0, 0],      # Class 1 - red
            [0, 255, 0],      # Class 2 - green
            [0, 0, 255],      # Class 3 - blue
            [255, 255, 0],    # Class 4 - yellow
            [255, 0, 255],    # Class 5 - magenta
            [0, 255, 255],    # Class 6 - cyan
            [255, 128, 0],    # Class 7 - orange
            [128, 0, 255],    # Class 8 - purple
            [0, 255, 128],    # Class 9 - lime
            [255, 128, 128],  # Class 10 - light red
            [128, 255, 128],  # Class 11 - light green
            [128, 128, 255],  # Class 12 - light blue
            [255, 255, 128],  # Class 13 - light yellow
        ]

        # Get segmentation as 2D/3D array
        seg = segmentation[0, 0] if segmentation.ndim > config['spatial_dims'] else segmentation

        # For 3D, take middle slice
        if config['spatial_dims'] == 3:
            mid_slice = seg.shape[0] // 2
            seg = seg[mid_slice]
            if original.ndim > 2:
                original = original[mid_slice] if original.shape[0] == segmentation.shape[2] else original

        # Normalize original image
        if original.max() > 1:
            original = original / original.max()

        # Create RGB overlay
        overlay = np.stack([original] * 3, axis=-1) if original.ndim == 2 else original
        overlay = (overlay * 255).astype(np.uint8)

        # Apply colors
        for class_idx in range(1, min(len(colors), config['out_channels'])):
            mask = seg == class_idx
            for c in range(3):
                overlay[..., c] = np.where(
                    mask,
                    (1 - alpha) * overlay[..., c] + alpha * colors[class_idx][c],
                    overlay[..., c]
                )

        # Convert to base64
        img = Image.fromarray(overlay.astype(np.uint8))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    def _extract_masks(self, segmentation: np.ndarray) -> Dict[str, str]:
        """Extract individual class masks as base64"""
        config = self.task_config
        masks = {}

        seg = segmentation[0, 0] if segmentation.ndim > config['spatial_dims'] else segmentation

        # For 3D, take middle slice
        if config['spatial_dims'] == 3:
            mid_slice = seg.shape[0] // 2
            seg = seg[mid_slice]

        for class_idx, class_name in enumerate(config['classes']):
            if class_name == 'Background':
                continue

            mask = (seg == class_idx).astype(np.uint8) * 255
            img = Image.fromarray(mask)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            masks[class_name] = base64.b64encode(buffer.getvalue()).decode()

        return masks

    def _decode_image(self, image_data: str) -> Optional[np.ndarray]:
        """Decode base64 image to numpy array"""
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))

            # Convert to grayscale for medical images
            if image.mode != 'L':
                image = image.convert('L')

            return np.array(image)

        except Exception as e:
            print(f"Image decode error: {e}")
            return None

    def _fallback_result(self) -> Dict[str, Any]:
        """Return fallback result when model unavailable"""
        return {
            'success': False,
            'error': 'Segmentation model not available',
            'message': 'Please ensure MONAI and PyTorch are properly installed',
            'available_tasks': list(self.SEGMENTATION_TASKS.keys())
        }

    @classmethod
    def get_available_tasks(cls) -> Dict[str, Dict]:
        """Get all available segmentation tasks"""
        return cls.SEGMENTATION_TASKS

    @classmethod
    def get_available_architectures(cls) -> List[str]:
        """Get all available network architectures"""
        return ['unet', 'basic_unet', 'attention_unet', 'segresnet', 'unetr', 'swin_unetr', 'vnet']


# Pre-configured service instances
lung_segmentation_service = MedicalSegmentationService(task='lung_2d', architecture='unet')
cardiac_segmentation_service = MedicalSegmentationService(task='cardiac_2d', architecture='attention_unet')

# Factory function
def create_segmentation_service(task: str, architecture: str = 'unet') -> MedicalSegmentationService:
    """Create a segmentation service for a specific task"""
    return MedicalSegmentationService(task=task, architecture=architecture)
