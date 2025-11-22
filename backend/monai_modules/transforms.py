# MONAI Transforms Module - Medical-Specific Image Processing
# Provides comprehensive transforms for medical imaging pipelines

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from PIL import Image
import io

try:
    from monai.transforms import (
        # Loading transforms
        LoadImage,
        LoadImaged,
        EnsureChannelFirst,
        EnsureChannelFirstd,

        # Spatial transforms
        Spacing,
        Spacingd,
        Orientation,
        Orientationd,
        Resize,
        Resized,
        RandFlip,
        RandFlipd,
        RandRotate,
        RandRotated,
        RandRotate90,
        RandRotate90d,
        RandZoom,
        RandZoomd,
        RandAffine,
        RandAffined,
        CropForeground,
        CropForegroundd,
        RandCropByPosNegLabel,
        RandCropByPosNegLabeld,
        RandSpatialCrop,
        RandSpatialCropd,
        CenterSpatialCrop,
        CenterSpatialCropd,
        SpatialPad,
        SpatialPadd,

        # Intensity transforms
        ScaleIntensity,
        ScaleIntensityd,
        ScaleIntensityRange,
        ScaleIntensityRanged,
        NormalizeIntensity,
        NormalizeIntensityd,
        RandScaleIntensity,
        RandScaleIntensityd,
        RandShiftIntensity,
        RandShiftIntensityd,
        RandGaussianNoise,
        RandGaussianNoised,
        RandGaussianSmooth,
        RandGaussianSmoothd,
        RandAdjustContrast,
        RandAdjustContrastd,
        RandHistogramShift,
        RandHistogramShiftd,
        ThresholdIntensity,
        ThresholdIntensityd,

        # Utility transforms
        ToTensor,
        ToTensord,
        Compose,
        Identity,
        Lambda,

        # Post-processing
        AsDiscrete,
        AsDiscreted,
        KeepLargestConnectedComponent,
        KeepLargestConnectedComponentd,
        Activations,
        Activationsd,
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


class MONAITransforms:
    """Comprehensive MONAI transforms for medical imaging"""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.available = MONAI_AVAILABLE

    def check_availability(self):
        """Check if MONAI transforms are available"""
        if not self.available:
            raise ImportError("MONAI transforms not available. Install with: pip install monai")
        return True

    # ==================== PREPROCESSING PIPELINES ====================

    def get_chest_xray_transforms(
        self,
        spatial_size: Tuple[int, int] = (224, 224),
        is_training: bool = False,
        is_grayscale: bool = False
    ) -> Callable:
        """Get transforms optimized for chest X-ray analysis

        Args:
            spatial_size: Target spatial dimensions
            is_training: Whether to include augmentation
            is_grayscale: If True, expects (H, W) grayscale input; if False, expects (H, W, C) RGB
        """
        self.check_availability()

        # Handle channel dimension based on input format
        if is_grayscale:
            channel_first = EnsureChannelFirst(channel_dim="no_channel")  # Grayscale (H, W) → (1, H, W)
        else:
            channel_first = EnsureChannelFirst(channel_dim=-1)  # RGB (H, W, C) → (C, H, W)

        base_transforms = [
            channel_first,
            Resize(spatial_size=spatial_size, mode='bilinear'),
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            NormalizeIntensity(subtrahend=0.485, divisor=0.229),  # ImageNet normalization
        ]

        if is_training:
            augmentation_transforms = [
                RandFlip(prob=0.5, spatial_axis=0),  # Horizontal flip
                RandRotate(range_x=0.1, prob=0.3),  # Small rotation
                RandZoom(min_zoom=0.95, max_zoom=1.05, prob=0.3),
                RandGaussianNoise(prob=0.2, mean=0.0, std=0.05),
                RandAdjustContrast(prob=0.3, gamma=(0.9, 1.1)),
            ]
            base_transforms.extend(augmentation_transforms)

        base_transforms.append(ToTensor())
        return Compose(base_transforms)

    def get_skin_lesion_transforms(
        self,
        spatial_size: Tuple[int, int] = (224, 224),
        is_training: bool = False
    ) -> Callable:
        """Get transforms optimized for skin lesion analysis (dermoscopy)"""
        self.check_availability()

        base_transforms = [
            EnsureChannelFirst(channel_dim=-1),  # PIL RGB images are (H, W, C) - channels last
            Resize(spatial_size=spatial_size, mode='bilinear'),
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            NormalizeIntensity(subtrahend=[0.485, 0.456, 0.406], divisor=[0.229, 0.224, 0.225]),
        ]

        if is_training:
            augmentation_transforms = [
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandRotate90(prob=0.5),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
                RandScaleIntensity(factors=0.1, prob=0.5),
                RandShiftIntensity(offsets=0.1, prob=0.5),
                RandGaussianSmooth(sigma_x=(0.5, 1.0), prob=0.2),
            ]
            base_transforms.extend(augmentation_transforms)

        base_transforms.append(ToTensor())
        return Compose(base_transforms)

    def get_fundus_transforms(
        self,
        spatial_size: Tuple[int, int] = (512, 512),
        is_training: bool = False
    ) -> Callable:
        """Get transforms optimized for fundus/retinal image analysis"""
        self.check_availability()

        base_transforms = [
            EnsureChannelFirst(channel_dim=-1),  # PIL RGB images are (H, W, C) - channels last
            Resize(spatial_size=spatial_size, mode='bilinear'),
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            NormalizeIntensity(subtrahend=[0.485, 0.456, 0.406], divisor=[0.229, 0.224, 0.225]),
        ]

        if is_training:
            augmentation_transforms = [
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandRotate(range_x=0.3, prob=0.5),
                RandZoom(min_zoom=0.85, max_zoom=1.15, prob=0.5),
                RandAdjustContrast(prob=0.4, gamma=(0.8, 1.2)),
                RandGaussianNoise(prob=0.2, mean=0.0, std=0.03),
            ]
            base_transforms.extend(augmentation_transforms)

        base_transforms.append(ToTensor())
        return Compose(base_transforms)

    def get_ct_scan_transforms(
        self,
        spatial_size: Tuple[int, int, int] = (96, 96, 96),
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        is_training: bool = False,
        window_center: float = 40,
        window_width: float = 400
    ) -> Callable:
        """Get transforms optimized for CT scan analysis (3D)"""
        self.check_availability()

        # Calculate HU window range
        hu_min = window_center - window_width / 2
        hu_max = window_center + window_width / 2

        base_transforms = [
            EnsureChannelFirst(),
            Spacing(pixdim=spacing, mode='bilinear'),
            Orientation(axcodes='RAS'),
            ScaleIntensityRange(a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True),
            CropForeground(select_fn=lambda x: x > 0),
            Resize(spatial_size=spatial_size, mode='trilinear'),
        ]

        if is_training:
            augmentation_transforms = [
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandFlip(prob=0.5, spatial_axis=2),
                RandRotate90(prob=0.5, max_k=3, spatial_axes=(0, 1)),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
                RandGaussianNoise(prob=0.2, mean=0.0, std=0.02),
                RandScaleIntensity(factors=0.1, prob=0.3),
            ]
            base_transforms.extend(augmentation_transforms)

        base_transforms.append(ToTensor())
        return Compose(base_transforms)

    def get_mri_transforms(
        self,
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        is_training: bool = False
    ) -> Callable:
        """Get transforms optimized for MRI analysis (3D)"""
        self.check_availability()

        base_transforms = [
            EnsureChannelFirst(),
            Spacing(pixdim=spacing, mode='bilinear'),
            Orientation(axcodes='RAS'),
            NormalizeIntensity(nonzero=True),
            CropForeground(select_fn=lambda x: x > 0),
            Resize(spatial_size=spatial_size, mode='trilinear'),
        ]

        if is_training:
            augmentation_transforms = [
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandRotate90(prob=0.3, max_k=3, spatial_axes=(0, 1)),
                RandAffine(
                    prob=0.5,
                    rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.1, 0.1, 0.1),
                    mode='bilinear'
                ),
                RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0)),
                RandHistogramShift(prob=0.3),
            ]
            base_transforms.extend(augmentation_transforms)

        base_transforms.append(ToTensor())
        return Compose(base_transforms)

    # ==================== SEGMENTATION TRANSFORMS ====================

    def get_segmentation_transforms(
        self,
        spatial_size: Tuple[int, int, int] = (96, 96, 96),
        num_samples: int = 4,
        is_training: bool = False
    ) -> Callable:
        """Get transforms for medical image segmentation tasks"""
        self.check_availability()

        if is_training:
            return Compose([
                EnsureChannelFirstd(keys=['image', 'label']),
                Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0)),
                Orientationd(keys=['image', 'label'], axcodes='RAS'),
                ScaleIntensityRanged(keys=['image'], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=['image', 'label'], source_key='image'),
                RandCropByPosNegLabeld(
                    keys=['image', 'label'],
                    label_key='label',
                    spatial_size=spatial_size,
                    pos=1,
                    neg=1,
                    num_samples=num_samples,
                ),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=['image', 'label'], prob=0.3, max_k=3),
                RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.3),
                RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.3),
                ToTensord(keys=['image', 'label']),
            ])
        else:
            return Compose([
                EnsureChannelFirstd(keys=['image', 'label']),
                Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0)),
                Orientationd(keys=['image', 'label'], axcodes='RAS'),
                ScaleIntensityRanged(keys=['image'], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=['image', 'label'], source_key='image'),
                ToTensord(keys=['image', 'label']),
            ])

    # ==================== POST-PROCESSING ====================

    def get_post_processing_transforms(
        self,
        num_classes: int,
        threshold: float = 0.5,
        keep_largest: bool = True
    ) -> Callable:
        """Get post-processing transforms for model outputs"""
        self.check_availability()

        transforms = [
            Activations(softmax=True),
            AsDiscrete(argmax=True, to_onehot=num_classes),
        ]

        if keep_largest:
            transforms.append(KeepLargestConnectedComponent())

        return Compose(transforms)

    # ==================== UTILITY METHODS ====================

    def preprocess_pil_image(
        self,
        image: Image.Image,
        transform_fn: Callable
    ) -> torch.Tensor:
        """Preprocess a PIL image using MONAI transforms"""
        # Convert PIL to numpy array
        img_array = np.array(image)

        # Apply transforms
        transformed = transform_fn(img_array)

        # Ensure tensor output
        if isinstance(transformed, np.ndarray):
            transformed = torch.from_numpy(transformed)

        return transformed

    def preprocess_bytes(
        self,
        image_bytes: bytes,
        transform_fn: Callable
    ) -> torch.Tensor:
        """Preprocess image bytes using MONAI transforms"""
        # Convert bytes to PIL image
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return self.preprocess_pil_image(image, transform_fn)


# ==================== CONVENIENCE FUNCTIONS ====================

def get_preprocessing_pipeline(
    modality: str,
    spatial_size: Optional[Tuple] = None,
    is_training: bool = False,
    is_grayscale: bool = False
) -> Callable:
    """
    Get a preprocessing pipeline for a specific imaging modality.

    Args:
        modality: One of 'chest_xray', 'skin_lesion', 'fundus', 'ct', 'mri'
        spatial_size: Optional spatial size override
        is_training: Whether to include augmentation transforms
        is_grayscale: For 2D modalities, whether input is grayscale (H, W) or RGB (H, W, C)

    Returns:
        MONAI Compose transform pipeline
    """
    transforms = MONAITransforms()

    modality_configs = {
        'chest_xray': {
            'fn': transforms.get_chest_xray_transforms,
            'default_size': (224, 224),
            'supports_grayscale': True
        },
        'skin_lesion': {
            'fn': transforms.get_skin_lesion_transforms,
            'default_size': (224, 224),
            'supports_grayscale': False
        },
        'fundus': {
            'fn': transforms.get_fundus_transforms,
            'default_size': (512, 512),
            'supports_grayscale': False
        },
        'ct': {
            'fn': transforms.get_ct_scan_transforms,
            'default_size': (96, 96, 96),
            'supports_grayscale': False
        },
        'mri': {
            'fn': transforms.get_mri_transforms,
            'default_size': (128, 128, 128),
            'supports_grayscale': False
        }
    }

    if modality not in modality_configs:
        raise ValueError(f"Unknown modality: {modality}. Choose from {list(modality_configs.keys())}")

    config = modality_configs[modality]
    size = spatial_size or config['default_size']

    # Pass is_grayscale for modalities that support it
    if config.get('supports_grayscale') and is_grayscale:
        return config['fn'](spatial_size=size, is_training=is_training, is_grayscale=True)
    return config['fn'](spatial_size=size, is_training=is_training)


def get_augmentation_pipeline(
    modality: str,
    intensity: str = 'medium'
) -> Callable:
    """
    Get an augmentation-only pipeline for a specific modality.

    Args:
        modality: One of 'chest_xray', 'skin_lesion', 'fundus', 'ct', 'mri'
        intensity: 'light', 'medium', or 'heavy' augmentation

    Returns:
        MONAI Compose transform pipeline for augmentation
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI not available")

    prob_map = {'light': 0.2, 'medium': 0.5, 'heavy': 0.8}
    prob = prob_map.get(intensity, 0.5)

    if modality in ['chest_xray', 'skin_lesion', 'fundus']:
        # 2D augmentations
        return Compose([
            RandFlip(prob=prob, spatial_axis=0),
            RandFlip(prob=prob, spatial_axis=1),
            RandRotate(range_x=0.2, prob=prob),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=prob),
            RandGaussianNoise(prob=prob * 0.5, mean=0.0, std=0.05),
            RandAdjustContrast(prob=prob * 0.5, gamma=(0.9, 1.1)),
        ])
    else:
        # 3D augmentations
        return Compose([
            RandFlip(prob=prob, spatial_axis=0),
            RandFlip(prob=prob, spatial_axis=1),
            RandFlip(prob=prob, spatial_axis=2),
            RandRotate90(prob=prob, max_k=3),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=prob * 0.5),
            RandGaussianNoise(prob=prob * 0.4, mean=0.0, std=0.02),
        ])
