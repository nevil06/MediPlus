# MONAI Networks Module - Advanced Medical Imaging Architectures
# State-of-the-art networks for segmentation, classification, and more

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Sequence

try:
    from monai.networks.nets import (
        # Classic segmentation networks
        UNet,
        BasicUNet,
        AttentionUnet,
        VNet,
        SegResNet,
        SegResNetVAE,

        # Transformer-based networks
        UNETR,
        SwinUNETR,
        ViT,

        # Classification networks
        DenseNet121,
        DenseNet169,
        DenseNet201,
        DenseNet264,
        EfficientNetBN,
        SENet154,
        SEResNet50,
        SEResNet101,
        SEResNet152,
        SEResNext50,
        SEResNext101,
        ResNet,
        ResNetFeatures,
        TorchVisionFCModel,

        # Autoencoders
        AutoEncoder,
        VarAutoEncoder,

        # Detection networks
        AHNet,

        # Other specialized networks
        HighResNet,
        DynUNet,
        FlexibleUNet,
    )
    MONAI_NETWORKS_AVAILABLE = True
except ImportError:
    MONAI_NETWORKS_AVAILABLE = False


class MONAINetworks:
    """Factory class for MONAI medical imaging networks"""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.available = MONAI_NETWORKS_AVAILABLE

    def check_availability(self):
        """Check if MONAI networks are available"""
        if not self.available:
            raise ImportError("MONAI networks not available. Install with: pip install monai")
        return True

    # ==================== SEGMENTATION NETWORKS ====================

    def get_unet(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: Sequence[int] = (16, 32, 64, 128, 256),
        strides: Sequence[int] = (2, 2, 2, 2),
        num_res_units: int = 2,
        norm: str = 'BATCH',
        dropout: float = 0.0,
    ) -> nn.Module:
        """
        Standard U-Net architecture.

        Best for: General medical image segmentation
        Efficiency: Good balance of accuracy and speed
        Memory: Moderate
        """
        self.check_availability()
        return UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            dropout=dropout,
        ).to(self.device)

    def get_basic_unet(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        dropout: float = 0.0,
    ) -> nn.Module:
        """
        Basic U-Net with simpler architecture.

        Best for: Quick experiments, limited resources
        Efficiency: Fast
        Memory: Low
        """
        self.check_availability()
        return BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            dropout=dropout,
        ).to(self.device)

    def get_attention_unet(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: Sequence[int] = (16, 32, 64, 128, 256),
        strides: Sequence[int] = (2, 2, 2, 2),
        dropout: float = 0.0,
    ) -> nn.Module:
        """
        U-Net with attention gates.

        Best for: Segmentation requiring focus on relevant regions
        Efficiency: Slightly slower than basic UNet
        Memory: Moderate to high
        """
        self.check_availability()
        return AttentionUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            dropout=dropout,
        ).to(self.device)

    def get_vnet(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float = 0.0,
    ) -> nn.Module:
        """
        V-Net for volumetric segmentation.

        Best for: 3D medical image segmentation (CT, MRI)
        Efficiency: Good for volumetric data
        Memory: Moderate
        """
        self.check_availability()
        return VNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
        ).to(self.device)

    def get_segresnet(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        init_filters: int = 8,
        blocks_down: Sequence[int] = (1, 2, 2, 4),
        blocks_up: Sequence[int] = (1, 1, 1),
        dropout_prob: float = 0.0,
    ) -> nn.Module:
        """
        SegResNet - ResNet-based segmentation.

        Best for: High-accuracy 3D segmentation
        Efficiency: Good
        Memory: Moderate
        """
        self.check_availability()
        return SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout_prob,
        ).to(self.device)

    def get_unetr(
        self,
        img_size: Tuple[int, ...] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 2,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = 'conv',
        norm_name: str = 'instance',
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> nn.Module:
        """
        UNETR - Transformer-based U-Net.

        Best for: Large-scale 3D segmentation with global context
        Efficiency: Slower but high accuracy
        Memory: High (requires significant GPU memory)
        """
        self.check_availability()
        return UNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed=pos_embed,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        ).to(self.device)

    def get_swin_unetr(
        self,
        img_size: Tuple[int, ...] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 2,
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        norm_name: str = 'instance',
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        spatial_dims: int = 3,
        use_checkpoint: bool = False,
    ) -> nn.Module:
        """
        Swin UNETR - State-of-the-art transformer for medical imaging.

        Best for: Highest accuracy 3D segmentation
        Winner of BTCV multi-organ segmentation challenge
        Efficiency: Slower
        Memory: Very high (benefits from gradient checkpointing)
        """
        self.check_availability()
        return SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            spatial_dims=spatial_dims,
            use_checkpoint=use_checkpoint,
        ).to(self.device)

    def get_dynunet(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        kernel_size: Sequence[Sequence[int]] = [[3, 3, 3]] * 6,
        strides: Sequence[Sequence[int]] = [[1, 1, 1]] + [[2, 2, 2]] * 5,
        upsample_kernel_size: Sequence[Sequence[int]] = [[2, 2, 2]] * 5,
        norm_name: str = 'INSTANCE',
        deep_supervision: bool = True,
        deep_supr_num: int = 3,
    ) -> nn.Module:
        """
        Dynamic U-Net - nnU-Net style architecture.

        Best for: Auto-configured medical image segmentation
        Used in nnU-Net framework (top performer on many benchmarks)
        Efficiency: Configurable
        Memory: Depends on configuration
        """
        self.check_availability()
        return DynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=norm_name,
            deep_supervision=deep_supervision,
            deep_supr_num=deep_supr_num,
        ).to(self.device)

    # ==================== CLASSIFICATION NETWORKS ====================

    def get_densenet121(
        self,
        spatial_dims: int = 2,
        in_channels: int = 3,
        out_channels: int = 2,
        pretrained: bool = True,
    ) -> nn.Module:
        """
        DenseNet-121 for classification.

        Best for: 2D medical image classification (chest X-ray, skin lesions)
        Efficiency: Good
        Memory: Moderate
        """
        self.check_availability()
        return DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained=pretrained,
        ).to(self.device)

    def get_efficientnet(
        self,
        model_name: str = 'efficientnet-b0',
        spatial_dims: int = 2,
        in_channels: int = 3,
        num_classes: int = 2,
        pretrained: bool = True,
    ) -> nn.Module:
        """
        EfficientNet for classification.

        Best for: Efficient 2D classification
        Models: b0-b7 (increasing size/accuracy)
        Efficiency: Excellent (designed for efficiency)
        Memory: Low to moderate depending on variant
        """
        self.check_availability()
        return EfficientNetBN(
            model_name=model_name,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
        ).to(self.device)

    def get_senet(
        self,
        spatial_dims: int = 2,
        in_channels: int = 3,
        num_classes: int = 2,
        pretrained: bool = True,
    ) -> nn.Module:
        """
        SE-Net with squeeze-and-excitation blocks.

        Best for: Classification requiring channel attention
        Efficiency: Moderate
        Memory: Moderate
        """
        self.check_availability()
        return SEResNet50(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
        ).to(self.device)

    def get_vit(
        self,
        in_channels: int = 1,
        img_size: Union[int, Sequence[int]] = 224,
        patch_size: Union[int, Sequence[int]] = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        classification: bool = True,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 2,
    ) -> nn.Module:
        """
        Vision Transformer (ViT).

        Best for: Large-scale image classification with pre-training
        Efficiency: Moderate (depends on image/patch size)
        Memory: High
        """
        self.check_availability()
        return ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            classification=classification,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        ).to(self.device)

    # ==================== SPECIALIZED NETWORKS ====================

    def get_highresnet(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        norm_type: str = 'batch',
        dropout_prob: Optional[float] = None,
    ) -> nn.Module:
        """
        HighRes3DNet for brain MRI segmentation.

        Best for: High-resolution medical image analysis
        Efficiency: Good
        Memory: Moderate
        """
        self.check_availability()
        return HighResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            norm_type=norm_type,
            dropout_prob=dropout_prob,
        ).to(self.device)

    def get_autoencoder(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] = (16, 32, 64),
        strides: Sequence[int] = (2, 2, 2),
    ) -> nn.Module:
        """
        Autoencoder for unsupervised learning.

        Best for: Pre-training, anomaly detection, denoising
        Efficiency: Depends on architecture
        Memory: Low to moderate
        """
        self.check_availability()
        return AutoEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
        ).to(self.device)

    def get_vae(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_size: int = 3,
        channels: Sequence[int] = (16, 32, 64),
        strides: Sequence[int] = (2, 2, 2),
    ) -> nn.Module:
        """
        Variational Autoencoder.

        Best for: Generative modeling, data augmentation
        Efficiency: Depends on architecture
        Memory: Low to moderate
        """
        self.check_availability()
        return VarAutoEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_size=latent_size,
            channels=channels,
            strides=strides,
        ).to(self.device)


# ==================== CONVENIENCE FUNCTIONS ====================

def get_network(
    task: str,
    architecture: str = 'auto',
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    img_size: Optional[Tuple[int, ...]] = None,
    pretrained: bool = True,
    device: str = 'cpu',
    **kwargs
) -> nn.Module:
    """
    Get a pre-configured network for a specific task.

    Args:
        task: One of:
            - 'segmentation_2d': 2D medical image segmentation
            - 'segmentation_3d': 3D volumetric segmentation
            - 'segmentation_large': Large-scale 3D segmentation
            - 'classification_2d': 2D image classification
            - 'classification_3d': 3D volume classification

        architecture: Network architecture. Options depend on task:
            - For segmentation: 'unet', 'attention_unet', 'vnet', 'segresnet',
                               'unetr', 'swin_unetr', 'dynunet', 'auto'
            - For classification: 'densenet121', 'efficientnet', 'senet', 'vit', 'auto'

        spatial_dims: 2 or 3 for 2D/3D
        in_channels: Number of input channels
        out_channels: Number of output channels/classes
        img_size: Image size (required for transformer-based networks)
        pretrained: Use pretrained weights (for classification)
        device: Device to load model on

    Returns:
        Configured PyTorch model
    """
    networks = MONAINetworks(device=device)

    # Auto architecture selection
    if architecture == 'auto':
        if 'segmentation' in task:
            if 'large' in task or (img_size and min(img_size) >= 128):
                architecture = 'swin_unetr'
            elif spatial_dims == 3:
                architecture = 'segresnet'
            else:
                architecture = 'unet'
        elif 'classification' in task:
            if spatial_dims == 2:
                architecture = 'efficientnet'
            else:
                architecture = 'densenet121'

    # Build network based on architecture
    arch_map = {
        # Segmentation
        'unet': lambda: networks.get_unet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        ),
        'basic_unet': lambda: networks.get_basic_unet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        ),
        'attention_unet': lambda: networks.get_attention_unet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        ),
        'vnet': lambda: networks.get_vnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        ),
        'segresnet': lambda: networks.get_segresnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        ),
        'unetr': lambda: networks.get_unetr(
            img_size=img_size or (96, 96, 96),
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            **kwargs
        ),
        'swin_unetr': lambda: networks.get_swin_unetr(
            img_size=img_size or (96, 96, 96),
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            **kwargs
        ),
        'dynunet': lambda: networks.get_dynunet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        ),
        'highresnet': lambda: networks.get_highresnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        ),

        # Classification
        'densenet121': lambda: networks.get_densenet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained=pretrained,
        ),
        'efficientnet': lambda: networks.get_efficientnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=out_channels,
            pretrained=pretrained,
            **kwargs
        ),
        'senet': lambda: networks.get_senet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=out_channels,
            pretrained=pretrained,
        ),
        'vit': lambda: networks.get_vit(
            in_channels=in_channels,
            img_size=img_size[0] if img_size else 224,
            num_classes=out_channels,
            spatial_dims=spatial_dims,
            **kwargs
        ),

        # Specialized
        'autoencoder': lambda: networks.get_autoencoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            **kwargs
        ),
        'vae': lambda: networks.get_vae(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            **kwargs
        ),
    }

    if architecture not in arch_map:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(arch_map.keys())}")

    return arch_map[architecture]()


# ==================== MODEL WEIGHTS UTILITIES ====================

class ModelWeights:
    """Utilities for managing model weights"""

    @staticmethod
    def load_pretrained(
        model: nn.Module,
        weights_path: str,
        strict: bool = True,
        device: str = 'cpu'
    ) -> nn.Module:
        """Load pretrained weights from file"""
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=strict)
        return model

    @staticmethod
    def save_weights(
        model: nn.Module,
        save_path: str,
    ):
        """Save model weights to file"""
        torch.save(model.state_dict(), save_path)

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count model parameters"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }

    @staticmethod
    def freeze_encoder(model: nn.Module) -> nn.Module:
        """Freeze encoder layers (useful for transfer learning)"""
        # This is model-specific, but commonly:
        for name, param in model.named_parameters():
            if 'encoder' in name or 'down' in name:
                param.requires_grad = False
        return model
