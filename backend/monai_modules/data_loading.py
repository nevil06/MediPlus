# MONAI Data Loading Module - Efficient Medical Image Data Management
# Optimized data loading for training and inference

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Sequence
import os
import json
from pathlib import Path

try:
    from monai.data import (
        # Dataset types
        Dataset as MonaiDataset,
        CacheDataset,
        PersistentDataset,
        SmartCacheDataset,
        ArrayDataset,
        ImageDataset,

        # Data loading utilities
        DataLoader as MonaiDataLoader,
        decollate_batch,
        list_data_collate,
        pad_list_data_collate,

        # Readers
        NibabelReader,
        PILReader,
        ITKReader,
        NumpyReader,

        # Partition and sampling
        partition_dataset,
        partition_dataset_classes,
        select_cross_validation_folds,
    )
    MONAI_DATA_AVAILABLE = True
except ImportError:
    MONAI_DATA_AVAILABLE = False


class MONAIDataLoading:
    """Efficient data loading utilities for medical imaging"""

    def __init__(self):
        self.available = MONAI_DATA_AVAILABLE

    def check_availability(self):
        """Check if MONAI data utilities are available"""
        if not self.available:
            raise ImportError("MONAI data utilities not available. Install with: pip install monai")
        return True

    # ==================== DATASET CREATION ====================

    def create_dataset(
        self,
        data: List[Dict[str, Any]],
        transform: Optional[Callable] = None,
        cache_type: str = 'none',
        cache_rate: float = 1.0,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
    ) -> Dataset:
        """
        Create an appropriate MONAI dataset.

        Args:
            data: List of data dictionaries (e.g., [{'image': path, 'label': path}, ...])
            transform: MONAI Compose transform pipeline
            cache_type: 'none', 'cache', 'persistent', or 'smart'
            cache_rate: Fraction of data to cache (for CacheDataset)
            num_workers: Number of workers for caching
            cache_dir: Directory for persistent cache

        Returns:
            Configured MONAI Dataset
        """
        self.check_availability()

        if cache_type == 'none':
            return MonaiDataset(data=data, transform=transform)

        elif cache_type == 'cache':
            return CacheDataset(
                data=data,
                transform=transform,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )

        elif cache_type == 'persistent':
            if cache_dir is None:
                raise ValueError("cache_dir required for persistent cache")
            return PersistentDataset(
                data=data,
                transform=transform,
                cache_dir=cache_dir,
            )

        elif cache_type == 'smart':
            return SmartCacheDataset(
                data=data,
                transform=transform,
                cache_rate=cache_rate,
                num_init_workers=num_workers,
                num_replace_workers=num_workers // 2,
            )

        else:
            raise ValueError(f"Unknown cache_type: {cache_type}")

    def create_image_dataset(
        self,
        image_files: List[str],
        labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        seg_files: Optional[List[str]] = None,
        seg_transform: Optional[Callable] = None,
        reader: str = 'auto',
    ) -> Dataset:
        """
        Create an image-based dataset (simpler interface).

        Args:
            image_files: List of image file paths
            labels: Optional list of classification labels
            transform: Transform for images
            seg_files: Optional list of segmentation mask paths
            seg_transform: Transform for segmentation masks
            reader: 'auto', 'nibabel', 'pil', 'itk', or 'numpy'

        Returns:
            ImageDataset instance
        """
        self.check_availability()

        # Select reader
        reader_map = {
            'nibabel': NibabelReader(),
            'pil': PILReader(),
            'itk': ITKReader(),
            'numpy': NumpyReader(),
            'auto': None,  # Let MONAI auto-detect
        }

        if reader not in reader_map:
            raise ValueError(f"Unknown reader: {reader}")

        return ImageDataset(
            image_files=image_files,
            labels=labels,
            transform=transform,
            seg_files=seg_files,
            seg_transform=seg_transform,
            reader=reader_map.get(reader),
        )

    # ==================== DATALOADER CREATION ====================

    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        collate_fn: str = 'default',
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Create an optimized DataLoader.

        Args:
            dataset: MONAI Dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
            collate_fn: 'default', 'list', or 'pad_list'
            drop_last: Drop last incomplete batch

        Returns:
            Configured DataLoader
        """
        self.check_availability()

        # Select collate function
        collate_map = {
            'default': None,
            'list': list_data_collate,
            'pad_list': pad_list_data_collate,
        }

        if collate_fn not in collate_map:
            raise ValueError(f"Unknown collate_fn: {collate_fn}")

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=collate_map.get(collate_fn),
            drop_last=drop_last,
        )

    # ==================== DATA PARTITIONING ====================

    def split_dataset(
        self,
        data: List[Dict[str, Any]],
        ratios: Sequence[float] = (0.8, 0.1, 0.1),
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[Dict], ...]:
        """
        Split dataset into train/val/test.

        Args:
            data: List of data dictionaries
            ratios: Split ratios (must sum to 1.0)
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        self.check_availability()

        return partition_dataset(
            data=data,
            ratios=list(ratios),
            shuffle=shuffle,
            seed=seed,
        )

    def split_by_class(
        self,
        data: List[Dict[str, Any]],
        classes: List[int],
        ratios: Sequence[float] = (0.8, 0.1, 0.1),
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[Dict], ...]:
        """
        Split dataset with stratification by class.

        Args:
            data: List of data dictionaries
            classes: List of class labels corresponding to data
            ratios: Split ratios
            shuffle: Whether to shuffle
            seed: Random seed

        Returns:
            Tuple of stratified splits
        """
        self.check_availability()

        return partition_dataset_classes(
            data=data,
            classes=classes,
            ratios=list(ratios),
            shuffle=shuffle,
            seed=seed,
        )

    def create_cross_validation_folds(
        self,
        data: List[Dict[str, Any]],
        num_folds: int = 5,
        fold: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Create cross-validation train/val split.

        Args:
            data: List of data dictionaries
            num_folds: Number of CV folds
            fold: Current fold index (0 to num_folds-1)
            shuffle: Whether to shuffle
            seed: Random seed

        Returns:
            Tuple of (train_data, val_data)
        """
        self.check_availability()

        partitions = partition_dataset(
            data=data,
            ratios=[1.0 / num_folds] * num_folds,
            shuffle=shuffle,
            seed=seed,
        )

        return select_cross_validation_folds(
            partitions=partitions,
            folds=fold,
        )

    # ==================== DATA UTILITIES ====================

    @staticmethod
    def create_data_list(
        data_dir: str,
        image_key: str = 'image',
        label_key: str = 'label',
        image_suffix: str = '.nii.gz',
        label_suffix: str = '_seg.nii.gz',
    ) -> List[Dict[str, str]]:
        """
        Create data list from directory structure.

        Assumes structure:
            data_dir/
                images/
                    case001.nii.gz
                    case002.nii.gz
                labels/
                    case001_seg.nii.gz
                    case002_seg.nii.gz

        Returns:
            List of dictionaries with image and label paths
        """
        data_dir = Path(data_dir)
        images_dir = data_dir / 'images'
        labels_dir = data_dir / 'labels'

        data_list = []

        for img_path in sorted(images_dir.glob(f'*{image_suffix}')):
            # Derive label path
            case_name = img_path.stem.replace(image_suffix.replace('.', ''), '')
            label_path = labels_dir / f'{case_name}{label_suffix}'

            if label_path.exists():
                data_list.append({
                    image_key: str(img_path),
                    label_key: str(label_path),
                })

        return data_list

    @staticmethod
    def create_classification_data_list(
        data_dir: str,
        class_mapping: Optional[Dict[str, int]] = None,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Create classification data list from directory structure.

        Assumes structure:
            data_dir/
                class_0/
                    image1.png
                    image2.png
                class_1/
                    image3.png
                    ...

        Returns:
            List of dictionaries with image path and label
        """
        data_dir = Path(data_dir)
        data_list = []

        # Get class directories
        class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

        # Create class mapping if not provided
        if class_mapping is None:
            class_mapping = {d.name: i for i, d in enumerate(class_dirs)}

        for class_dir in class_dirs:
            class_label = class_mapping.get(class_dir.name)
            if class_label is None:
                continue

            for ext in extensions:
                for img_path in class_dir.glob(f'*{ext}'):
                    data_list.append({
                        'image': str(img_path),
                        'label': class_label,
                    })

        return data_list


# ==================== CONVENIENCE FUNCTIONS ====================

def get_dataloader(
    data: List[Dict[str, Any]],
    transform: Optional[Callable] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    cache_type: str = 'none',
    cache_rate: float = 1.0,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> DataLoader:
    """
    Get a ready-to-use DataLoader with optimal settings.

    Args:
        data: List of data dictionaries
        transform: MONAI transform pipeline
        batch_size: Batch size
        shuffle: Whether to shuffle
        cache_type: 'none', 'cache', 'persistent', or 'smart'
        cache_rate: Cache rate for CacheDataset
        num_workers: Number of workers
        pin_memory: Pin memory for GPU

    Returns:
        Configured DataLoader
    """
    loader = MONAIDataLoading()

    dataset = loader.create_dataset(
        data=data,
        transform=transform,
        cache_type=cache_type,
        cache_rate=cache_rate,
        num_workers=num_workers,
        **kwargs
    )

    return loader.create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ==================== DICOM/NIFTI UTILITIES ====================

class MedicalImageReader:
    """Unified reader for various medical image formats"""

    @staticmethod
    def read_dicom_series(
        dicom_dir: str,
    ) -> Tuple[Any, Dict]:
        """
        Read DICOM series from directory.

        Returns:
            Tuple of (image_array, metadata_dict)
        """
        try:
            import pydicom
            from pydicom.filereader import dcmread

            # Get all DICOM files
            dicom_files = sorted([
                os.path.join(dicom_dir, f)
                for f in os.listdir(dicom_dir)
                if f.endswith('.dcm')
            ])

            if not dicom_files:
                raise ValueError(f"No DICOM files found in {dicom_dir}")

            # Read first file for metadata
            ref_ds = dcmread(dicom_files[0])

            # Read all slices
            slices = [dcmread(f) for f in dicom_files]
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

            # Stack into 3D array
            import numpy as np
            image_array = np.stack([s.pixel_array for s in slices])

            # Apply rescale slope and intercept
            if hasattr(ref_ds, 'RescaleSlope') and hasattr(ref_ds, 'RescaleIntercept'):
                image_array = image_array * ref_ds.RescaleSlope + ref_ds.RescaleIntercept

            # Extract metadata
            metadata = {
                'patient_id': getattr(ref_ds, 'PatientID', None),
                'modality': getattr(ref_ds, 'Modality', None),
                'spacing': [
                    float(ref_ds.PixelSpacing[0]) if hasattr(ref_ds, 'PixelSpacing') else 1.0,
                    float(ref_ds.PixelSpacing[1]) if hasattr(ref_ds, 'PixelSpacing') else 1.0,
                    float(ref_ds.SliceThickness) if hasattr(ref_ds, 'SliceThickness') else 1.0,
                ],
            }

            return image_array, metadata

        except ImportError:
            raise ImportError("pydicom required for DICOM reading. Install with: pip install pydicom")

    @staticmethod
    def read_nifti(
        nifti_path: str,
    ) -> Tuple[Any, Dict]:
        """
        Read NIfTI file.

        Returns:
            Tuple of (image_array, metadata_dict)
        """
        try:
            import nibabel as nib

            nii = nib.load(nifti_path)
            image_array = nii.get_fdata()

            # Extract metadata
            header = nii.header
            metadata = {
                'shape': image_array.shape,
                'spacing': header.get_zooms()[:3],
                'affine': nii.affine.tolist(),
                'datatype': str(header.get_data_dtype()),
            }

            return image_array, metadata

        except ImportError:
            raise ImportError("nibabel required for NIfTI reading. Install with: pip install nibabel")
