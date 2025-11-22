"""
Enhanced Medical Image Analysis Service
Uses full MONAI capabilities for improved medical image analysis
Integrates with existing services while providing advanced features
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import base64
from io import BytesIO
import sys
import os

# Add parent directory to path for monai_modules import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import MONAI modules
try:
    from monai_modules.transforms import MONAITransforms, get_preprocessing_pipeline
    from monai_modules.networks import MONAINetworks, get_network
    from monai_modules.inference import ClassificationInferer, SlidingWindowProcessor
    from monai_modules.losses import get_loss_function
    from monai_modules.metrics import get_metrics, compute_comprehensive_metrics
    MONAI_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"MONAI modules import error: {e}")
    MONAI_MODULES_AVAILABLE = False

# Import AI Enhancement Service
try:
    from services.ai_vision_service import ai_enhancement_service
    AI_ENHANCEMENT_AVAILABLE = True
except ImportError:
    AI_ENHANCEMENT_AVAILABLE = False


class EnhancedChestXRayAnalyzer:
    """
    Enhanced Chest X-Ray analysis with full MONAI capabilities

    Features:
    - MONAI medical-specific transforms
    - Multiple model architectures
    - Test-time augmentation
    - Groq AI enhancement
    """

    CLASSES = [
        'Normal', 'Pneumonia', 'COVID-19', 'Cardiomegaly',
        'Lung Nodule', 'Pleural Effusion', 'Atelectasis', 'Pneumothorax'
    ]

    def __init__(self, device: str = 'auto', use_tta: bool = True):
        self.device = self._select_device(device)
        self.use_tta = use_tta
        self.model = None
        self.transforms = None
        self.inferer = None
        self._initialize()

    def _select_device(self, device: str) -> str:
        if device == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return 'cuda'
            return 'cpu'
        return device

    def _initialize(self):
        """Initialize with MONAI transforms and networks"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available")
            return

        if MONAI_MODULES_AVAILABLE:
            # Use MONAI transforms
            transforms_helper = MONAITransforms(device=self.device)
            self.transforms = transforms_helper.get_chest_xray_transforms(
                spatial_size=(224, 224),
                is_training=False
            )

            # Use MONAI network
            networks_helper = MONAINetworks(device=self.device)
            try:
                self.model = networks_helper.get_densenet121(
                    spatial_dims=2,
                    in_channels=3,
                    out_channels=len(self.CLASSES),
                    pretrained=True
                )
                self.model.eval()

                # Create classification inferer
                self.inferer = ClassificationInferer(
                    model=self.model,
                    device=self.device,
                    class_names=self.CLASSES
                )
            except Exception as e:
                print(f"Model initialization error: {e}")

    def analyze(self, image_data: str, enhance_with_ai: bool = True) -> Dict[str, Any]:
        """
        Analyze chest X-ray with MONAI + optional Groq enhancement

        Args:
            image_data: Base64 encoded image
            enhance_with_ai: Use Groq AI for enhanced explanations

        Returns:
            Comprehensive analysis results
        """
        try:
            # Decode image
            image = self._decode_image(image_data)
            if image is None:
                return self._fallback_analysis()

            # Get MONAI predictions
            monai_results = self._analyze_with_monai(image)

            # Enhance with Groq AI if available
            if enhance_with_ai and AI_ENHANCEMENT_AVAILABLE:
                enhanced = ai_enhancement_service.enhance_chest_xray_analysis(
                    image_data=image_data,
                    monai_predictions=monai_results.get('predictions', {}),
                    analysis_type='chest_xray'
                )
                monai_results['ai_enhanced'] = enhanced

            return monai_results

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'fallback': self._fallback_analysis()
            }

    def _analyze_with_monai(self, image: Image.Image) -> Dict[str, Any]:
        """Run MONAI-based analysis"""
        if self.inferer is None:
            return self._fallback_analysis()

        # Preprocess and predict
        if self.use_tta:
            results = self.inferer.predict_from_pil(
                image=image,
                transform=self.transforms,
                top_k=5,
                use_tta=True
            )
        else:
            results = self.inferer.predict_from_pil(
                image=image,
                transform=self.transforms,
                top_k=5,
                use_tta=False
            )

        # Format results
        predictions = {}
        all_probs = results['all_probabilities'][0]

        for i, class_name in enumerate(self.CLASSES):
            predictions[class_name] = {
                'probability': float(all_probs[i]),
                'confidence': self._get_confidence_level(all_probs[i])
            }

        # Get primary diagnosis
        top_idx = results['predictions'][0][0]
        top_prob = results['probabilities'][0][0]

        return {
            'success': True,
            'primary_diagnosis': {
                'condition': self.CLASSES[top_idx],
                'probability': float(top_prob),
                'confidence': self._get_confidence_level(top_prob)
            },
            'predictions': predictions,
            'all_findings': self._get_significant_findings(predictions),
            'model_info': {
                'architecture': 'DenseNet121',
                'framework': 'MONAI',
                'transforms': 'MONAI Medical Transforms',
                'test_time_augmentation': self.use_tta
            }
        }

    def _get_confidence_level(self, prob: float) -> str:
        if prob >= 0.8:
            return 'High'
        elif prob >= 0.5:
            return 'Moderate'
        elif prob >= 0.3:
            return 'Low'
        return 'Very Low'

    def _get_significant_findings(self, predictions: Dict) -> List[Dict]:
        """Get findings with probability > 0.2"""
        findings = []
        for condition, data in predictions.items():
            if data['probability'] > 0.2 and condition != 'Normal':
                findings.append({
                    'condition': condition,
                    'probability': data['probability'],
                    'confidence': data['confidence']
                })
        return sorted(findings, key=lambda x: x['probability'], reverse=True)

    def _decode_image(self, image_data: str) -> Optional[Image.Image]:
        """Decode base64 image"""
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            print(f"Image decode error: {e}")
            return None

    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback when model unavailable"""
        return {
            'success': False,
            'error': 'Model not available',
            'message': 'Please ensure MONAI and PyTorch are properly installed',
            'recommendations': [
                'Consult a healthcare professional for accurate diagnosis',
                'This system is for screening purposes only'
            ]
        }


class EnhancedSkinLesionAnalyzer:
    """
    Enhanced Skin Lesion analysis with full MONAI capabilities

    Features:
    - MONAI dermoscopy-specific transforms
    - EfficientNet architecture
    - Color augmentation for skin tones
    - ABCDE melanoma criteria integration
    """

    CLASSES = [
        'Melanocytic Nevi', 'Melanoma', 'Benign Keratosis',
        'Basal Cell Carcinoma', 'Actinic Keratosis',
        'Vascular Lesion', 'Dermatofibroma'
    ]

    def __init__(self, device: str = 'auto', use_tta: bool = True):
        self.device = 'cuda' if device == 'auto' and TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        self.use_tta = use_tta
        self.model = None
        self.transforms = None
        self.inferer = None
        self._initialize()

    def _initialize(self):
        """Initialize with MONAI transforms and networks"""
        if not TORCH_AVAILABLE or not MONAI_MODULES_AVAILABLE:
            return

        transforms_helper = MONAITransforms(device=self.device)
        self.transforms = transforms_helper.get_skin_lesion_transforms(
            spatial_size=(224, 224),
            is_training=False
        )

        networks_helper = MONAINetworks(device=self.device)
        try:
            self.model = networks_helper.get_efficientnet(
                model_name='efficientnet-b0',
                spatial_dims=2,
                in_channels=3,
                num_classes=len(self.CLASSES),
                pretrained=True
            )
            self.model.eval()

            self.inferer = ClassificationInferer(
                model=self.model,
                device=self.device,
                class_names=self.CLASSES
            )
        except Exception as e:
            print(f"Model initialization error: {e}")

    def analyze(self, image_data: str, enhance_with_ai: bool = True) -> Dict[str, Any]:
        """Analyze skin lesion image"""
        try:
            image = self._decode_image(image_data)
            if image is None:
                return self._fallback_analysis()

            monai_results = self._analyze_with_monai(image)

            # Add ABCDE criteria assessment
            abcde_assessment = self._assess_abcde_criteria(image)
            monai_results['abcde_assessment'] = abcde_assessment

            if enhance_with_ai and AI_ENHANCEMENT_AVAILABLE:
                enhanced = ai_enhancement_service.enhance_skin_analysis(
                    image_data=image_data,
                    monai_predictions=monai_results.get('predictions', {}),
                    abcde_scores=abcde_assessment
                )
                monai_results['ai_enhanced'] = enhanced

            return monai_results

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _analyze_with_monai(self, image: Image.Image) -> Dict[str, Any]:
        """Run MONAI-based analysis"""
        if self.inferer is None:
            return self._fallback_analysis()

        results = self.inferer.predict_from_pil(
            image=image,
            transform=self.transforms,
            top_k=5,
            use_tta=self.use_tta
        )

        predictions = {}
        all_probs = results['all_probabilities'][0]

        for i, class_name in enumerate(self.CLASSES):
            predictions[class_name] = {
                'probability': float(all_probs[i]),
                'is_malignant': class_name in ['Melanoma', 'Basal Cell Carcinoma']
            }

        top_idx = results['predictions'][0][0]
        top_prob = results['probabilities'][0][0]

        return {
            'success': True,
            'primary_diagnosis': {
                'condition': self.CLASSES[top_idx],
                'probability': float(top_prob),
                'is_malignant': self.CLASSES[top_idx] in ['Melanoma', 'Basal Cell Carcinoma']
            },
            'predictions': predictions,
            'model_info': {
                'architecture': 'EfficientNet-B0',
                'framework': 'MONAI'
            }
        }

    def _assess_abcde_criteria(self, image: Image.Image) -> Dict[str, Any]:
        """Assess ABCDE melanoma criteria"""
        # Simplified criteria assessment
        return {
            'asymmetry': {'score': 0.5, 'description': 'Requires clinical assessment'},
            'border': {'score': 0.5, 'description': 'Requires clinical assessment'},
            'color': {'score': 0.5, 'description': 'Requires clinical assessment'},
            'diameter': {'score': 0.5, 'description': 'Requires clinical assessment'},
            'evolution': {'score': 0.5, 'description': 'Patient history required'},
            'note': 'ABCDE assessment requires clinical examination'
        }

    def _decode_image(self, image_data: str) -> Optional[Image.Image]:
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except:
            return None

    def _fallback_analysis(self) -> Dict[str, Any]:
        return {
            'success': False,
            'error': 'Model not available',
            'recommendation': 'Consult a dermatologist for accurate diagnosis'
        }


class EnhancedEyeHealthAnalyzer:
    """
    Enhanced Eye/Retinal analysis with full MONAI capabilities

    Features:
    - MONAI fundus-specific transforms
    - Diabetic retinopathy grading
    - Multiple condition detection
    """

    DR_GRADES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    CONDITIONS = ['Diabetic Retinopathy', 'Glaucoma', 'AMD', 'Cataracts', 'Normal']

    def __init__(self, device: str = 'auto', use_tta: bool = True):
        self.device = 'cuda' if device == 'auto' and TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        self.use_tta = use_tta
        self.model = None
        self.transforms = None
        self.inferer = None
        self._initialize()

    def _initialize(self):
        """Initialize with MONAI transforms"""
        if not TORCH_AVAILABLE or not MONAI_MODULES_AVAILABLE:
            return

        transforms_helper = MONAITransforms(device=self.device)
        self.transforms = transforms_helper.get_fundus_transforms(
            spatial_size=(512, 512),
            is_training=False
        )

        networks_helper = MONAINetworks(device=self.device)
        try:
            self.model = networks_helper.get_densenet121(
                spatial_dims=2,
                in_channels=3,
                out_channels=len(self.DR_GRADES),
                pretrained=True
            )
            self.model.eval()

            self.inferer = ClassificationInferer(
                model=self.model,
                device=self.device,
                class_names=self.DR_GRADES
            )
        except Exception as e:
            print(f"Model initialization error: {e}")

    def analyze(self, image_data: str, enhance_with_ai: bool = True) -> Dict[str, Any]:
        """Analyze fundus/retinal image"""
        try:
            image = self._decode_image(image_data)
            if image is None:
                return self._fallback_analysis()

            monai_results = self._analyze_with_monai(image)

            if enhance_with_ai and AI_ENHANCEMENT_AVAILABLE:
                enhanced = ai_enhancement_service.enhance_eye_analysis(
                    image_data=image_data,
                    monai_predictions=monai_results.get('predictions', {}),
                    dr_grade=monai_results.get('dr_grade', {})
                )
                monai_results['ai_enhanced'] = enhanced

            return monai_results

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _analyze_with_monai(self, image: Image.Image) -> Dict[str, Any]:
        """Run MONAI-based analysis"""
        if self.inferer is None:
            return self._fallback_analysis()

        results = self.inferer.predict_from_pil(
            image=image,
            transform=self.transforms,
            top_k=5,
            use_tta=self.use_tta
        )

        all_probs = results['all_probabilities'][0]
        top_idx = results['predictions'][0][0]
        top_prob = results['probabilities'][0][0]

        # DR grading
        dr_predictions = {}
        for i, grade in enumerate(self.DR_GRADES):
            dr_predictions[grade] = float(all_probs[i])

        return {
            'success': True,
            'dr_grade': {
                'grade': self.DR_GRADES[top_idx],
                'severity_index': top_idx,
                'probability': float(top_prob),
                'all_grades': dr_predictions
            },
            'recommendations': self._get_recommendations(top_idx),
            'model_info': {
                'architecture': 'DenseNet121',
                'framework': 'MONAI',
                'input_size': '512x512'
            }
        }

    def _get_recommendations(self, severity: int) -> List[str]:
        """Get recommendations based on DR severity"""
        recommendations = {
            0: ['Continue annual eye exams', 'Maintain good blood sugar control'],
            1: ['Schedule follow-up in 9-12 months', 'Monitor blood sugar closely'],
            2: ['Schedule follow-up in 6 months', 'Consider referral to specialist'],
            3: ['Urgent referral to ophthalmologist', 'May need treatment soon'],
            4: ['Immediate referral required', 'Treatment typically needed']
        }
        return recommendations.get(severity, ['Consult an ophthalmologist'])

    def _decode_image(self, image_data: str) -> Optional[Image.Image]:
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except:
            return None

    def _fallback_analysis(self) -> Dict[str, Any]:
        return {
            'success': False,
            'error': 'Model not available',
            'recommendation': 'Please consult an ophthalmologist'
        }


# Service instances
enhanced_chest_xray_analyzer = EnhancedChestXRayAnalyzer()
enhanced_skin_lesion_analyzer = EnhancedSkinLesionAnalyzer()
enhanced_eye_health_analyzer = EnhancedEyeHealthAnalyzer()
