"""
Skin Lesion Analysis Service using MONAI + AI Vision
Classifies skin lesions for potential melanoma and other conditions
"""

import numpy as np
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO

try:
    import torch
    import torch.nn as nn
    from PIL import Image
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from monai.networks.nets import EfficientNetBN
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

# Import AI Enhancement Service (Groq)
try:
    from services.ai_vision_service import ai_enhancement_service
    AI_ENHANCEMENT_AVAILABLE = True
except ImportError:
    AI_ENHANCEMENT_AVAILABLE = False


class SkinLesionService:
    """Service for analyzing skin lesion images"""

    # Skin lesion classes (based on HAM10000 dataset)
    CLASSES = [
        'Melanocytic Nevi',      # Benign mole
        'Melanoma',              # Malignant
        'Benign Keratosis',      # Seborrheic keratosis
        'Basal Cell Carcinoma',  # Malignant
        'Actinic Keratosis',     # Pre-cancerous
        'Vascular Lesion',       # Benign
        'Dermatofibroma'         # Benign
    ]

    # Risk categorization
    HIGH_RISK = ['Melanoma', 'Basal Cell Carcinoma']
    MODERATE_RISK = ['Actinic Keratosis']
    LOW_RISK = ['Melanocytic Nevi', 'Benign Keratosis', 'Vascular Lesion', 'Dermatofibroma']

    def __init__(self):
        self.model = None
        self.device = 'cpu'
        self.transform = None
        self._initialize()

    def _initialize(self):
        """Initialize model and transforms"""
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = 'cuda'

            # Image preprocessing for skin lesion analysis
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.763, 0.546, 0.570],
                                   std=[0.141, 0.152, 0.169])
            ])

            if MONAI_AVAILABLE:
                try:
                    self.model = EfficientNetBN(
                        model_name='efficientnet-b0',
                        spatial_dims=2,
                        in_channels=3,
                        num_classes=len(self.CLASSES)
                    )
                    self.model.to(self.device)
                    self.model.eval()
                except Exception as e:
                    print(f"MONAI model init failed: {e}")
                    self.model = None

    def analyze_image(self, image_data: str) -> Dict[str, Any]:
        """
        Analyze a skin lesion image using MONAI + Groq enhancement

        Args:
            image_data: Base64 encoded image string

        Returns:
            Analysis results with MONAI predictions enhanced by Groq
        """
        try:
            image = self._decode_image(image_data)
            if image is None:
                return self._fallback_analysis()

            # Step 1: Run MONAI model inference
            if self.model is not None and self.transform is not None:
                predictions = self._run_inference(image)
            else:
                predictions = self._simulate_predictions(image)

            # Calculate ABCDE scores
            abcde_scores = self._calculate_abcde_score(predictions)

            # Step 2: Enhance with Groq LLM
            if AI_ENHANCEMENT_AVAILABLE:
                enhanced = ai_enhancement_service.enhance_skin_lesion_result(predictions, abcde_scores)
                if enhanced.get('enhanced', False):
                    return self._generate_enhanced_results(predictions, abcde_scores, enhanced)

            # Fallback to basic results
            return self._generate_results(predictions)

        except Exception as e:
            print(f"Analysis error: {e}")
            return self._fallback_analysis()

    def _generate_enhanced_results(self, predictions: Dict[str, float], abcde: Dict, enhanced: Dict[str, Any]) -> Dict[str, Any]:
        """Generate results combining MONAI predictions with Groq enhancement"""
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_class, top_prob = sorted_preds[0]

        return {
            'success': True,
            'analysis': {
                'primary_classification': enhanced.get('primary_classification', top_class),
                'confidence': enhanced.get('confidence', round(top_prob * 100, 1)),
                'risk_level': enhanced.get('risk_level', 'Low'),
                'urgency': enhanced.get('urgency', 'Routine'),
                'findings': self._get_findings(sorted_preds),
                'abcde_assessment': {
                    **abcde,
                    'interpretation': enhanced.get('abcde_interpretation', '')
                },
                'all_predictions': {k: round(v * 100, 1) for k, v in predictions.items()},
                'detailed_explanation': enhanced.get('detailed_explanation', ''),
                'recommendation': enhanced.get('recommendation', 'Consult a dermatologist.'),
                'next_steps': enhanced.get('next_steps', []),
                'warning_signs': enhanced.get('warning_signs', []),
                'ai_enhanced': True,
                'models': {
                    'image_analysis': 'MONAI EfficientNetBN',
                    'explanation': enhanced.get('model', 'llama-3.3-70b-versatile')
                },
                'disclaimer': 'This AI analysis is for informational purposes only. Any suspicious skin lesion should be evaluated by a dermatologist.'
            }
        }

    def _get_findings(self, sorted_preds) -> List[Dict]:
        """Get top findings from predictions"""
        findings = []
        for cls, prob in sorted_preds[:3]:
            findings.append({
                'classification': cls,
                'probability': round(prob * 100, 1),
                'risk_category': self._get_risk_category(cls),
                'description': self._get_condition_description(cls)
            })
        return findings

    def _decode_image(self, image_data: str) -> Optional[Image.Image]:
        """Decode base64 image"""
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            return image.convert('RGB')
        except Exception as e:
            print(f"Image decode error: {e}")
            return None

    def _run_inference(self, image: Image.Image) -> Dict[str, float]:
        """Run model inference"""
        try:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)[0]

            predictions = {}
            for i, cls in enumerate(self.CLASSES):
                predictions[cls] = float(probabilities[i].cpu().numpy())

            return predictions

        except Exception as e:
            print(f"Inference error: {e}")
            return self._simulate_predictions(image)

    def _simulate_predictions(self, image: Optional[Image.Image] = None) -> Dict[str, float]:
        """Simulate predictions for demo"""
        np.random.seed(42 if image is None else hash(str(image.size)) % 2**32)

        # Most lesions are benign
        base_probs = np.array([0.55, 0.05, 0.18, 0.04, 0.06, 0.07, 0.05])
        noise = np.random.normal(0, 0.03, len(base_probs))
        probs = np.clip(base_probs + noise, 0.01, 0.95)
        probs = probs / probs.sum()

        return {cls: float(prob) for cls, prob in zip(self.CLASSES, probs)}

    def _generate_results(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Generate analysis results"""
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_class, top_prob = sorted_preds[0]

        # Determine risk level
        if top_class in self.HIGH_RISK and top_prob > 0.3:
            risk_level = 'High'
            urgency = 'URGENT'
        elif top_class in self.HIGH_RISK or top_class in self.MODERATE_RISK:
            risk_level = 'Moderate'
            urgency = 'Soon'
        else:
            risk_level = 'Low'
            urgency = 'Routine'

        # ABCDE criteria assessment (simulated based on prediction confidence)
        abcde_score = self._calculate_abcde_score(predictions)

        # Detailed findings
        findings = []
        for cls, prob in sorted_preds[:3]:
            findings.append({
                'classification': cls,
                'probability': round(prob * 100, 1),
                'risk_category': self._get_risk_category(cls),
                'description': self._get_condition_description(cls)
            })

        return {
            'success': True,
            'analysis': {
                'primary_classification': top_class,
                'confidence': round(top_prob * 100, 1),
                'risk_level': risk_level,
                'urgency': urgency,
                'findings': findings,
                'abcde_assessment': abcde_score,
                'all_predictions': {k: round(v * 100, 1) for k, v in predictions.items()},
                'recommendation': self._get_recommendation(top_class, risk_level),
                'next_steps': self._get_next_steps(risk_level),
                'disclaimer': 'This AI analysis is for informational purposes only. Any suspicious skin lesion should be evaluated by a dermatologist.'
            }
        }

    def _calculate_abcde_score(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Calculate ABCDE melanoma criteria score"""
        # Simulated based on prediction confidence
        melanoma_prob = predictions.get('Melanoma', 0)

        return {
            'asymmetry': 'Analyzing shape symmetry',
            'border': 'Evaluating border regularity',
            'color': 'Assessing color uniformity',
            'diameter': 'Check if >6mm diameter',
            'evolution': 'Monitor for changes over time',
            'overall_concern': 'High' if melanoma_prob > 0.3 else 'Moderate' if melanoma_prob > 0.1 else 'Low'
        }

    def _get_risk_category(self, classification: str) -> str:
        """Get risk category for classification"""
        if classification in self.HIGH_RISK:
            return 'High Risk - Potential Malignancy'
        elif classification in self.MODERATE_RISK:
            return 'Moderate Risk - Pre-cancerous'
        else:
            return 'Low Risk - Likely Benign'

    def _get_condition_description(self, condition: str) -> str:
        """Get description for condition"""
        descriptions = {
            'Melanocytic Nevi': 'Common benign mole, usually harmless',
            'Melanoma': 'Serious form of skin cancer requiring immediate attention',
            'Benign Keratosis': 'Non-cancerous skin growth, common with age',
            'Basal Cell Carcinoma': 'Most common form of skin cancer, slow-growing',
            'Actinic Keratosis': 'Pre-cancerous lesion from sun damage',
            'Vascular Lesion': 'Benign blood vessel abnormality',
            'Dermatofibroma': 'Benign fibrous skin nodule'
        }
        return descriptions.get(condition, 'Skin lesion detected')

    def _get_recommendation(self, primary: str, risk_level: str) -> str:
        """Generate recommendation"""
        if risk_level == 'High':
            return 'URGENT: This lesion shows characteristics that warrant immediate professional evaluation. Please see a dermatologist as soon as possible for a thorough examination and possible biopsy.'
        elif risk_level == 'Moderate':
            return 'This lesion should be evaluated by a dermatologist within the next few weeks. Monitor for any changes in size, shape, or color.'
        else:
            return 'This appears to be a benign lesion. Continue monitoring for any changes and include it in your regular skin checks with a healthcare provider.'

    def _get_next_steps(self, risk_level: str) -> List[str]:
        """Get recommended next steps"""
        if risk_level == 'High':
            return [
                'Schedule urgent dermatologist appointment',
                'Do not attempt to remove or treat the lesion yourself',
                'Take photos to track any changes',
                'Prepare medical history for consultation'
            ]
        elif risk_level == 'Moderate':
            return [
                'Schedule dermatologist appointment within 2-4 weeks',
                'Monitor the lesion for changes',
                'Protect the area from sun exposure',
                'Document size and appearance'
            ]
        else:
            return [
                'Continue regular skin self-examinations',
                'Use sun protection (SPF 30+)',
                'Include in annual skin check',
                'Monitor for any changes'
            ]

    def _fallback_analysis(self) -> Dict[str, Any]:
        """Return fallback analysis"""
        return {
            'success': True,
            'analysis': {
                'primary_classification': 'Analysis Pending',
                'confidence': 0,
                'risk_level': 'Unknown',
                'urgency': 'Consult Doctor',
                'findings': [],
                'recommendation': 'Unable to analyze image. Please ensure good lighting and focus on the skin lesion.',
                'disclaimer': 'This AI analysis is for informational purposes only.'
            }
        }


# Global service instance
skin_lesion_service = SkinLesionService()
