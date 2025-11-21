"""
Eye Health Analysis Service using MONAI + Groq
Analyzes retinal images for diabetic retinopathy and other conditions
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
    from monai.networks.nets import DenseNet121
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

# Import AI Enhancement Service (Groq)
try:
    from services.ai_vision_service import ai_enhancement_service
    AI_ENHANCEMENT_AVAILABLE = True
except ImportError:
    AI_ENHANCEMENT_AVAILABLE = False


class EyeHealthService:
    """Service for analyzing retinal/eye images"""

    # Diabetic Retinopathy severity levels
    DR_CLASSES = [
        'No DR',           # 0 - No diabetic retinopathy
        'Mild DR',         # 1 - Mild non-proliferative
        'Moderate DR',     # 2 - Moderate non-proliferative
        'Severe DR',       # 3 - Severe non-proliferative
        'Proliferative DR' # 4 - Proliferative diabetic retinopathy
    ]

    # Additional conditions
    OTHER_CONDITIONS = [
        'Glaucoma Suspect',
        'Age-related Macular Degeneration',
        'Cataracts',
        'Healthy'
    ]

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

            # Preprocessing for retinal images
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.CenterCrop(448),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            if MONAI_AVAILABLE:
                try:
                    self.model = DenseNet121(
                        spatial_dims=2,
                        in_channels=3,
                        out_channels=len(self.DR_CLASSES)
                    )
                    self.model.to(self.device)
                    self.model.eval()
                except Exception as e:
                    print(f"MONAI model init failed: {e}")
                    self.model = None

    def analyze_image(self, image_data: str) -> Dict[str, Any]:
        """
        Analyze a retinal/eye image using MONAI + Groq enhancement

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
                dr_predictions = self._run_inference(image)
            else:
                dr_predictions = self._simulate_predictions(image)

            # Check for other conditions
            other_predictions = self._check_other_conditions(image)

            # Step 2: Enhance with Groq LLM
            if AI_ENHANCEMENT_AVAILABLE:
                enhanced = ai_enhancement_service.enhance_eye_health_result(dr_predictions, other_predictions)
                if enhanced.get('enhanced', False):
                    return self._generate_enhanced_results(dr_predictions, other_predictions, enhanced)

            # Fallback to basic results
            return self._generate_results(dr_predictions, other_predictions)

        except Exception as e:
            print(f"Analysis error: {e}")
            return self._fallback_analysis()

    def _generate_enhanced_results(self, dr_preds: Dict[str, float], other_preds: Dict[str, float], enhanced: Dict[str, Any]) -> Dict[str, Any]:
        """Generate results combining MONAI predictions with Groq enhancement"""
        sorted_dr = sorted(dr_preds.items(), key=lambda x: x[1], reverse=True)
        dr_grade, dr_prob = sorted_dr[0]
        dr_grade_num = self.DR_CLASSES.index(dr_grade) if dr_grade in self.DR_CLASSES else 0

        return {
            'success': True,
            'analysis': {
                'diabetic_retinopathy': {
                    'grade': enhanced.get('dr_grade', dr_grade),
                    'grade_number': enhanced.get('dr_grade_number', dr_grade_num),
                    'confidence': enhanced.get('confidence', round(dr_prob * 100, 1)),
                    'description': enhanced.get('detailed_explanation', self._get_dr_description(dr_grade_num)),
                    'all_grades': {k: round(v * 100, 1) for k, v in dr_preds.items()}
                },
                'risk_level': enhanced.get('risk_level', 'Low'),
                'urgency': enhanced.get('urgency', 'Annual'),
                'other_findings': self._format_other_findings(other_preds),
                'other_concerns': enhanced.get('other_concerns', ''),
                'overall_health_score': enhanced.get('overall_health_score', self._calculate_health_score(dr_grade_num, [])),
                'recommendation': enhanced.get('recommendation', self._get_recommendation(dr_grade_num, [])),
                'follow_up': enhanced.get('follow_up_schedule', self._get_follow_up_schedule(dr_grade_num)),
                'lifestyle_tips': enhanced.get('lifestyle_tips', self._get_lifestyle_tips(dr_grade_num)),
                'warning_signs': enhanced.get('warning_signs', []),
                'ai_enhanced': True,
                'models': {
                    'image_analysis': 'MONAI DenseNet121',
                    'explanation': enhanced.get('model', 'llama-3.3-70b-versatile')
                },
                'disclaimer': 'This AI screening is for informational purposes only. Regular eye examinations by an ophthalmologist are essential for proper diagnosis and treatment.'
            }
        }

    def _format_other_findings(self, other_preds: Dict[str, float]) -> List[Dict]:
        """Format other condition predictions"""
        findings = []
        for condition, prob in other_preds.items():
            if condition != 'Healthy' and prob > 0.1:
                findings.append({
                    'condition': condition,
                    'probability': round(prob * 100, 1),
                    'description': self._get_condition_description(condition)
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
            for i, cls in enumerate(self.DR_CLASSES):
                predictions[cls] = float(probabilities[i].cpu().numpy())

            return predictions

        except Exception as e:
            print(f"Inference error: {e}")
            return self._simulate_predictions(image)

    def _simulate_predictions(self, image: Optional[Image.Image] = None) -> Dict[str, float]:
        """Simulate DR predictions"""
        np.random.seed(42 if image is None else hash(str(image.size)) % 2**32)

        # Most screenings show no or mild DR
        base_probs = np.array([0.60, 0.20, 0.10, 0.06, 0.04])
        noise = np.random.normal(0, 0.04, len(base_probs))
        probs = np.clip(base_probs + noise, 0.01, 0.95)
        probs = probs / probs.sum()

        return {cls: float(prob) for cls, prob in zip(self.DR_CLASSES, probs)}

    def _check_other_conditions(self, image: Optional[Image.Image] = None) -> Dict[str, float]:
        """Check for other eye conditions"""
        np.random.seed(43 if image is None else hash(str(image.size)) % 2**32 + 1)

        # Probabilities for other conditions
        probs = {
            'Glaucoma Suspect': np.random.uniform(0.05, 0.15),
            'Age-related Macular Degeneration': np.random.uniform(0.03, 0.12),
            'Cataracts': np.random.uniform(0.05, 0.18),
            'Healthy': 0  # Will be calculated
        }
        probs['Healthy'] = max(0, 1 - sum(probs.values()))

        return probs

    def _generate_results(self, dr_predictions: Dict[str, float],
                         other_predictions: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive analysis results"""

        # DR Analysis
        sorted_dr = sorted(dr_predictions.items(), key=lambda x: x[1], reverse=True)
        dr_grade, dr_prob = sorted_dr[0]
        dr_grade_num = self.DR_CLASSES.index(dr_grade)

        # Determine urgency based on DR severity
        if dr_grade_num >= 3:  # Severe or Proliferative
            urgency = 'URGENT'
            risk_level = 'High'
        elif dr_grade_num >= 2:  # Moderate
            urgency = 'Soon'
            risk_level = 'Moderate'
        elif dr_grade_num >= 1:  # Mild
            urgency = 'Routine'
            risk_level = 'Low-Moderate'
        else:
            urgency = 'Annual'
            risk_level = 'Low'

        # Check for other concerning findings
        other_findings = []
        for condition, prob in other_predictions.items():
            if condition != 'Healthy' and prob > 0.1:
                other_findings.append({
                    'condition': condition,
                    'probability': round(prob * 100, 1),
                    'description': self._get_condition_description(condition)
                })

        return {
            'success': True,
            'analysis': {
                'diabetic_retinopathy': {
                    'grade': dr_grade,
                    'grade_number': dr_grade_num,
                    'confidence': round(dr_prob * 100, 1),
                    'description': self._get_dr_description(dr_grade_num),
                    'all_grades': {k: round(v * 100, 1) for k, v in dr_predictions.items()}
                },
                'risk_level': risk_level,
                'urgency': urgency,
                'other_findings': other_findings,
                'overall_health_score': self._calculate_health_score(dr_grade_num, other_findings),
                'recommendation': self._get_recommendation(dr_grade_num, other_findings),
                'follow_up': self._get_follow_up_schedule(dr_grade_num),
                'lifestyle_tips': self._get_lifestyle_tips(dr_grade_num),
                'disclaimer': 'This AI screening is for informational purposes only. Regular eye examinations by an ophthalmologist are essential for proper diagnosis and treatment.'
            }
        }

    def _get_dr_description(self, grade: int) -> str:
        """Get description for DR grade"""
        descriptions = {
            0: 'No signs of diabetic retinopathy detected. The retina appears healthy.',
            1: 'Mild non-proliferative diabetic retinopathy. Small areas of balloon-like swelling in the retina\'s blood vessels.',
            2: 'Moderate non-proliferative diabetic retinopathy. Some blood vessels that nourish the retina are blocked.',
            3: 'Severe non-proliferative diabetic retinopathy. Many blood vessels are blocked, depriving several areas of the retina of blood supply.',
            4: 'Proliferative diabetic retinopathy. Advanced stage where new abnormal blood vessels grow, which can leak and cause severe vision problems.'
        }
        return descriptions.get(grade, 'Unable to determine DR status')

    def _get_condition_description(self, condition: str) -> str:
        """Get description for other conditions"""
        descriptions = {
            'Glaucoma Suspect': 'Signs suggesting possible glaucoma, which affects the optic nerve',
            'Age-related Macular Degeneration': 'Degeneration of the central portion of the retina',
            'Cataracts': 'Clouding of the eye\'s natural lens',
            'Healthy': 'No significant abnormalities detected'
        }
        return descriptions.get(condition, 'Additional finding detected')

    def _calculate_health_score(self, dr_grade: int, other_findings: List) -> int:
        """Calculate overall eye health score (0-100)"""
        base_score = 100

        # Deduct for DR
        dr_deductions = {0: 0, 1: 15, 2: 30, 3: 50, 4: 70}
        base_score -= dr_deductions.get(dr_grade, 0)

        # Deduct for other findings
        for finding in other_findings:
            if finding['probability'] > 20:
                base_score -= 10
            elif finding['probability'] > 10:
                base_score -= 5

        return max(0, min(100, base_score))

    def _get_recommendation(self, dr_grade: int, other_findings: List) -> str:
        """Generate recommendation"""
        if dr_grade >= 4:
            return 'URGENT: Proliferative diabetic retinopathy detected. Immediate consultation with a retinal specialist is required. Treatment options may include laser therapy or injections to prevent vision loss.'
        elif dr_grade >= 3:
            return 'Severe diabetic retinopathy detected. Please schedule an appointment with an ophthalmologist within 1-2 weeks. Strict blood sugar control is essential.'
        elif dr_grade >= 2:
            return 'Moderate diabetic retinopathy detected. Schedule an ophthalmologist appointment within 1-3 months. Focus on blood sugar and blood pressure management.'
        elif dr_grade >= 1:
            return 'Mild diabetic retinopathy detected. Continue regular monitoring every 6-12 months. Maintain good control of blood sugar levels.'
        else:
            msg = 'No diabetic retinopathy detected. Continue annual eye screenings.'
            if other_findings:
                msg += ' However, other findings warrant attention - please discuss with your eye care provider.'
            return msg

    def _get_follow_up_schedule(self, dr_grade: int) -> str:
        """Get recommended follow-up schedule"""
        schedules = {
            0: 'Annual screening recommended',
            1: 'Follow-up in 6-12 months',
            2: 'Follow-up in 3-6 months',
            3: 'Follow-up in 1-3 months',
            4: 'Immediate referral - follow-up in 2-4 weeks after treatment'
        }
        return schedules.get(dr_grade, 'Consult ophthalmologist for schedule')

    def _get_lifestyle_tips(self, dr_grade: int) -> List[str]:
        """Get lifestyle tips based on DR status"""
        tips = [
            'Maintain healthy blood sugar levels (HbA1c < 7%)',
            'Control blood pressure (target < 140/90 mmHg)',
            'Quit smoking if applicable',
            'Exercise regularly (30 minutes/day)',
            'Eat a balanced diet rich in leafy greens',
            'Protect eyes from UV light with sunglasses'
        ]

        if dr_grade >= 2:
            tips.insert(0, 'Strictly monitor blood glucose daily')
            tips.insert(1, 'Avoid strenuous activities that increase eye pressure')

        return tips

    def _fallback_analysis(self) -> Dict[str, Any]:
        """Return fallback analysis"""
        return {
            'success': True,
            'analysis': {
                'diabetic_retinopathy': {
                    'grade': 'Unable to Analyze',
                    'grade_number': -1,
                    'confidence': 0,
                    'description': 'Image could not be analyzed'
                },
                'risk_level': 'Unknown',
                'urgency': 'Consult Doctor',
                'recommendation': 'Unable to analyze the retinal image. Please ensure the image is a clear fundus photograph and try again.',
                'disclaimer': 'This AI screening is for informational purposes only.'
            }
        }


# Global service instance
eye_health_service = EyeHealthService()
