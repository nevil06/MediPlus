"""
AI Vision Service - Enhances MONAI results with Groq LLM
MONAI handles image analysis, Groq provides detailed explanations
"""

import os
import json
from typing import Dict, Any
import requests


class AIEnhancementService:
    """Service to enhance MONAI predictions with Groq LLM explanations"""

    def __init__(self):
        self.api_key = os.environ.get('GROQ_API_KEY')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"

        if not self.api_key:
            print("Warning: GROQ_API_KEY not set. AI enhancement will be disabled.")

    def enhance_chest_xray_analysis(
        self,
        image_data: str = None,
        monai_predictions: Dict[str, Any] = None,
        analysis_type: str = 'chest_xray'
    ) -> Dict[str, Any]:
        """Alias for enhance_chest_xray_result with flexible parameters"""
        predictions = {}
        if monai_predictions:
            for condition, data in monai_predictions.items():
                if isinstance(data, dict):
                    predictions[condition] = data.get('probability', 0)
                else:
                    predictions[condition] = data
        return self.enhance_chest_xray_result(predictions)

    def enhance_chest_xray_result(self, monai_predictions: Dict[str, float]) -> Dict[str, Any]:
        """Enhance MONAI chest X-ray predictions with detailed Groq explanation"""

        prompt = f"""You are a medical AI assistant. Based on the following MONAI deep learning model predictions for a chest X-ray analysis, provide a detailed patient-friendly explanation.

MONAI Model Predictions (condition: probability %):
{json.dumps(monai_predictions, indent=2)}

Provide your response in this exact JSON format:
{{
    "primary_finding": "the condition with highest probability",
    "confidence": the highest probability value,
    "risk_level": "Low", "Moderate", or "High" based on findings,
    "detailed_explanation": "2-3 sentence explanation of what these findings mean",
    "recommendation": "specific actionable recommendation for the patient",
    "when_to_seek_help": "specific symptoms that warrant immediate medical attention",
    "lifestyle_advice": ["list of 3-4 relevant lifestyle tips"]
}}

Respond ONLY with valid JSON."""

        return self._get_groq_enhancement(prompt)

    def enhance_skin_analysis(
        self,
        image_data: str = None,
        monai_predictions: Dict[str, Any] = None,
        abcde_scores: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Alias for enhance_skin_lesion_result with flexible parameters"""
        predictions = {}
        if monai_predictions:
            for condition, data in monai_predictions.items():
                if isinstance(data, dict):
                    predictions[condition] = data.get('probability', 0)
                else:
                    predictions[condition] = data
        return self.enhance_skin_lesion_result(predictions, abcde_scores or {})

    def enhance_skin_lesion_result(self, monai_predictions: Dict[str, float], abcde_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance MONAI skin lesion predictions with Groq explanation"""

        prompt = f"""You are a dermatology AI assistant. Based on the following MONAI model predictions for skin lesion analysis, provide a detailed explanation.

MONAI Model Predictions (classification: probability %):
{json.dumps(monai_predictions, indent=2)}

ABCDE Assessment Scores:
{json.dumps(abcde_scores, indent=2)}

High-risk conditions: Melanoma, Basal Cell Carcinoma
Moderate-risk: Actinic Keratosis

Provide your response in this exact JSON format:
{{
    "primary_classification": "most likely classification",
    "confidence": highest probability value,
    "risk_level": "Low", "Moderate", or "High",
    "urgency": "Routine", "Soon", or "URGENT",
    "detailed_explanation": "explanation of findings in patient-friendly terms",
    "abcde_interpretation": "what the ABCDE scores indicate",
    "recommendation": "specific recommendation",
    "next_steps": ["list of recommended actions"],
    "warning_signs": ["signs that require immediate attention"]
}}

Respond ONLY with valid JSON."""

        return self._get_groq_enhancement(prompt)

    def enhance_eye_analysis(
        self,
        image_data: str = None,
        monai_predictions: Dict[str, Any] = None,
        dr_grade: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Alias for enhance_eye_health_result with flexible parameters"""
        dr_predictions = {}
        if dr_grade and 'all_grades' in dr_grade:
            dr_predictions = dr_grade['all_grades']
        elif monai_predictions:
            for condition, data in monai_predictions.items():
                if isinstance(data, dict):
                    dr_predictions[condition] = data.get('probability', 0)
                else:
                    dr_predictions[condition] = data
        return self.enhance_eye_health_result(dr_predictions, {})

    def enhance_eye_health_result(self, dr_predictions: Dict[str, float], other_findings: Dict[str, float]) -> Dict[str, Any]:
        """Enhance MONAI eye health predictions with Groq explanation"""

        prompt = f"""You are an ophthalmology AI assistant. Based on the following MONAI model predictions for retinal image analysis, provide a detailed explanation.

Diabetic Retinopathy Predictions (grade: probability %):
{json.dumps(dr_predictions, indent=2)}

Other Condition Predictions:
{json.dumps(other_findings, indent=2)}

DR Grade Reference:
- No DR (0): No diabetic retinopathy
- Mild DR (1): Microaneurysms only
- Moderate DR (2): More than mild but less than severe
- Severe DR (3): Significant hemorrhages, venous changes
- Proliferative DR (4): New abnormal blood vessel growth

Provide your response in this exact JSON format:
{{
    "dr_grade": "detected grade name",
    "dr_grade_number": 0-4,
    "confidence": highest DR probability,
    "risk_level": "Low", "Moderate", or "High",
    "urgency": "Annual", "Routine", "Soon", or "URGENT",
    "detailed_explanation": "patient-friendly explanation of findings",
    "other_concerns": "explanation of any other detected conditions",
    "recommendation": "specific recommendation based on severity",
    "follow_up_schedule": "when to get next examination",
    "lifestyle_tips": ["diabetes and eye health management tips"],
    "warning_signs": ["symptoms requiring immediate attention"]
}}

Respond ONLY with valid JSON."""

        return self._get_groq_enhancement(prompt)

    def enhance_segmentation_analysis(
        self,
        statistics: Dict[str, Any] = None,
        task: str = None
    ) -> Dict[str, Any]:
        """Enhance segmentation results with Groq explanation"""
        if not statistics:
            return {'enhanced': False, 'error': 'No statistics provided'}

        prompt = f"""You are a medical imaging AI assistant. Based on the following segmentation statistics for a {task or 'medical image'} analysis, provide a detailed interpretation.

Segmentation Statistics:
{json.dumps(statistics, indent=2)}

Provide your response in this exact JSON format:
{{
    "interpretation": "detailed interpretation of what the segmentation shows",
    "findings": ["list of key findings from the segmentation"],
    "clinical_significance": "what these findings might indicate clinically",
    "recommendations": ["list of recommended follow-up actions"],
    "confidence_note": "note about the confidence level and limitations"
}}

Respond ONLY with valid JSON."""

        return self._get_groq_enhancement(prompt)

    def _get_groq_enhancement(self, prompt: str) -> Dict[str, Any]:
        """Call Groq API to get enhanced explanation"""

        if not self.api_key:
            return {'enhanced': False, 'error': 'Groq API key not configured'}

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant that provides clear, accurate, patient-friendly explanations. Always recommend consulting healthcare professionals. Respond only in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.3
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                print(f"Groq API error: {response.status_code}")
                return {'enhanced': False, 'error': 'Groq API error'}

            data = response.json()
            response_text = data['choices'][0]['message']['content'].strip()

            # Clean markdown if present
            if '```' in response_text:
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            result = json.loads(response_text)
            result['enhanced'] = True
            result['model'] = self.model
            return result

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return {'enhanced': False, 'error': 'Failed to parse response'}
        except Exception as e:
            print(f"Groq enhancement failed: {e}")
            return {'enhanced': False, 'error': str(e)}


# Global service instance
ai_enhancement_service = AIEnhancementService()
