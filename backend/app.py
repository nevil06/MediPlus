"""
MediPlus Backend Server
Flask-based API server for heart disease prediction using MONAI/PyTorch
Optimized for Intel i7 Ultra + Iris GPU
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Import services (lazy loading to handle missing dependencies gracefully)
heart_predictor = None
chest_xray_service = None
skin_lesion_service = None
eye_health_service = None

def get_heart_predictor():
    """Lazy load heart disease predictor"""
    global heart_predictor
    if heart_predictor is None:
        try:
            from services.heart_disease_service import HeartDiseasePredictor
            heart_predictor = HeartDiseasePredictor()
            logger.info("Heart disease predictor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load heart predictor: {e}")
            heart_predictor = None
    return heart_predictor

def get_chest_xray_service():
    """Lazy load chest X-ray analysis service"""
    global chest_xray_service
    if chest_xray_service is None:
        try:
            from services.chest_xray_service import ChestXRayService
            chest_xray_service = ChestXRayService()
            logger.info("Chest X-Ray service loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load chest X-ray service: {e}")
            chest_xray_service = None
    return chest_xray_service

def get_skin_lesion_service():
    """Lazy load skin lesion analysis service"""
    global skin_lesion_service
    if skin_lesion_service is None:
        try:
            from services.skin_lesion_service import SkinLesionService
            skin_lesion_service = SkinLesionService()
            logger.info("Skin lesion service loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load skin lesion service: {e}")
            skin_lesion_service = None
    return skin_lesion_service

def get_eye_health_service():
    """Lazy load eye health analysis service"""
    global eye_health_service
    if eye_health_service is None:
        try:
            from services.eye_health_service import EyeHealthService
            eye_health_service = EyeHealthService()
            logger.info("Eye health service loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load eye health service: {e}")
            eye_health_service = None
    return eye_health_service


# ============== Health Check ==============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'MediPlus Backend',
        'version': '1.0.0'
    })


# ============== Heart Disease Prediction ==============

@app.route('/api/predict/heart', methods=['POST'])
def predict_heart_disease():
    """
    Predict heart disease risk from health metrics

    Expected JSON payload:
    {
        "age": 45,
        "sex": 1,  # 1=male, 0=female
        "cp": 2,   # chest pain type (0-3)
        "trestbps": 130,  # resting blood pressure
        "chol": 250,  # cholesterol mg/dl
        "fbs": 0,  # fasting blood sugar > 120 mg/dl
        "restecg": 1,  # resting ECG results (0-2)
        "thalach": 150,  # max heart rate achieved
        "exang": 0,  # exercise induced angina
        "oldpeak": 1.5,  # ST depression
        "slope": 1,  # slope of peak exercise ST (0-2)
        "ca": 0,  # number of major vessels (0-3)
        "thal": 2  # thalassemia (0-3)
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # Validate required fields
        required_fields = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]

        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing fields: {missing_fields}'
            }), 400

        # Get predictor
        predictor = get_heart_predictor()

        if predictor is None:
            # Fallback to rule-based prediction if ML model not available
            prediction = rule_based_heart_prediction(data)
        else:
            prediction = predictor.predict(data)

        return jsonify({
            'success': True,
            'prediction': prediction
        })

    except Exception as e:
        logger.error(f"Heart prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def rule_based_heart_prediction(data):
    """
    Rule-based fallback prediction when ML model is not available
    Based on Framingham Risk Score factors
    """
    risk_score = 0
    risk_factors = []

    # Age risk
    age = data.get('age', 0)
    if age >= 65:
        risk_score += 3
        risk_factors.append("Age 65+")
    elif age >= 55:
        risk_score += 2
        risk_factors.append("Age 55-64")
    elif age >= 45:
        risk_score += 1
        risk_factors.append("Age 45-54")

    # Blood pressure
    bp = data.get('trestbps', 120)
    if bp >= 160:
        risk_score += 3
        risk_factors.append("High blood pressure (Stage 2)")
    elif bp >= 140:
        risk_score += 2
        risk_factors.append("High blood pressure (Stage 1)")
    elif bp >= 130:
        risk_score += 1
        risk_factors.append("Elevated blood pressure")

    # Cholesterol
    chol = data.get('chol', 200)
    if chol >= 280:
        risk_score += 3
        risk_factors.append("Very high cholesterol")
    elif chol >= 240:
        risk_score += 2
        risk_factors.append("High cholesterol")
    elif chol >= 200:
        risk_score += 1
        risk_factors.append("Borderline high cholesterol")

    # Fasting blood sugar
    if data.get('fbs', 0) == 1:
        risk_score += 2
        risk_factors.append("High fasting blood sugar")

    # Chest pain type
    cp = data.get('cp', 0)
    if cp == 0:  # Typical angina
        risk_score += 3
        risk_factors.append("Typical angina")
    elif cp == 1:  # Atypical angina
        risk_score += 2
        risk_factors.append("Atypical angina")
    elif cp == 2:  # Non-anginal pain
        risk_score += 1
        risk_factors.append("Non-anginal chest pain")

    # Max heart rate
    thalach = data.get('thalach', 150)
    expected_max = 220 - age
    if thalach < expected_max * 0.6:
        risk_score += 2
        risk_factors.append("Low max heart rate")

    # Exercise induced angina
    if data.get('exang', 0) == 1:
        risk_score += 2
        risk_factors.append("Exercise induced angina")

    # ST depression
    oldpeak = data.get('oldpeak', 0)
    if oldpeak >= 2:
        risk_score += 3
        risk_factors.append("Significant ST depression")
    elif oldpeak >= 1:
        risk_score += 1
        risk_factors.append("Mild ST depression")

    # Number of major vessels
    ca = data.get('ca', 0)
    if ca >= 2:
        risk_score += 3
        risk_factors.append(f"{ca} major vessels colored")
    elif ca == 1:
        risk_score += 1
        risk_factors.append("1 major vessel colored")

    # Calculate risk level
    max_score = 25
    risk_percentage = min((risk_score / max_score) * 100, 100)

    if risk_percentage >= 70:
        risk_level = "High"
        recommendation = "Immediate medical consultation recommended"
    elif risk_percentage >= 40:
        risk_level = "Moderate"
        recommendation = "Schedule a check-up with your doctor"
    else:
        risk_level = "Low"
        recommendation = "Maintain healthy lifestyle"

    return {
        'risk_score': round(risk_percentage, 1),
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'recommendation': recommendation,
        'model_type': 'rule_based'
    }


# ============== Quick Health Assessment ==============

@app.route('/api/assess/quick', methods=['POST'])
def quick_health_assessment():
    """
    Quick health assessment based on basic vitals

    Expected JSON:
    {
        "heart_rate": 72,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "blood_sugar": 100,
        "oxygen_level": 98
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        assessments = []
        overall_status = "Good"

        # Heart rate assessment
        hr = data.get('heart_rate')
        if hr:
            if hr < 60:
                assessments.append({
                    'metric': 'Heart Rate',
                    'value': hr,
                    'status': 'Low',
                    'message': 'Heart rate is below normal (bradycardia)'
                })
            elif hr > 100:
                assessments.append({
                    'metric': 'Heart Rate',
                    'value': hr,
                    'status': 'High',
                    'message': 'Heart rate is above normal (tachycardia)'
                })
                overall_status = "Attention Needed"
            else:
                assessments.append({
                    'metric': 'Heart Rate',
                    'value': hr,
                    'status': 'Normal',
                    'message': 'Heart rate is within normal range'
                })

        # Blood pressure assessment
        sys_bp = data.get('systolic_bp')
        dia_bp = data.get('diastolic_bp')
        if sys_bp and dia_bp:
            if sys_bp >= 180 or dia_bp >= 120:
                assessments.append({
                    'metric': 'Blood Pressure',
                    'value': f'{sys_bp}/{dia_bp}',
                    'status': 'Critical',
                    'message': 'Hypertensive crisis - seek immediate medical attention'
                })
                overall_status = "Critical"
            elif sys_bp >= 140 or dia_bp >= 90:
                assessments.append({
                    'metric': 'Blood Pressure',
                    'value': f'{sys_bp}/{dia_bp}',
                    'status': 'High',
                    'message': 'High blood pressure (Stage 2 hypertension)'
                })
                overall_status = "Attention Needed"
            elif sys_bp >= 130 or dia_bp >= 80:
                assessments.append({
                    'metric': 'Blood Pressure',
                    'value': f'{sys_bp}/{dia_bp}',
                    'status': 'Elevated',
                    'message': 'Elevated blood pressure'
                })
            else:
                assessments.append({
                    'metric': 'Blood Pressure',
                    'value': f'{sys_bp}/{dia_bp}',
                    'status': 'Normal',
                    'message': 'Blood pressure is within normal range'
                })

        # Blood sugar assessment
        bs = data.get('blood_sugar')
        if bs:
            if bs < 70:
                assessments.append({
                    'metric': 'Blood Sugar',
                    'value': bs,
                    'status': 'Low',
                    'message': 'Blood sugar is low (hypoglycemia)'
                })
                overall_status = "Attention Needed"
            elif bs > 180:
                assessments.append({
                    'metric': 'Blood Sugar',
                    'value': bs,
                    'status': 'High',
                    'message': 'Blood sugar is high - consult doctor'
                })
                overall_status = "Attention Needed"
            elif bs > 140:
                assessments.append({
                    'metric': 'Blood Sugar',
                    'value': bs,
                    'status': 'Elevated',
                    'message': 'Blood sugar is slightly elevated'
                })
            else:
                assessments.append({
                    'metric': 'Blood Sugar',
                    'value': bs,
                    'status': 'Normal',
                    'message': 'Blood sugar is within normal range'
                })

        # Oxygen level assessment
        o2 = data.get('oxygen_level')
        if o2:
            if o2 < 90:
                assessments.append({
                    'metric': 'Oxygen Level',
                    'value': o2,
                    'status': 'Critical',
                    'message': 'Oxygen level critically low - seek immediate help'
                })
                overall_status = "Critical"
            elif o2 < 95:
                assessments.append({
                    'metric': 'Oxygen Level',
                    'value': o2,
                    'status': 'Low',
                    'message': 'Oxygen level is below normal'
                })
                overall_status = "Attention Needed"
            else:
                assessments.append({
                    'metric': 'Oxygen Level',
                    'value': o2,
                    'status': 'Normal',
                    'message': 'Oxygen level is healthy'
                })

        return jsonify({
            'success': True,
            'overall_status': overall_status,
            'assessments': assessments,
            'timestamp': str(np.datetime64('now'))
        })

    except Exception as e:
        logger.error(f"Quick assessment error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============== Medicine Information ==============

@app.route('/api/medicine/info', methods=['POST'])
def get_medicine_info():
    """
    Get medicine information (placeholder for future OCR integration)

    Expected JSON:
    {
        "medicine_name": "Aspirin"
    }
    """
    try:
        data = request.get_json()
        medicine_name = data.get('medicine_name', '').lower()

        # Basic medicine database (expandable)
        medicine_db = {
            'aspirin': {
                'name': 'Aspirin',
                'generic_name': 'Acetylsalicylic acid',
                'category': 'NSAID / Blood thinner',
                'uses': ['Pain relief', 'Fever reduction', 'Heart attack prevention'],
                'dosage': '325-650mg every 4-6 hours as needed',
                'warnings': ['Do not take with blood thinners', 'Avoid if allergic to NSAIDs'],
                'side_effects': ['Stomach upset', 'Heartburn', 'Bleeding risk']
            },
            'metformin': {
                'name': 'Metformin',
                'generic_name': 'Metformin hydrochloride',
                'category': 'Antidiabetic',
                'uses': ['Type 2 diabetes management', 'Blood sugar control'],
                'dosage': '500-2000mg daily with meals',
                'warnings': ['Monitor kidney function', 'Avoid alcohol'],
                'side_effects': ['Nausea', 'Diarrhea', 'Vitamin B12 deficiency']
            },
            'lisinopril': {
                'name': 'Lisinopril',
                'generic_name': 'Lisinopril',
                'category': 'ACE Inhibitor',
                'uses': ['High blood pressure', 'Heart failure', 'Post heart attack'],
                'dosage': '10-40mg once daily',
                'warnings': ['Do not use during pregnancy', 'Monitor potassium levels'],
                'side_effects': ['Dry cough', 'Dizziness', 'Headache']
            },
            'atorvastatin': {
                'name': 'Atorvastatin',
                'generic_name': 'Atorvastatin calcium',
                'category': 'Statin',
                'uses': ['High cholesterol', 'Heart disease prevention'],
                'dosage': '10-80mg once daily',
                'warnings': ['Monitor liver function', 'Report muscle pain'],
                'side_effects': ['Muscle pain', 'Digestive issues', 'Headache']
            }
        }

        if medicine_name in medicine_db:
            return jsonify({
                'success': True,
                'medicine': medicine_db[medicine_name]
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Medicine not found in database',
                'suggestion': 'Please verify the medicine name'
            }), 404

    except Exception as e:
        logger.error(f"Medicine info error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============== Image Analysis Endpoints ==============

@app.route('/api/analyze/chest-xray', methods=['POST'])
def analyze_chest_xray():
    """
    Analyze chest X-ray image for various conditions

    Expected JSON payload:
    {
        "image": "base64_encoded_image_string"
    }

    Returns analysis for: Pneumonia, COVID-19, Cardiomegaly, Lung Nodule, etc.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        image_data = data.get('image')
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400

        # Get service
        service = get_chest_xray_service()

        if service is None:
            return jsonify({
                'success': False,
                'error': 'Chest X-ray analysis service not available'
            }), 503

        # Analyze image
        result = service.analyze_image(image_data)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Chest X-ray analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analyze/skin', methods=['POST'])
def analyze_skin_lesion():
    """
    Analyze skin lesion image for potential conditions

    Expected JSON payload:
    {
        "image": "base64_encoded_image_string"
    }

    Returns analysis for: Melanoma, Basal Cell Carcinoma, Benign Keratosis, etc.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        image_data = data.get('image')
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400

        # Get service
        service = get_skin_lesion_service()

        if service is None:
            return jsonify({
                'success': False,
                'error': 'Skin lesion analysis service not available'
            }), 503

        # Analyze image
        result = service.analyze_image(image_data)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Skin lesion analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analyze/eye', methods=['POST'])
def analyze_eye_health():
    """
    Analyze retinal/eye image for diabetic retinopathy and other conditions

    Expected JSON payload:
    {
        "image": "base64_encoded_image_string"
    }

    Returns analysis for: Diabetic Retinopathy (grades 0-4), Glaucoma, AMD, Cataracts
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        image_data = data.get('image')
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400

        # Get service
        service = get_eye_health_service()

        if service is None:
            return jsonify({
                'success': False,
                'error': 'Eye health analysis service not available'
            }), 503

        # Analyze image
        result = service.analyze_image(image_data)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Eye health analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============== Server Configuration ==============

if __name__ == '__main__':
    # Get configuration from environment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'

    logger.info(f"Starting MediPlus Backend on {host}:{port}")
    logger.info("Optimized for Intel i7 Ultra + Iris GPU")

    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )
