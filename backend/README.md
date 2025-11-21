# MediPlus Backend

Flask-based API server for medical AI analysis using MONAI/PyTorch with Groq LLM enhancement.
Optimized for Intel i7 Ultra + Iris GPU.

## Features

- **Heart Disease Prediction**: ML-powered risk assessment using clinical parameters
- **Chest X-Ray Analysis**: MONAI DenseNet121 + Groq LLM for pneumonia, COVID-19 detection
- **Skin Lesion Analysis**: MONAI EfficientNet + Groq for melanoma screening
- **Eye Health Screening**: Diabetic retinopathy grading with AI recommendations
- **AI Chatbot**: Groq-powered medical information assistant
- **Intel Optimized**: IPEX and OpenVINO support for Intel hardware

## Tech Stack

- **Framework**: Flask + Flask-CORS
- **ML**: PyTorch + MONAI (DenseNet121, EfficientNet)
- **LLM**: Groq API (llama-3.3-70b-versatile)
- **Optimization**: Intel Extension for PyTorch (IPEX), OpenVINO

## Setup

### 1. Create Virtual Environment

```bash
cd backend
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Standard installation
pip install -r requirements.txt

# For Intel i7 Ultra + Iris GPU optimization
pip install intel-extension-for-pytorch
pip install openvino openvino-dev
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Groq API key
# Get your key at: https://console.groq.com/keys
```

### 4. Run Server

```bash
# Development
python app.py

# Production
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Server runs at `http://localhost:5000`

## Environment Variables

```bash
# Server Configuration
HOST=0.0.0.0
PORT=5000
DEBUG=True

# Groq API Key (required for AI features)
GROQ_API_KEY=your_groq_api_key_here

# Intel Optimization (optional)
INTEL_OPTIMIZATION=false
OPENVINO_DEVICE=CPU
```

## API Endpoints

### Health Check
```
GET /api/health
```

### Heart Disease Prediction
```
POST /api/predict/heart
Content-Type: application/json

{
    "age": 45,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
}
```

### Chest X-Ray Analysis
```
POST /api/analyze/chest-xray
Content-Type: application/json

{
    "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
    "success": true,
    "analysis": {
        "primary_finding": "Normal",
        "confidence": 85.2,
        "risk_level": "Low",
        "findings": [...],
        "detailed_explanation": "AI-generated explanation",
        "recommendation": "...",
        "ai_enhanced": true,
        "models": {
            "image_analysis": "MONAI DenseNet121",
            "explanation": "llama-3.3-70b-versatile"
        }
    }
}
```

### Skin Lesion Analysis
```
POST /api/analyze/skin-lesion
Content-Type: application/json

{
    "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
    "success": true,
    "analysis": {
        "primary_classification": "Melanocytic Nevi",
        "confidence": 78.5,
        "risk_level": "Low",
        "urgency": "Routine",
        "findings": [...],
        "abcde_assessment": {...},
        "detailed_explanation": "...",
        "recommendation": "...",
        "warning_signs": [...]
    }
}
```

### Eye Health / Diabetic Retinopathy
```
POST /api/analyze/eye-health
Content-Type: application/json

{
    "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
    "success": true,
    "analysis": {
        "diabetic_retinopathy": {
            "grade": "No DR",
            "grade_number": 0,
            "confidence": 82.3,
            "description": "..."
        },
        "risk_level": "Low",
        "urgency": "Annual",
        "other_findings": [...],
        "recommendation": "...",
        "follow_up": "Annual screening recommended",
        "lifestyle_tips": [...]
    }
}
```

### AI Chatbot
```
POST /api/chat
Content-Type: application/json

{
    "message": "What are symptoms of diabetes?",
    "conversation_history": []
}
```

### Quick Health Assessment
```
POST /api/assess/quick
Content-Type: application/json

{
    "heart_rate": 72,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "blood_sugar": 100,
    "oxygen_level": 98
}
```

## AI Architecture

### MONAI + Groq Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                  Image Input (Base64)                    │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 MONAI Model Inference                    │
│                                                          │
│  chest_xray_service.py  → DenseNet121                   │
│  skin_lesion_service.py → EfficientNetBN                │
│  eye_health_service.py  → DenseNet121                   │
│                                                          │
│  Output: Class probabilities                             │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              ai_vision_service.py                        │
│                                                          │
│  Groq LLM (llama-3.3-70b-versatile)                    │
│  - enhance_chest_xray_result()                          │
│  - enhance_skin_lesion_result()                         │
│  - enhance_eye_health_result()                          │
│                                                          │
│  Output: Detailed explanations, recommendations          │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Enhanced Response                       │
│  - MONAI predictions + Groq explanations                │
│  - Risk assessment and urgency                          │
│  - Actionable recommendations                           │
│  - Warning signs and follow-up schedule                 │
└─────────────────────────────────────────────────────────┘
```

## Input Parameters

### Heart Disease Prediction

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| age | int | Age in years | 29-77 |
| sex | int | Gender (1=male, 0=female) | 0-1 |
| cp | int | Chest pain type | 0-3 |
| trestbps | int | Resting blood pressure (mm Hg) | 94-200 |
| chol | int | Serum cholesterol (mg/dl) | 126-564 |
| fbs | int | Fasting blood sugar >120 mg/dl | 0-1 |
| restecg | int | Resting ECG results | 0-2 |
| thalach | int | Max heart rate achieved | 71-202 |
| exang | int | Exercise induced angina | 0-1 |
| oldpeak | float | ST depression | 0-6.2 |
| slope | int | Slope of peak exercise ST | 0-2 |
| ca | int | Number of major vessels | 0-4 |
| thal | int | Thalassemia | 0-3 |

### Image Analysis

| Parameter | Type | Description |
|-----------|------|-------------|
| image | string | Base64 encoded image (JPEG/PNG) |

## Model Details

### Chest X-Ray Classes
- Normal
- Pneumonia
- COVID-19
- Cardiomegaly
- Lung Nodule
- Pleural Effusion
- Atelectasis
- Pneumothorax

### Skin Lesion Classes (HAM10000)
- Melanocytic Nevi (Benign)
- Melanoma (Malignant)
- Benign Keratosis
- Basal Cell Carcinoma (Malignant)
- Actinic Keratosis (Pre-cancerous)
- Vascular Lesion (Benign)
- Dermatofibroma (Benign)

### Diabetic Retinopathy Grades
- No DR (Grade 0)
- Mild DR (Grade 1)
- Moderate DR (Grade 2)
- Severe DR (Grade 3)
- Proliferative DR (Grade 4)

## Intel Optimization

For best performance on Intel i7 Ultra + Iris GPU:

```python
# IPEX optimization (automatic)
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)

# OpenVINO for Intel GPU
import openvino as ov
# Model converted to IR format for Iris GPU acceleration
```

## Project Structure

```
backend/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── README.md                       # This file
├── services/
│   ├── __init__.py
│   ├── heart_disease_service.py    # Heart prediction
│   ├── chest_xray_service.py       # Chest X-ray analysis (MONAI)
│   ├── skin_lesion_service.py      # Skin lesion analysis (MONAI)
│   ├── eye_health_service.py       # Eye health analysis (MONAI)
│   ├── ai_vision_service.py        # Groq LLM enhancement
│   └── chat_service.py             # AI chatbot
├── models/
│   └── __init__.py
├── utils/
│   └── __init__.py
└── data/                           # Model weights and data files
```

## Testing

```bash
# Test API manually
curl -X POST http://localhost:5000/api/health

# Test heart prediction
curl -X POST http://localhost:5000/api/predict/heart \
  -H "Content-Type: application/json" \
  -d '{"age":45,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.5,"slope":1,"ca":0,"thal":2}'

# Test image analysis (replace with actual base64)
curl -X POST http://localhost:5000/api/analyze/chest-xray \
  -H "Content-Type: application/json" \
  -d '{"image":"base64_encoded_image_here"}'
```

## License

Part of MediPlus hackathon project.
