# MediPlus Backend

Flask-based API server for heart disease prediction using MONAI/PyTorch pretrained models.
Optimized for Intel i7 Ultra + Iris GPU.

## Features

- **Heart Disease Prediction**: ML-powered risk assessment using pretrained MONAI models
- **Quick Health Assessment**: Real-time vital signs analysis
- **Medicine Information**: Drug database lookup
- **Intel Optimized**: IPEX and OpenVINO support for Intel hardware

## Tech Stack

- **Framework**: Flask + Flask-CORS
- **ML**: PyTorch + MONAI (pretrained models)
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

### 3. Run Server

```bash
# Development
python app.py

# Production
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Server runs at `http://localhost:5000`

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

**Response:**
```json
{
    "success": true,
    "prediction": {
        "risk_score": 35.2,
        "risk_level": "Moderate",
        "risk_factors": ["Elevated blood pressure", "High cholesterol"],
        "recommendation": "Discuss findings with your primary care physician.",
        "model_type": "clinical_statistical",
        "confidence": 0.85
    }
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

### Medicine Information
```
POST /api/medicine/info
Content-Type: application/json

{
    "medicine_name": "aspirin"
}
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

## Model Architecture

The prediction service uses an ensemble approach:

1. **Clinical Risk Score** (40%): Based on Framingham Risk Score and ACC/AHA guidelines
2. **Statistical Analysis** (30%): Feature correlations from UCI Heart Disease dataset
3. **Neural Network** (30%): Pretrained MONAI DenseNet121 with custom classification head

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

## Environment Variables

```bash
HOST=0.0.0.0
PORT=5000
DEBUG=True
```

## Testing

```bash
# Run tests
pytest tests/

# Test API manually
curl -X POST http://localhost:5000/api/predict/heart \
  -H "Content-Type: application/json" \
  -d '{"age":45,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.5,"slope":1,"ca":0,"thal":2}'
```

## Project Structure

```
backend/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── services/
│   ├── __init__.py
│   └── heart_disease_service.py  # Heart prediction service
├── models/
│   └── __init__.py
├── utils/
│   └── __init__.py
└── data/                  # Model weights and data files
```

## License

Part of MediPlus hackathon project.
