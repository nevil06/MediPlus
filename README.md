# MediPlus

A comprehensive digital health assistant mobile application built with React Native and Expo, powered by MONAI medical imaging AI and Groq LLM.

## Overview

MediPlus leverages advanced machine learning to provide early disease detection, medical image analysis, and intelligent health insights. The app combines MONAI (Medical Open Network for AI) for medical image analysis with Groq's LLM for detailed, patient-friendly explanations.

## Features

### Medical Image Analysis (MONAI + Groq AI)

#### Chest X-Ray Analysis
- Detects: Pneumonia, COVID-19, Cardiomegaly, Lung Nodules, Pleural Effusion
- MONAI DenseNet121 for image classification
- Groq LLM provides detailed findings and recommendations

#### Skin Lesion Analysis
- Classifies: Melanoma, Basal Cell Carcinoma, Benign Keratosis, and more
- ABCDE melanoma criteria assessment
- Risk-based urgency recommendations

#### Eye Health / Diabetic Retinopathy
- Grades DR from No DR to Proliferative DR
- Detects: Glaucoma, Macular Degeneration, Cataracts
- Follow-up scheduling based on severity

### Heart Disease Prediction
- ML-powered early detection of heart conditions
- 13 clinical parameters for risk assessment
- Real-time health risk score and recommendations

### MediChatBot
- AI-powered health assistant using Groq
- Natural language health queries
- Contextual medical information

### Smart Medicine Reminder
- Camera-based medicine scanning (OCR)
- Automated dosage tracking
- Customizable reminder notifications

### Health Dashboard
- Comprehensive health metrics visualization
- Historical health data tracking
- Progress monitoring and trends

## Tech Stack

### Frontend
- **Framework**: React Native with Expo
- **Language**: TypeScript
- **Navigation**: React Navigation
- **Camera**: expo-image-picker
- **Storage**: AsyncStorage

### Backend
- **Framework**: Flask + Flask-CORS
- **ML Framework**: PyTorch + MONAI
- **LLM**: Groq API (llama-3.3-70b-versatile)
- **Optimization**: Intel Extension for PyTorch (IPEX)

## Getting Started

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn
- Python 3.9+
- Expo CLI
- Android/iOS device or emulator

### Installation

```bash
# Clone the repository
git clone https://github.com/nevil06/MediPlus.git

# Navigate to project directory
cd MediPlus

# Copy environment file
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Install frontend dependencies
npm install

# Start the development server
npm start
```

### Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run server
python app.py
```

### Running the App

```bash
# Android
npm run android

# iOS
npm run ios

# Web
npm run web
```

## Environment Variables

### Frontend (.env)
```bash
GROQ_API_KEY=your_groq_api_key_here
BACKEND_URL=http://localhost:5000
```

### Backend (backend/.env)
```bash
HOST=0.0.0.0
PORT=5000
DEBUG=True
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key at: https://console.groq.com/keys

## Project Structure

```
MediPlus/
├── src/
│   ├── screens/
│   │   ├── HomeScreen.tsx
│   │   ├── HeartHealthScreen.tsx
│   │   ├── ImageAnalysisScreen.tsx    # Medical image AI
│   │   ├── ChatBotScreen.tsx          # AI chatbot
│   │   └── SettingsScreen.tsx
│   ├── services/
│   │   ├── apiService.ts
│   │   ├── imageAnalysisService.ts    # Image analysis API
│   │   └── chatService.ts
│   ├── components/
│   │   └── Header.tsx
│   └── navigation/
│       └── TabNavigator.tsx
├── backend/
│   ├── app.py                         # Flask API
│   ├── services/
│   │   ├── heart_disease_service.py
│   │   ├── chest_xray_service.py      # MONAI + Groq
│   │   ├── skin_lesion_service.py     # MONAI + Groq
│   │   ├── eye_health_service.py      # MONAI + Groq
│   │   └── ai_vision_service.py       # Groq enhancement
│   └── requirements.txt
├── assets/
│   └── logo.png
├── .env.example
└── App.tsx
```

## API Endpoints

### Medical Image Analysis
```
POST /api/analyze/chest-xray     # Chest X-Ray analysis
POST /api/analyze/skin-lesion    # Skin lesion classification
POST /api/analyze/eye-health     # Diabetic retinopathy screening
```

### Heart Disease
```
POST /api/predict/heart          # Heart disease risk prediction
POST /api/assess/quick           # Quick health assessment
```

### AI Chat
```
POST /api/chat                   # AI chatbot responses
```

## AI Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Image Upload                          │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              MONAI Model Inference                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ DenseNet121 │  │EfficientNet │  │ DenseNet121 │     │
│  │ Chest X-Ray │  │Skin Lesion  │  │ Eye Health  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────┬───────────────────────────────┘
                          │ Predictions
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Groq LLM Enhancement                        │
│         (llama-3.3-70b-versatile)                       │
│  - Detailed explanations                                │
│  - Risk assessment                                      │
│  - Actionable recommendations                           │
│  - Warning signs to watch                               │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Enhanced Results                            │
│  - AI predictions + Human-readable insights             │
│  - Risk level and urgency                               │
│  - Next steps and follow-up schedule                    │
└─────────────────────────────────────────────────────────┘
```

## Team

- **Nevil D'Souza** - Team Leader & Developer
- **Harsha N** - Developer (harsha210108@gmail.com)
- **Naren V** - Developer (narenbhaskar2007@gmail.com)
- **Manas Kiran Habbu** - Developer (manaskiranhabbu@gmail.com)
- **Mithun Gowda B** - Developer (mithungowda.b7411@gmail.com)

## Future Roadmap

- [x] Medical image analysis (Chest X-Ray, Skin, Eye)
- [x] MONAI + Groq AI integration
- [x] AI-powered chatbot
- [ ] Advanced health analytics
- [ ] Integration with wearable devices
- [ ] Telemedicine consultation features
- [ ] Multi-language support
- [ ] Cloud sync for health records

## License

This project is developed as part of a hackathon.

---

*Built with care for better health outcomes*
