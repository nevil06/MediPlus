# MediPlus

A comprehensive digital health assistant mobile application built with React Native and Expo.

## Overview

MediPlus leverages machine learning to provide early heart disease detection and quick health insights. The app features a smart medicine reminder system with camera-based medicine scanning for tracking timing and dosage.

## Features

### Heart Disease Detection
- ML-powered early detection of heart conditions
- Real-time health risk assessment
- Personalized health insights and recommendations

### Smart Medicine Reminder
- Camera-based medicine scanning (OCR)
- Automated dosage tracking
- Customizable reminder notifications
- Medicine schedule management

### Health Dashboard
- Comprehensive health metrics visualization
- Historical health data tracking
- Progress monitoring and trends

## Tech Stack

- **Framework**: React Native with Expo
- **Language**: TypeScript
- **ML**: TensorFlow.js / TensorFlow Lite
- **Camera**: Expo Camera, Expo Barcode Scanner
- **Notifications**: Expo Notifications
- **Storage**: Expo Secure Store

## Getting Started

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn
- Expo CLI
- Android/iOS device or emulator

### Installation

```bash
# Clone the repository
git clone https://github.com/nevil06/MediPlus.git

# Navigate to project directory
cd MediPlus

# Install dependencies
npm install

# Start the development server
npm start
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

## Project Structure

```
MediPlus/
├── src/
│   ├── screens/
│   │   ├── HeartHealth/      # ML prediction UI
│   │   ├── MedicineScanner/  # Camera + OCR
│   │   ├── Reminders/        # Notification scheduling
│   │   └── Dashboard/        # Health insights
│   ├── services/
│   │   ├── ml/               # TensorFlow models
│   │   ├── notifications/    # Reminder logic
│   │   └── api/              # Backend calls
│   └── components/
├── assets/
│   └── models/               # ML model files
└── App.tsx
```

## Team

- **Harsha N** - harsha210108@gmail.com
- **Naren V** - narenbhaskar2007@gmail.com
- **Manas Kiran Habbu** - manaskiranhabbu@gmail.com
- **Mithun Gowda B** - mithungowda.b7411@gmail.com

## Future Roadmap

- [ ] Advanced health analytics
- [ ] Integration with wearable devices
- [ ] Telemedicine consultation features
- [ ] Multi-language support
- [ ] Cloud sync for health records

## License

This project is developed as part of a hackathon.

---

*Built with care for better health outcomes*
