// MediPlus Expo Configuration
// This file loads environment variables for the app

import 'dotenv/config';

export default {
  expo: {
    name: 'MediPlus',
    slug: 'MediPlus',
    version: '1.0.0',
    orientation: 'portrait',
    icon: './assets/logo.png',
    userInterfaceStyle: 'light',
    newArchEnabled: true,
    splash: {
      image: './assets/logo.png',
      resizeMode: 'contain',
      backgroundColor: '#ffffff',
    },
    ios: {
      supportsTablet: true,
      bundleIdentifier: 'com.care.mediplus',
    },
    android: {
      package: 'com.care.mediplus',
      adaptiveIcon: {
        foregroundImage: './assets/logo.png',
        backgroundColor: '#ffffff',
      },
      edgeToEdgeEnabled: true,
      predictiveBackGestureEnabled: false,
    },
    web: {
      favicon: './assets/logo.png',
    },
    // Environment variables accessible via Constants.expoConfig.extra
    extra: {
      GROQ_API_KEY: process.env.GROQ_API_KEY || '',
      BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:5000',
    },
  },
};
