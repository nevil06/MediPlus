// MediPlus Configuration
// Environment variables should be set in .env file
// For Expo, use app.config.js extra field or expo-env

import Constants from 'expo-constants';

// Get environment variables from Expo config extra or process.env
const getEnvVariable = (key: string, defaultValue: string = ''): string => {
  // Try Expo Constants first (from app.config.js extra)
  const expoExtra = Constants.expoConfig?.extra;
  if (expoExtra && expoExtra[key]) {
    return expoExtra[key];
  }

  // Fallback to process.env (for web/development)
  if (typeof process !== 'undefined' && process.env && process.env[key]) {
    return process.env[key] as string;
  }

  return defaultValue;
};

export const API_CONFIG = {
  // Backend URL - set in .env or app.config.js
  BACKEND_URL: getEnvVariable('BACKEND_URL', 'http://localhost:5000'),

  // Groq API Key - MUST be set in environment, no default exposed
  GROQ_API_KEY: getEnvVariable('GROQ_API_KEY', ''),
};

// Validate configuration
export const validateConfig = (): { valid: boolean; errors: string[] } => {
  const errors: string[] = [];

  if (!API_CONFIG.GROQ_API_KEY) {
    errors.push('GROQ_API_KEY is not configured. Chat features will not work.');
  }

  if (!API_CONFIG.BACKEND_URL) {
    errors.push('BACKEND_URL is not configured.');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
};

// MediBot System Prompt
export const MEDIBOT_SYSTEM_PROMPT = `You are MediBot, a friendly and knowledgeable medical health assistant for the MediPlus app.

Your role:
- Provide helpful, accurate health information and guidance
- Answer questions about symptoms, medications, and general health topics
- Offer wellness tips and preventive care advice
- Help users understand their health metrics and test results
- Guide users to appropriate features in the MediPlus app

Important guidelines:
- Always be empathetic, patient, and supportive
- Use simple, easy-to-understand language
- Never diagnose conditions - always recommend consulting healthcare professionals for diagnosis
- For emergencies, immediately advise calling emergency services
- Remind users that you provide general information, not medical advice
- Be concise but thorough in your responses
- Ask clarifying questions when needed

When discussing heart health:
- Explain risk factors clearly (age, blood pressure, cholesterol, etc.)
- Encourage users to use the Heart Disease Prediction feature
- Promote healthy lifestyle choices

Start conversations warmly and be helpful throughout the chat.`;
