// Heart Disease Prediction Service
import { API_CONFIG } from '../constants/config';

export interface HeartPredictionInput {
  age: number;
  sex: number; // 1=male, 0=female
  cp: number; // chest pain type (0-3)
  trestbps: number; // resting blood pressure
  chol: number; // cholesterol mg/dl
  fbs: number; // fasting blood sugar > 120 mg/dl
  restecg: number; // resting ECG results (0-2)
  thalach: number; // max heart rate achieved
  exang: number; // exercise induced angina
  oldpeak: number; // ST depression
  slope: number; // slope of peak exercise ST (0-2)
  ca: number; // number of major vessels (0-3)
  thal: number; // thalassemia (0-3)
}

export interface HeartPredictionResult {
  risk_score: number;
  risk_level: string;
  risk_factors: string[];
  recommendation: string;
  model_type: string;
  confidence: number;
  details: {
    clinical_score: number;
    statistical_score: number;
    nn_score: number | null;
    data_quality: string;
  };
}

export interface PredictionResponse {
  success: boolean;
  prediction?: HeartPredictionResult;
  error?: string;
}

class HeartService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_CONFIG.BACKEND_URL;
  }

  setBaseUrl(url: string) {
    this.baseUrl = url;
  }

  async predictHeartDisease(input: HeartPredictionInput): Promise<PredictionResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/predict/heart`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(input),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      return data;
    } catch (error) {
      console.error('Heart Prediction Error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to connect to server',
      };
    }
  }

  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`);
      const data = await response.json();
      return data.status === 'healthy';
    } catch {
      return false;
    }
  }
}

export const heartService = new HeartService();
