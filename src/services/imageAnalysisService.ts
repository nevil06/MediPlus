/**
 * Image Analysis Service
 * Handles API calls for medical image analysis (Chest X-Ray, Skin, Eye)
 */

import { API_CONFIG } from '../constants/config';

export type AnalysisType = 'chest-xray' | 'skin' | 'eye';

export interface ChestXRayResult {
  success: boolean;
  analysis?: {
    primary_finding: string;
    confidence: number;
    risk_level: string;
    findings: Array<{
      condition: string;
      probability: number;
      description: string;
    }>;
    all_predictions: Record<string, number>;
    recommendation: string;
    disclaimer: string;
  };
  error?: string;
}

export interface SkinLesionResult {
  success: boolean;
  analysis?: {
    primary_classification: string;
    confidence: number;
    risk_level: string;
    urgency: string;
    findings: Array<{
      classification: string;
      probability: number;
      risk_category: string;
      description: string;
    }>;
    abcde_assessment: {
      asymmetry: string;
      border: string;
      color: string;
      diameter: string;
      evolution: string;
      overall_concern: string;
    };
    all_predictions: Record<string, number>;
    recommendation: string;
    next_steps: string[];
    disclaimer: string;
  };
  error?: string;
}

export interface EyeHealthResult {
  success: boolean;
  analysis?: {
    diabetic_retinopathy: {
      grade: string;
      grade_number: number;
      confidence: number;
      description: string;
      all_grades: Record<string, number>;
    };
    risk_level: string;
    urgency: string;
    other_findings: Array<{
      condition: string;
      probability: number;
      description: string;
    }>;
    overall_health_score: number;
    recommendation: string;
    follow_up: string;
    lifestyle_tips: string[];
    disclaimer: string;
  };
  error?: string;
}

export type AnalysisResult = ChestXRayResult | SkinLesionResult | EyeHealthResult;

class ImageAnalysisService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_CONFIG.BACKEND_URL;
  }

  /**
   * Analyze a chest X-ray image
   */
  async analyzeChestXRay(imageBase64: string): Promise<ChestXRayResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/analyze/chest-xray`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageBase64 }),
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Chest X-Ray analysis error:', error);
      return {
        success: false,
        error: 'Failed to analyze chest X-ray. Please try again.',
      };
    }
  }

  /**
   * Analyze a skin lesion image
   */
  async analyzeSkinLesion(imageBase64: string): Promise<SkinLesionResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/analyze/skin`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageBase64 }),
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Skin lesion analysis error:', error);
      return {
        success: false,
        error: 'Failed to analyze skin lesion. Please try again.',
      };
    }
  }

  /**
   * Analyze an eye/retinal image
   */
  async analyzeEyeHealth(imageBase64: string): Promise<EyeHealthResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/analyze/eye`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageBase64 }),
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Eye health analysis error:', error);
      return {
        success: false,
        error: 'Failed to analyze eye image. Please try again.',
      };
    }
  }

  /**
   * Generic analyze method that routes to specific analysis
   */
  async analyze(type: AnalysisType, imageBase64: string): Promise<AnalysisResult> {
    switch (type) {
      case 'chest-xray':
        return this.analyzeChestXRay(imageBase64);
      case 'skin':
        return this.analyzeSkinLesion(imageBase64);
      case 'eye':
        return this.analyzeEyeHealth(imageBase64);
      default:
        return {
          success: false,
          error: 'Invalid analysis type',
        };
    }
  }
}

export const imageAnalysisService = new ImageAnalysisService();
