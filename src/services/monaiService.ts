/**
 * MONAI Service
 * Handles API calls for enhanced MONAI-based medical image analysis
 * Includes segmentation, enhanced analysis, and MONAI capabilities info
 */

import { API_CONFIG } from '../constants/config';

// ============== Types ==============

export type SegmentationTask =
  | 'lung_2d'
  | 'cardiac_2d'
  | 'organ_3d'
  | 'liver_tumor'
  | 'brain_tumor';

export type SegmentationArchitecture =
  | 'unet'
  | 'basic_unet'
  | 'attention_unet'
  | 'segresnet'
  | 'unetr'
  | 'swin_unetr'
  | 'vnet';

export interface SegmentationOptions {
  task: SegmentationTask;
  architecture?: SegmentationArchitecture;
  returnOverlay?: boolean;
  returnMasks?: boolean;
  enhanceWithAI?: boolean;
}

export interface SegmentationStatistics {
  [className: string]: {
    pixel_count: number;
    percentage: number;
    present: boolean;
    volume_mm3?: number;
  };
}

export interface SegmentationResult {
  success: boolean;
  segmentation?: number[][];
  classes?: string[];
  task?: string;
  architecture?: string;
  overlay?: string; // Base64 encoded colored overlay
  masks?: Record<string, string>; // Class name to base64 mask
  statistics?: SegmentationStatistics;
  ai_analysis?: {
    interpretation: string;
    findings: string[];
    recommendations: string[];
  };
  error?: string;
}

export interface EnhancedAnalysisResult {
  success: boolean;
  primary_diagnosis?: {
    condition: string;
    probability: number;
    confidence: string;
  };
  predictions?: Record<string, {
    probability: number;
    confidence?: string;
    is_malignant?: boolean;
  }>;
  all_findings?: Array<{
    condition: string;
    probability: number;
    confidence: string;
  }>;
  model_info?: {
    architecture: string;
    framework: string;
    transforms: string;
    test_time_augmentation: boolean;
  };
  ai_enhanced?: {
    interpretation: string;
    findings: string[];
    recommendations: string[];
  };
  // Eye-specific fields
  dr_grade?: {
    grade: string;
    severity_index: number;
    probability: number;
    all_grades: Record<string, number>;
  };
  recommendations?: string[];
  // Skin-specific fields
  abcde_assessment?: {
    asymmetry: { score: number; description: string };
    border: { score: number; description: string };
    color: { score: number; description: string };
    diameter: { score: number; description: string };
    evolution: { score: number; description: string };
    note: string;
  };
  error?: string;
}

export interface SegmentationTaskInfo {
  name: string;
  spatial_dims: number;
  in_channels: number;
  out_channels: number;
  roi_size: number[];
  classes: string[];
}

export interface MONAIInfo {
  monai_available: boolean;
  torch_available: boolean;
  cuda_available: boolean;
  cuda_device?: string;
  monai_version?: string;
  transforms: string[];
  networks: string[];
  losses: string[];
  metrics: string[];
}

export interface SegmentationTasksResponse {
  success: boolean;
  tasks?: Record<string, SegmentationTaskInfo>;
  architectures?: string[];
  error?: string;
}

// ============== Service ==============

class MONAIService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_CONFIG.BACKEND_URL;
  }

  /**
   * Get MONAI capabilities information
   */
  async getMonaiInfo(): Promise<MONAIInfo> {
    try {
      const response = await fetch(`${this.baseUrl}/api/monai/info`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      // Check if response is OK and content-type is JSON
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error('Server returned non-JSON response');
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('MONAI info error:', error);
      return {
        monai_available: false,
        torch_available: false,
        cuda_available: false,
        transforms: [],
        networks: [],
        losses: [],
        metrics: [],
      };
    }
  }

  /**
   * Get available segmentation tasks and architectures
   */
  async getSegmentationTasks(): Promise<SegmentationTasksResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/segment/tasks`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Segmentation tasks error:', error);
      return {
        success: false,
        error: 'Failed to fetch segmentation tasks',
      };
    }
  }

  /**
   * Perform medical image segmentation
   */
  async segment(
    imageBase64: string,
    options: SegmentationOptions
  ): Promise<SegmentationResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/segment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageBase64,
          task: options.task,
          architecture: options.architecture || 'unet',
          return_overlay: options.returnOverlay ?? true,
          return_masks: options.returnMasks ?? true,
          enhance_with_ai: options.enhanceWithAI ?? true,
        }),
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Segmentation error:', error);
      return {
        success: false,
        error: 'Failed to perform segmentation. Please try again.',
      };
    }
  }

  /**
   * Enhanced chest X-ray analysis with full MONAI capabilities
   */
  async analyzeChestXRayEnhanced(
    imageBase64: string,
    enhanceWithAI: boolean = true
  ): Promise<EnhancedAnalysisResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/analyze/chest-xray`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageBase64,
          enhance_with_ai: enhanceWithAI,
        }),
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Enhanced chest X-ray analysis error:', error);
      return {
        success: false,
        error: 'Failed to analyze chest X-ray. Please try again.',
      };
    }
  }

  /**
   * Enhanced skin lesion analysis with full MONAI capabilities
   */
  async analyzeSkinEnhanced(
    imageBase64: string,
    enhanceWithAI: boolean = true
  ): Promise<EnhancedAnalysisResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/analyze/skin`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageBase64,
          enhance_with_ai: enhanceWithAI,
        }),
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Enhanced skin analysis error:', error);
      return {
        success: false,
        error: 'Failed to analyze skin lesion. Please try again.',
      };
    }
  }

  /**
   * Enhanced eye/retinal analysis with full MONAI capabilities
   */
  async analyzeEyeEnhanced(
    imageBase64: string,
    enhanceWithAI: boolean = true
  ): Promise<EnhancedAnalysisResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/analyze/eye`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageBase64,
          enhance_with_ai: enhanceWithAI,
        }),
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Enhanced eye analysis error:', error);
      return {
        success: false,
        error: 'Failed to analyze eye image. Please try again.',
      };
    }
  }
}

export const monaiService = new MONAIService();
