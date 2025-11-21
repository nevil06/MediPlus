import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { COLORS, SIZES, SHADOWS, COMPONENT_SIZES } from '../constants/theme';
import { scale, moderateScale, wp, hp } from '../utils/responsive';
import Header from '../components/Header';
import {
  imageAnalysisService,
  AnalysisType,
  ChestXRayResult,
  SkinLesionResult,
  EyeHealthResult,
} from '../services/imageAnalysisService';

type AnalysisOption = {
  id: AnalysisType;
  title: string;
  description: string;
  icon: keyof typeof Ionicons.glyphMap;
  color: string;
};

const analysisOptions: AnalysisOption[] = [
  {
    id: 'chest-xray',
    title: 'Chest X-Ray',
    description: 'Detect pneumonia, COVID-19, cardiomegaly',
    icon: 'body-outline',
    color: '#4A90D9',
  },
  {
    id: 'skin',
    title: 'Skin Lesion',
    description: 'Screen for melanoma and skin conditions',
    icon: 'scan-outline',
    color: '#E67E22',
  },
  {
    id: 'eye',
    title: 'Eye Health',
    description: 'Diabetic retinopathy screening',
    icon: 'eye-outline',
    color: '#27AE60',
  },
];

export default function ImageAnalysisScreen() {
  const [selectedType, setSelectedType] = useState<AnalysisType | null>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<ChestXRayResult | SkinLesionResult | EyeHealthResult | null>(null);

  const requestPermissions = async () => {
    const { status: cameraStatus } = await ImagePicker.requestCameraPermissionsAsync();
    const { status: mediaStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (cameraStatus !== 'granted' || mediaStatus !== 'granted') {
      Alert.alert(
        'Permissions Required',
        'Please grant camera and photo library permissions to use this feature.'
      );
      return false;
    }
    return true;
  };

  const pickImage = async (useCamera: boolean) => {
    const hasPermission = await requestPermissions();
    if (!hasPermission) return;

    const options: ImagePicker.ImagePickerOptions = {
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.8,
      base64: true,
    };

    try {
      const result = useCamera
        ? await ImagePicker.launchCameraAsync(options)
        : await ImagePicker.launchImageLibraryAsync(options);

      if (!result.canceled && result.assets[0]) {
        setImageUri(result.assets[0].uri);
        setImageBase64(result.assets[0].base64 || null);
        setResult(null);
      }
    } catch (error) {
      console.error('Image picker error:', error);
      Alert.alert('Error', 'Failed to pick image. Please try again.');
    }
  };

  const analyzeImage = async () => {
    if (!selectedType || !imageBase64) {
      Alert.alert('Error', 'Please select an analysis type and upload an image.');
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      const analysisResult = await imageAnalysisService.analyze(selectedType, imageBase64);
      setResult(analysisResult as ChestXRayResult | SkinLesionResult | EyeHealthResult);
    } catch (error) {
      console.error('Analysis error:', error);
      Alert.alert('Error', 'Failed to analyze image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setImageUri(null);
    setImageBase64(null);
    setResult(null);
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel?.toLowerCase()) {
      case 'high':
      case 'critical':
      case 'urgent':
        return COLORS.error;
      case 'moderate':
      case 'soon':
        return COLORS.warning;
      case 'low':
      case 'routine':
      case 'annual':
        return COLORS.success;
      default:
        return COLORS.textSecondary;
    }
  };

  const renderAnalysisOptions = () => (
    <View style={styles.optionsContainer}>
      <Text style={styles.sectionTitle}>Select Analysis Type</Text>
      {analysisOptions.map((option) => (
        <TouchableOpacity
          key={option.id}
          style={[
            styles.optionCard,
            selectedType === option.id && styles.optionCardSelected,
            { borderLeftColor: option.color },
          ]}
          onPress={() => {
            setSelectedType(option.id);
            setResult(null);
          }}
        >
          <View style={[styles.optionIcon, { backgroundColor: option.color + '20' }]}>
            <Ionicons name={option.icon} size={scale(24)} color={option.color} />
          </View>
          <View style={styles.optionInfo}>
            <Text style={styles.optionTitle}>{option.title}</Text>
            <Text style={styles.optionDescription}>{option.description}</Text>
          </View>
          {selectedType === option.id && (
            <Ionicons name="checkmark-circle" size={scale(24)} color={COLORS.primary} />
          )}
        </TouchableOpacity>
      ))}
    </View>
  );

  const renderImagePicker = () => (
    <View style={styles.imagePickerContainer}>
      <Text style={styles.sectionTitle}>Upload Image</Text>

      {imageUri ? (
        <View style={styles.imagePreviewContainer}>
          <Image source={{ uri: imageUri }} style={styles.imagePreview} />
          <TouchableOpacity style={styles.removeImageBtn} onPress={resetAnalysis}>
            <Ionicons name="close-circle" size={scale(28)} color={COLORS.error} />
          </TouchableOpacity>
        </View>
      ) : (
        <View style={styles.uploadOptions}>
          <TouchableOpacity
            style={styles.uploadBtn}
            onPress={() => pickImage(true)}
          >
            <Ionicons name="camera" size={scale(32)} color={COLORS.primary} />
            <Text style={styles.uploadBtnText}>Take Photo</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.uploadBtn}
            onPress={() => pickImage(false)}
          >
            <Ionicons name="images" size={scale(32)} color={COLORS.primary} />
            <Text style={styles.uploadBtnText}>Gallery</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );

  const renderChestXRayResult = (data: ChestXRayResult) => {
    if (!data.analysis) return null;
    const { analysis } = data;

    return (
      <View style={styles.resultContainer}>
        <View style={styles.resultHeader}>
          <Text style={styles.resultTitle}>Chest X-Ray Analysis</Text>
          <View style={[styles.riskBadge, { backgroundColor: getRiskColor(analysis.risk_level) + '20' }]}>
            <Text style={[styles.riskText, { color: getRiskColor(analysis.risk_level) }]}>
              {analysis.risk_level} Risk
            </Text>
          </View>
        </View>

        <View style={styles.primaryFinding}>
          <Text style={styles.findingLabel}>Primary Finding</Text>
          <Text style={styles.findingValue}>{analysis.primary_finding}</Text>
          <Text style={styles.confidenceText}>Confidence: {analysis.confidence}%</Text>
        </View>

        {analysis.findings.length > 0 && (
          <View style={styles.findingsSection}>
            <Text style={styles.findingsSectionTitle}>Detected Conditions</Text>
            {analysis.findings.map((finding, index) => (
              <View key={index} style={styles.findingItem}>
                <View style={styles.findingHeader}>
                  <Text style={styles.findingCondition}>{finding.condition}</Text>
                  <Text style={styles.findingProb}>{finding.probability}%</Text>
                </View>
                <Text style={styles.findingDesc}>{finding.description}</Text>
              </View>
            ))}
          </View>
        )}

        <View style={styles.recommendationBox}>
          <Ionicons name="information-circle" size={scale(20)} color={COLORS.primary} />
          <Text style={styles.recommendationText}>{analysis.recommendation}</Text>
        </View>

        <Text style={styles.disclaimer}>{analysis.disclaimer}</Text>
      </View>
    );
  };

  const renderSkinResult = (data: SkinLesionResult) => {
    if (!data.analysis) return null;
    const { analysis } = data;

    return (
      <View style={styles.resultContainer}>
        <View style={styles.resultHeader}>
          <Text style={styles.resultTitle}>Skin Lesion Analysis</Text>
          <View style={[styles.riskBadge, { backgroundColor: getRiskColor(analysis.risk_level) + '20' }]}>
            <Text style={[styles.riskText, { color: getRiskColor(analysis.risk_level) }]}>
              {analysis.urgency}
            </Text>
          </View>
        </View>

        <View style={styles.primaryFinding}>
          <Text style={styles.findingLabel}>Classification</Text>
          <Text style={styles.findingValue}>{analysis.primary_classification}</Text>
          <Text style={styles.confidenceText}>Confidence: {analysis.confidence}%</Text>
        </View>

        <View style={styles.abcdeSection}>
          <Text style={styles.findingsSectionTitle}>ABCDE Assessment</Text>
          <View style={styles.abcdeGrid}>
            <Text style={styles.abcdeItem}>A - {analysis.abcde_assessment.asymmetry}</Text>
            <Text style={styles.abcdeItem}>B - {analysis.abcde_assessment.border}</Text>
            <Text style={styles.abcdeItem}>C - {analysis.abcde_assessment.color}</Text>
            <Text style={styles.abcdeItem}>D - {analysis.abcde_assessment.diameter}</Text>
            <Text style={styles.abcdeItem}>E - {analysis.abcde_assessment.evolution}</Text>
          </View>
        </View>

        {analysis.next_steps && analysis.next_steps.length > 0 && (
          <View style={styles.nextStepsSection}>
            <Text style={styles.findingsSectionTitle}>Next Steps</Text>
            {analysis.next_steps.map((step, index) => (
              <View key={index} style={styles.stepItem}>
                <Ionicons name="checkmark-circle-outline" size={scale(16)} color={COLORS.primary} />
                <Text style={styles.stepText}>{step}</Text>
              </View>
            ))}
          </View>
        )}

        <View style={styles.recommendationBox}>
          <Ionicons name="information-circle" size={scale(20)} color={COLORS.primary} />
          <Text style={styles.recommendationText}>{analysis.recommendation}</Text>
        </View>

        <Text style={styles.disclaimer}>{analysis.disclaimer}</Text>
      </View>
    );
  };

  const renderEyeResult = (data: EyeHealthResult) => {
    if (!data.analysis) return null;
    const { analysis } = data;

    return (
      <View style={styles.resultContainer}>
        <View style={styles.resultHeader}>
          <Text style={styles.resultTitle}>Eye Health Analysis</Text>
          <View style={[styles.riskBadge, { backgroundColor: getRiskColor(analysis.urgency) + '20' }]}>
            <Text style={[styles.riskText, { color: getRiskColor(analysis.urgency) }]}>
              {analysis.urgency}
            </Text>
          </View>
        </View>

        <View style={styles.healthScoreContainer}>
          <Text style={styles.healthScoreLabel}>Eye Health Score</Text>
          <Text style={[
            styles.healthScoreValue,
            { color: analysis.overall_health_score >= 70 ? COLORS.success :
                     analysis.overall_health_score >= 40 ? COLORS.warning : COLORS.error }
          ]}>
            {analysis.overall_health_score}/100
          </Text>
        </View>

        <View style={styles.primaryFinding}>
          <Text style={styles.findingLabel}>Diabetic Retinopathy</Text>
          <Text style={styles.findingValue}>{analysis.diabetic_retinopathy.grade}</Text>
          <Text style={styles.confidenceText}>
            Confidence: {analysis.diabetic_retinopathy.confidence}%
          </Text>
          <Text style={styles.drDescription}>{analysis.diabetic_retinopathy.description}</Text>
        </View>

        {analysis.other_findings && analysis.other_findings.length > 0 && (
          <View style={styles.findingsSection}>
            <Text style={styles.findingsSectionTitle}>Other Findings</Text>
            {analysis.other_findings.map((finding, index) => (
              <View key={index} style={styles.findingItem}>
                <View style={styles.findingHeader}>
                  <Text style={styles.findingCondition}>{finding.condition}</Text>
                  <Text style={styles.findingProb}>{finding.probability}%</Text>
                </View>
                <Text style={styles.findingDesc}>{finding.description}</Text>
              </View>
            ))}
          </View>
        )}

        <View style={styles.followUpBox}>
          <Ionicons name="calendar-outline" size={scale(20)} color={COLORS.primary} />
          <Text style={styles.followUpText}>{analysis.follow_up}</Text>
        </View>

        {analysis.lifestyle_tips && analysis.lifestyle_tips.length > 0 && (
          <View style={styles.tipsSection}>
            <Text style={styles.findingsSectionTitle}>Lifestyle Tips</Text>
            {analysis.lifestyle_tips.slice(0, 4).map((tip, index) => (
              <View key={index} style={styles.tipItem}>
                <Ionicons name="leaf-outline" size={scale(14)} color={COLORS.success} />
                <Text style={styles.tipText}>{tip}</Text>
              </View>
            ))}
          </View>
        )}

        <View style={styles.recommendationBox}>
          <Ionicons name="information-circle" size={scale(20)} color={COLORS.primary} />
          <Text style={styles.recommendationText}>{analysis.recommendation}</Text>
        </View>

        <Text style={styles.disclaimer}>{analysis.disclaimer}</Text>
      </View>
    );
  };

  const renderResult = () => {
    if (!result || !result.success) return null;

    switch (selectedType) {
      case 'chest-xray':
        return renderChestXRayResult(result as ChestXRayResult);
      case 'skin':
        return renderSkinResult(result as SkinLesionResult);
      case 'eye':
        return renderEyeResult(result as EyeHealthResult);
      default:
        return null;
    }
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <Header />
      <ScrollView
        style={styles.content}
        contentContainerStyle={styles.contentContainer}
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.pageTitle}>Medical Image Analysis</Text>
        <Text style={styles.pageSubtitle}>
          AI-powered analysis using MONAI medical imaging models
        </Text>

        {renderAnalysisOptions()}

        {selectedType && renderImagePicker()}

        {selectedType && imageUri && !result && (
          <TouchableOpacity
            style={[styles.analyzeBtn, isAnalyzing && styles.analyzeBtnDisabled]}
            onPress={analyzeImage}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? (
              <>
                <ActivityIndicator color={COLORS.surface} size="small" />
                <Text style={styles.analyzeBtnText}>Analyzing...</Text>
              </>
            ) : (
              <>
                <Ionicons name="analytics" size={scale(20)} color={COLORS.surface} />
                <Text style={styles.analyzeBtnText}>Analyze Image</Text>
              </>
            )}
          </TouchableOpacity>
        )}

        {result && !result.success && (
          <View style={styles.errorBox}>
            <Ionicons name="alert-circle" size={scale(24)} color={COLORS.error} />
            <Text style={styles.errorText}>{result.error || 'Analysis failed. Please try again.'}</Text>
          </View>
        )}

        {renderResult()}

        <View style={styles.bottomPadding} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: scale(16),
  },
  pageTitle: {
    fontSize: moderateScale(22),
    fontWeight: '700',
    color: COLORS.text,
    marginBottom: scale(4),
  },
  pageSubtitle: {
    fontSize: moderateScale(13),
    color: COLORS.textSecondary,
    marginBottom: scale(20),
  },
  sectionTitle: {
    fontSize: moderateScale(16),
    fontWeight: '600',
    color: COLORS.text,
    marginBottom: scale(12),
  },
  optionsContainer: {
    marginBottom: scale(20),
  },
  optionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderRadius: scale(12),
    padding: scale(14),
    marginBottom: scale(10),
    borderLeftWidth: 4,
    ...SHADOWS.small,
  },
  optionCardSelected: {
    backgroundColor: COLORS.primary + '10',
    borderColor: COLORS.primary,
  },
  optionIcon: {
    width: scale(44),
    height: scale(44),
    borderRadius: scale(22),
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: scale(12),
  },
  optionInfo: {
    flex: 1,
  },
  optionTitle: {
    fontSize: moderateScale(15),
    fontWeight: '600',
    color: COLORS.text,
  },
  optionDescription: {
    fontSize: moderateScale(12),
    color: COLORS.textSecondary,
    marginTop: scale(2),
  },
  imagePickerContainer: {
    marginBottom: scale(20),
  },
  uploadOptions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  uploadBtn: {
    width: wp(40),
    backgroundColor: COLORS.surface,
    borderRadius: scale(12),
    padding: scale(20),
    alignItems: 'center',
    borderWidth: 2,
    borderColor: COLORS.primary + '30',
    borderStyle: 'dashed',
  },
  uploadBtnText: {
    fontSize: moderateScale(14),
    color: COLORS.primary,
    marginTop: scale(8),
    fontWeight: '500',
  },
  imagePreviewContainer: {
    position: 'relative',
    alignItems: 'center',
  },
  imagePreview: {
    width: wp(85),
    height: hp(30),
    borderRadius: scale(12),
    backgroundColor: COLORS.border,
  },
  removeImageBtn: {
    position: 'absolute',
    top: scale(8),
    right: scale(20),
    backgroundColor: COLORS.surface,
    borderRadius: scale(14),
  },
  analyzeBtn: {
    flexDirection: 'row',
    backgroundColor: COLORS.primary,
    borderRadius: scale(12),
    padding: scale(16),
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: scale(20),
    ...SHADOWS.medium,
  },
  analyzeBtnDisabled: {
    opacity: 0.7,
  },
  analyzeBtnText: {
    fontSize: moderateScale(16),
    fontWeight: '600',
    color: COLORS.surface,
    marginLeft: scale(8),
  },
  resultContainer: {
    backgroundColor: COLORS.surface,
    borderRadius: scale(16),
    padding: scale(16),
    marginBottom: scale(20),
    ...SHADOWS.medium,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: scale(16),
  },
  resultTitle: {
    fontSize: moderateScale(18),
    fontWeight: '700',
    color: COLORS.text,
  },
  riskBadge: {
    paddingHorizontal: scale(12),
    paddingVertical: scale(4),
    borderRadius: scale(12),
  },
  riskText: {
    fontSize: moderateScale(12),
    fontWeight: '600',
  },
  primaryFinding: {
    backgroundColor: COLORS.background,
    borderRadius: scale(12),
    padding: scale(14),
    marginBottom: scale(16),
  },
  findingLabel: {
    fontSize: moderateScale(12),
    color: COLORS.textSecondary,
    marginBottom: scale(4),
  },
  findingValue: {
    fontSize: moderateScale(20),
    fontWeight: '700',
    color: COLORS.text,
  },
  confidenceText: {
    fontSize: moderateScale(12),
    color: COLORS.primary,
    marginTop: scale(4),
  },
  drDescription: {
    fontSize: moderateScale(12),
    color: COLORS.textSecondary,
    marginTop: scale(8),
    lineHeight: scale(18),
  },
  healthScoreContainer: {
    alignItems: 'center',
    marginBottom: scale(16),
  },
  healthScoreLabel: {
    fontSize: moderateScale(12),
    color: COLORS.textSecondary,
  },
  healthScoreValue: {
    fontSize: moderateScale(36),
    fontWeight: '700',
  },
  findingsSection: {
    marginBottom: scale(16),
  },
  findingsSectionTitle: {
    fontSize: moderateScale(14),
    fontWeight: '600',
    color: COLORS.text,
    marginBottom: scale(10),
  },
  findingItem: {
    backgroundColor: COLORS.background,
    borderRadius: scale(8),
    padding: scale(12),
    marginBottom: scale(8),
  },
  findingHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: scale(4),
  },
  findingCondition: {
    fontSize: moderateScale(14),
    fontWeight: '600',
    color: COLORS.text,
  },
  findingProb: {
    fontSize: moderateScale(13),
    fontWeight: '600',
    color: COLORS.primary,
  },
  findingDesc: {
    fontSize: moderateScale(12),
    color: COLORS.textSecondary,
  },
  abcdeSection: {
    marginBottom: scale(16),
  },
  abcdeGrid: {
    backgroundColor: COLORS.background,
    borderRadius: scale(8),
    padding: scale(12),
  },
  abcdeItem: {
    fontSize: moderateScale(12),
    color: COLORS.text,
    marginBottom: scale(6),
  },
  nextStepsSection: {
    marginBottom: scale(16),
  },
  stepItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: scale(8),
  },
  stepText: {
    flex: 1,
    fontSize: moderateScale(13),
    color: COLORS.text,
    marginLeft: scale(8),
  },
  followUpBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.primary + '10',
    borderRadius: scale(8),
    padding: scale(12),
    marginBottom: scale(16),
  },
  followUpText: {
    flex: 1,
    fontSize: moderateScale(13),
    color: COLORS.primary,
    marginLeft: scale(8),
    fontWeight: '500',
  },
  tipsSection: {
    marginBottom: scale(16),
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: scale(6),
  },
  tipText: {
    flex: 1,
    fontSize: moderateScale(12),
    color: COLORS.text,
    marginLeft: scale(6),
  },
  recommendationBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: COLORS.primary + '10',
    borderRadius: scale(8),
    padding: scale(12),
    marginBottom: scale(12),
  },
  recommendationText: {
    flex: 1,
    fontSize: moderateScale(13),
    color: COLORS.text,
    marginLeft: scale(8),
    lineHeight: scale(20),
  },
  disclaimer: {
    fontSize: moderateScale(11),
    color: COLORS.textSecondary,
    fontStyle: 'italic',
    textAlign: 'center',
  },
  errorBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.error + '10',
    borderRadius: scale(12),
    padding: scale(16),
    marginBottom: scale(20),
  },
  errorText: {
    flex: 1,
    fontSize: moderateScale(14),
    color: COLORS.error,
    marginLeft: scale(10),
  },
  bottomPadding: {
    height: COMPONENT_SIZES.tabBarHeight + scale(20),
  },
});
