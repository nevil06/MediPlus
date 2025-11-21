import React, { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES, COMPONENT_SIZES } from '../constants/theme';
import { heartService, HeartPredictionInput, HeartPredictionResult } from '../services/heartService';
import Header from '../components/Header';
import { scale, wp } from '../utils/responsive';

interface FormField {
  key: keyof HeartPredictionInput;
  label: string;
  placeholder: string;
  type: 'number' | 'select';
  options?: { label: string; value: number }[];
  min?: number;
  max?: number;
}

const formFields: FormField[] = [
  { key: 'age', label: 'Age', placeholder: 'Enter age (29-77)', type: 'number', min: 29, max: 77 },
  {
    key: 'sex',
    label: 'Sex',
    placeholder: 'Select',
    type: 'select',
    options: [
      { label: 'Female', value: 0 },
      { label: 'Male', value: 1 },
    ],
  },
  {
    key: 'cp',
    label: 'Chest Pain Type',
    placeholder: 'Select',
    type: 'select',
    options: [
      { label: 'Typical Angina', value: 0 },
      { label: 'Atypical Angina', value: 1 },
      { label: 'Non-anginal Pain', value: 2 },
      { label: 'Asymptomatic', value: 3 },
    ],
  },
  { key: 'trestbps', label: 'Resting Blood Pressure (mm Hg)', placeholder: '94-200', type: 'number', min: 94, max: 200 },
  { key: 'chol', label: 'Cholesterol (mg/dl)', placeholder: '126-564', type: 'number', min: 126, max: 564 },
  {
    key: 'fbs',
    label: 'Fasting Blood Sugar > 120 mg/dl',
    placeholder: 'Select',
    type: 'select',
    options: [
      { label: 'No', value: 0 },
      { label: 'Yes', value: 1 },
    ],
  },
  {
    key: 'restecg',
    label: 'Resting ECG',
    placeholder: 'Select',
    type: 'select',
    options: [
      { label: 'Normal', value: 0 },
      { label: 'ST-T Abnormality', value: 1 },
      { label: 'LV Hypertrophy', value: 2 },
    ],
  },
  { key: 'thalach', label: 'Max Heart Rate', placeholder: '71-202', type: 'number', min: 71, max: 202 },
  {
    key: 'exang',
    label: 'Exercise Induced Angina',
    placeholder: 'Select',
    type: 'select',
    options: [
      { label: 'No', value: 0 },
      { label: 'Yes', value: 1 },
    ],
  },
  { key: 'oldpeak', label: 'ST Depression', placeholder: '0-6.2', type: 'number', min: 0, max: 6.2 },
  {
    key: 'slope',
    label: 'ST Slope',
    placeholder: 'Select',
    type: 'select',
    options: [
      { label: 'Upsloping', value: 0 },
      { label: 'Flat', value: 1 },
      { label: 'Downsloping', value: 2 },
    ],
  },
  {
    key: 'ca',
    label: 'Major Vessels (Fluoroscopy)',
    placeholder: 'Select',
    type: 'select',
    options: [
      { label: '0', value: 0 },
      { label: '1', value: 1 },
      { label: '2', value: 2 },
      { label: '3', value: 3 },
    ],
  },
  {
    key: 'thal',
    label: 'Thalassemia',
    placeholder: 'Select',
    type: 'select',
    options: [
      { label: 'Normal', value: 0 },
      { label: 'Fixed Defect', value: 1 },
      { label: 'Normal (2)', value: 2 },
      { label: 'Reversible Defect', value: 3 },
    ],
  },
];

export default function HeartScreen() {
  const [formData, setFormData] = useState<Partial<HeartPredictionInput>>({});
  const [result, setResult] = useState<HeartPredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [expandedField, setExpandedField] = useState<string | null>(null);

  const TAB_BAR_SPACE = COMPONENT_SIZES.tabBarHeight + COMPONENT_SIZES.tabBarBottom;

  const updateField = (key: keyof HeartPredictionInput, value: number) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
    setExpandedField(null);
  };

  const handlePredict = async () => {
    const missingFields = formFields.filter((f) => formData[f.key] === undefined);
    if (missingFields.length > 0) {
      Alert.alert('Missing Fields', `Please fill in: ${missingFields.map((f) => f.label).join(', ')}`);
      return;
    }

    setIsLoading(true);
    const response = await heartService.predictHeartDisease(formData as HeartPredictionInput);
    setIsLoading(false);

    if (response.success && response.prediction) {
      setResult(response.prediction);
    } else {
      Alert.alert('Error', response.error || 'Failed to get prediction');
    }
  };

  const getRiskColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'high':
        return COLORS.riskHigh;
      case 'moderate-high':
        return COLORS.warning;
      case 'moderate':
        return COLORS.warning;
      case 'low-moderate':
        return COLORS.riskLow;
      case 'low':
        return COLORS.riskLow;
      default:
        return COLORS.textSecondary;
    }
  };

  const resetForm = () => {
    setFormData({});
    setResult(null);
  };

  if (result) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <Header />
        <ScrollView
          contentContainerStyle={[styles.resultContainer, { paddingBottom: TAB_BAR_SPACE + scale(20) }]}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.resultHeader}>
            <Ionicons
              name="heart"
              size={scale(50)}
              color={getRiskColor(result.risk_level)}
            />
            <Text style={styles.resultTitle}>Heart Health Assessment</Text>
          </View>

          <View style={[styles.scoreCard, { borderColor: getRiskColor(result.risk_level) }]}>
            <Text style={styles.scoreLabel}>Risk Score</Text>
            <Text style={[styles.scoreValue, { color: getRiskColor(result.risk_level) }]}>
              {result.risk_score}%
            </Text>
            <View style={[styles.riskBadge, { backgroundColor: getRiskColor(result.risk_level) }]}>
              <Text style={styles.riskBadgeText}>{result.risk_level} Risk</Text>
            </View>
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Recommendation</Text>
            <Text style={styles.recommendationText}>{result.recommendation}</Text>
          </View>

          {result.risk_factors.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Risk Factors Identified</Text>
              {result.risk_factors.map((factor, index) => (
                <View key={index} style={styles.factorItem}>
                  <Ionicons name="alert-circle" size={scale(14)} color={COLORS.warning} />
                  <Text style={styles.factorText}>{factor}</Text>
                </View>
              ))}
            </View>
          )}

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Analysis Details</Text>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Clinical Score</Text>
              <Text style={styles.detailValue}>{result.details.clinical_score}%</Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Statistical Score</Text>
              <Text style={styles.detailValue}>{result.details.statistical_score}%</Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Confidence</Text>
              <Text style={styles.detailValue}>{(result.confidence * 100).toFixed(0)}%</Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Data Quality</Text>
              <Text style={styles.detailValue}>{result.details.data_quality}</Text>
            </View>
          </View>

          <TouchableOpacity style={styles.resetButton} onPress={resetForm}>
            <Ionicons name="refresh" size={scale(18)} color={COLORS.surface} />
            <Text style={styles.resetButtonText}>New Assessment</Text>
          </TouchableOpacity>

          <Text style={styles.disclaimer}>
            This is not a medical diagnosis. Please consult a healthcare professional for proper evaluation.
          </Text>
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <Header />
      <ScrollView
        contentContainerStyle={[styles.scrollContent, { paddingBottom: TAB_BAR_SPACE + scale(20) }]}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.pageHeader}>
          <Ionicons name="heart" size={scale(28)} color={COLORS.accent} />
          <Text style={styles.pageTitle}>Heart Disease Risk</Text>
          <Text style={styles.pageSubtitle}>Enter your health metrics for assessment</Text>
        </View>

        {formFields.map((field) => (
          <View key={field.key} style={styles.fieldContainer}>
            <Text style={styles.fieldLabel}>{field.label}</Text>
            {field.type === 'number' ? (
              <TextInput
                style={styles.textInput}
                placeholder={field.placeholder}
                placeholderTextColor={COLORS.textLight}
                keyboardType="decimal-pad"
                value={formData[field.key]?.toString() || ''}
                onChangeText={(text) => {
                  const num = parseFloat(text);
                  if (!isNaN(num)) {
                    updateField(field.key, num);
                  } else if (text === '') {
                    setFormData((prev) => {
                      const newData = { ...prev };
                      delete newData[field.key];
                      return newData;
                    });
                  }
                }}
              />
            ) : (
              <TouchableOpacity
                style={styles.selectButton}
                onPress={() => setExpandedField(expandedField === field.key ? null : field.key)}
              >
                <Text
                  style={[
                    styles.selectButtonText,
                    formData[field.key] === undefined && styles.selectPlaceholder,
                  ]}
                >
                  {formData[field.key] !== undefined
                    ? field.options?.find((o) => o.value === formData[field.key])?.label
                    : field.placeholder}
                </Text>
                <Ionicons
                  name={expandedField === field.key ? 'chevron-up' : 'chevron-down'}
                  size={scale(18)}
                  color={COLORS.textSecondary}
                />
              </TouchableOpacity>
            )}
            {expandedField === field.key && field.options && (
              <View style={styles.optionsContainer}>
                {field.options.map((option) => (
                  <TouchableOpacity
                    key={option.value}
                    style={[
                      styles.optionItem,
                      formData[field.key] === option.value && styles.optionItemSelected,
                    ]}
                    onPress={() => updateField(field.key, option.value)}
                  >
                    <Text
                      style={[
                        styles.optionText,
                        formData[field.key] === option.value && styles.optionTextSelected,
                      ]}
                    >
                      {option.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            )}
          </View>
        ))}

        <TouchableOpacity
          style={[styles.predictButton, isLoading && styles.predictButtonDisabled]}
          onPress={handlePredict}
          disabled={isLoading}
        >
          {isLoading ? (
            <ActivityIndicator color={COLORS.surface} />
          ) : (
            <>
              <Ionicons name="analytics" size={scale(18)} color={COLORS.surface} />
              <Text style={styles.predictButtonText}>Analyze Risk</Text>
            </>
          )}
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  scrollContent: {
    padding: SIZES.padding,
  },
  pageHeader: {
    alignItems: 'center',
    marginBottom: scale(20),
    marginTop: scale(14),
  },
  pageTitle: {
    fontSize: SIZES.xxl,
    fontWeight: 'bold',
    color: COLORS.text,
    marginTop: scale(6),
  },
  pageSubtitle: {
    fontSize: SIZES.sm,
    color: COLORS.textSecondary,
    marginTop: scale(4),
  },
  fieldContainer: {
    marginBottom: scale(14),
  },
  fieldLabel: {
    fontSize: SIZES.sm,
    fontWeight: '600',
    color: COLORS.text,
    marginBottom: scale(6),
  },
  textInput: {
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    paddingHorizontal: scale(14),
    paddingVertical: scale(12),
    fontSize: SIZES.sm,
    color: COLORS.text,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  selectButton: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    paddingHorizontal: scale(14),
    paddingVertical: scale(12),
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  selectButtonText: {
    fontSize: SIZES.sm,
    color: COLORS.text,
  },
  selectPlaceholder: {
    color: COLORS.textLight,
  },
  optionsContainer: {
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    marginTop: scale(6),
    borderWidth: 1,
    borderColor: COLORS.border,
    overflow: 'hidden',
  },
  optionItem: {
    paddingHorizontal: scale(14),
    paddingVertical: scale(10),
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  optionItemSelected: {
    backgroundColor: COLORS.primaryLight + '20',
  },
  optionText: {
    fontSize: SIZES.sm,
    color: COLORS.text,
  },
  optionTextSelected: {
    color: COLORS.primary,
    fontWeight: '600',
  },
  predictButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.accent,
    borderRadius: SIZES.radius,
    paddingVertical: scale(14),
    marginTop: scale(20),
    gap: scale(8),
  },
  predictButtonDisabled: {
    opacity: 0.7,
  },
  predictButtonText: {
    fontSize: SIZES.md,
    fontWeight: 'bold',
    color: COLORS.surface,
  },
  resultContainer: {
    padding: SIZES.padding,
  },
  resultHeader: {
    alignItems: 'center',
    marginBottom: scale(20),
    marginTop: scale(14),
  },
  resultTitle: {
    fontSize: SIZES.xxl,
    fontWeight: 'bold',
    color: COLORS.text,
    marginTop: scale(10),
  },
  scoreCard: {
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radiusLg,
    padding: scale(20),
    alignItems: 'center',
    borderWidth: 2,
    marginBottom: scale(20),
  },
  scoreLabel: {
    fontSize: SIZES.sm,
    color: COLORS.textSecondary,
  },
  scoreValue: {
    fontSize: scale(40),
    fontWeight: 'bold',
    marginVertical: scale(6),
  },
  riskBadge: {
    paddingHorizontal: scale(14),
    paddingVertical: scale(6),
    borderRadius: scale(20),
  },
  riskBadgeText: {
    color: COLORS.surface,
    fontWeight: 'bold',
    fontSize: SIZES.sm,
  },
  section: {
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    padding: scale(14),
    marginBottom: scale(14),
  },
  sectionTitle: {
    fontSize: SIZES.md,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: scale(10),
  },
  recommendationText: {
    fontSize: SIZES.sm,
    color: COLORS.text,
    lineHeight: SIZES.sm * 1.5,
  },
  factorItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: scale(6),
    gap: scale(8),
  },
  factorText: {
    fontSize: SIZES.sm,
    color: COLORS.text,
    flex: 1,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: scale(8),
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  detailLabel: {
    fontSize: SIZES.sm,
    color: COLORS.textSecondary,
  },
  detailValue: {
    fontSize: SIZES.sm,
    fontWeight: '600',
    color: COLORS.text,
  },
  resetButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.primary,
    borderRadius: SIZES.radius,
    paddingVertical: scale(14),
    marginTop: scale(6),
    gap: scale(8),
  },
  resetButtonText: {
    fontSize: SIZES.md,
    fontWeight: 'bold',
    color: COLORS.surface,
  },
  disclaimer: {
    fontSize: SIZES.xs,
    color: COLORS.textLight,
    textAlign: 'center',
    marginTop: scale(14),
    fontStyle: 'italic',
  },
});
