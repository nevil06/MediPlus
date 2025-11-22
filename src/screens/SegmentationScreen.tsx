import React, { useState, useEffect } from 'react';
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
  monaiService,
  SegmentationTask,
  SegmentationArchitecture,
  SegmentationResult,
  SegmentationTaskInfo,
} from '../services/monaiService';

type TaskOption = {
  id: SegmentationTask;
  title: string;
  description: string;
  icon: keyof typeof Ionicons.glyphMap;
  color: string;
  dimensions: string;
};

const taskOptions: TaskOption[] = [
  {
    id: 'lung_2d',
    title: 'Lung Segmentation',
    description: 'Segment left and right lungs from chest X-ray',
    icon: 'body-outline',
    color: '#4A90D9',
    dimensions: '2D',
  },
  {
    id: 'cardiac_2d',
    title: 'Cardiac Segmentation',
    description: 'Segment heart chambers and myocardium',
    icon: 'heart-outline',
    color: '#E74C3C',
    dimensions: '2D',
  },
  {
    id: 'organ_3d',
    title: 'Multi-Organ (3D)',
    description: 'Segment 13 abdominal organs from CT',
    icon: 'fitness-outline',
    color: '#27AE60',
    dimensions: '3D',
  },
  {
    id: 'liver_tumor',
    title: 'Liver & Tumor',
    description: 'Segment liver and detect tumors',
    icon: 'medical-outline',
    color: '#9B59B6',
    dimensions: '3D',
  },
];

const architectureOptions: { id: SegmentationArchitecture; name: string; description: string }[] = [
  { id: 'unet', name: 'UNet', description: 'Standard, balanced' },
  { id: 'attention_unet', name: 'Attention UNet', description: 'Focus on relevant regions' },
  { id: 'segresnet', name: 'SegResNet', description: 'ResNet-based, high accuracy' },
  { id: 'unetr', name: 'UNETR', description: 'Transformer-based' },
  { id: 'swin_unetr', name: 'Swin UNETR', description: 'State-of-the-art' },
];

export default function SegmentationScreen() {
  const [selectedTask, setSelectedTask] = useState<SegmentationTask | null>(null);
  const [selectedArchitecture, setSelectedArchitecture] = useState<SegmentationArchitecture>('unet');
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [showArchitectureSelector, setShowArchitectureSelector] = useState(false);

  const TAB_BAR_SPACE = COMPONENT_SIZES.tabBarHeight + COMPONENT_SIZES.tabBarBottom;

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
      aspect: [1, 1],
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

  const performSegmentation = async () => {
    if (!selectedTask || !imageBase64) {
      Alert.alert('Error', 'Please select a segmentation task and upload an image.');
      return;
    }

    setIsProcessing(true);
    setResult(null);

    try {
      const segResult = await monaiService.segment(imageBase64, {
        task: selectedTask,
        architecture: selectedArchitecture,
        returnOverlay: true,
        returnMasks: true,
        enhanceWithAI: true,
      });
      setResult(segResult);
    } catch (error) {
      console.error('Segmentation error:', error);
      Alert.alert('Error', 'Failed to perform segmentation. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const resetSegmentation = () => {
    setImageUri(null);
    setImageBase64(null);
    setResult(null);
  };

  const getSelectedTaskInfo = () => {
    return taskOptions.find(t => t.id === selectedTask);
  };

  const renderTaskSelector = () => (
    <View style={styles.sectionContainer}>
      <Text style={styles.sectionTitle}>Select Segmentation Task</Text>
      <View style={styles.taskGrid}>
        {taskOptions.map((task) => (
          <TouchableOpacity
            key={task.id}
            style={[
              styles.taskCard,
              selectedTask === task.id && styles.taskCardSelected,
              { borderColor: selectedTask === task.id ? task.color : COLORS.border },
            ]}
            onPress={() => setSelectedTask(task.id)}
          >
            <View style={[styles.taskIconContainer, { backgroundColor: task.color + '20' }]}>
              <Ionicons name={task.icon} size={scale(24)} color={task.color} />
            </View>
            <Text style={styles.taskTitle}>{task.title}</Text>
            <Text style={styles.taskDescription}>{task.description}</Text>
            <View style={[styles.dimensionBadge, { backgroundColor: task.color + '30' }]}>
              <Text style={[styles.dimensionText, { color: task.color }]}>{task.dimensions}</Text>
            </View>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );

  const renderArchitectureSelector = () => (
    <View style={styles.sectionContainer}>
      <TouchableOpacity
        style={styles.architectureHeader}
        onPress={() => setShowArchitectureSelector(!showArchitectureSelector)}
      >
        <View>
          <Text style={styles.sectionTitle}>Model Architecture</Text>
          <Text style={styles.selectedArchitecture}>
            {architectureOptions.find(a => a.id === selectedArchitecture)?.name || 'UNet'}
          </Text>
        </View>
        <Ionicons
          name={showArchitectureSelector ? 'chevron-up' : 'chevron-down'}
          size={scale(20)}
          color={COLORS.textSecondary}
        />
      </TouchableOpacity>

      {showArchitectureSelector && (
        <View style={styles.architectureList}>
          {architectureOptions.map((arch) => (
            <TouchableOpacity
              key={arch.id}
              style={[
                styles.architectureOption,
                selectedArchitecture === arch.id && styles.architectureOptionSelected,
              ]}
              onPress={() => {
                setSelectedArchitecture(arch.id);
                setShowArchitectureSelector(false);
              }}
            >
              <View style={styles.architectureInfo}>
                <Text style={styles.architectureName}>{arch.name}</Text>
                <Text style={styles.architectureDesc}>{arch.description}</Text>
              </View>
              {selectedArchitecture === arch.id && (
                <Ionicons name="checkmark-circle" size={scale(20)} color={COLORS.primary} />
              )}
            </TouchableOpacity>
          ))}
        </View>
      )}
    </View>
  );

  const renderImageUpload = () => (
    <View style={styles.sectionContainer}>
      <Text style={styles.sectionTitle}>Upload Image</Text>

      {imageUri ? (
        <View style={styles.imagePreviewContainer}>
          <Image source={{ uri: imageUri }} style={styles.imagePreview} />
          <TouchableOpacity style={styles.removeImageButton} onPress={resetSegmentation}>
            <Ionicons name="close-circle" size={scale(28)} color={COLORS.error} />
          </TouchableOpacity>
        </View>
      ) : (
        <View style={styles.uploadButtons}>
          <TouchableOpacity
            style={styles.uploadButton}
            onPress={() => pickImage(true)}
          >
            <Ionicons name="camera-outline" size={scale(28)} color={COLORS.primary} />
            <Text style={styles.uploadButtonText}>Camera</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.uploadButton}
            onPress={() => pickImage(false)}
          >
            <Ionicons name="images-outline" size={scale(28)} color={COLORS.primary} />
            <Text style={styles.uploadButtonText}>Gallery</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );

  const renderResults = () => {
    if (!result) return null;

    if (!result.success) {
      return (
        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle-outline" size={scale(48)} color={COLORS.error} />
          <Text style={styles.errorText}>{result.error || 'Segmentation failed'}</Text>
        </View>
      );
    }

    const taskInfo = getSelectedTaskInfo();

    return (
      <View style={styles.resultsContainer}>
        <Text style={styles.resultsTitle}>Segmentation Results</Text>

        {/* Overlay Image */}
        {result.overlay && (
          <View style={styles.overlayContainer}>
            <Text style={styles.overlayTitle}>Segmented Image</Text>
            <Image
              source={{ uri: `data:image/png;base64,${result.overlay}` }}
              style={styles.overlayImage}
              resizeMode="contain"
            />
          </View>
        )}

        {/* Statistics */}
        {result.statistics && (
          <View style={styles.statisticsContainer}>
            <Text style={styles.statisticsTitle}>Segmentation Statistics</Text>
            {Object.entries(result.statistics).map(([className, stats]) => {
              if (className === 'Background') return null;
              return (
                <View key={className} style={styles.statRow}>
                  <View style={styles.statInfo}>
                    <Text style={styles.statClassName}>{className}</Text>
                    <Text style={styles.statValue}>
                      {stats.percentage.toFixed(1)}% of image
                    </Text>
                  </View>
                  <View style={styles.statBar}>
                    <View
                      style={[
                        styles.statBarFill,
                        { width: `${Math.min(stats.percentage * 2, 100)}%` },
                      ]}
                    />
                  </View>
                </View>
              );
            })}
          </View>
        )}

        {/* Model Info */}
        <View style={styles.modelInfoContainer}>
          <Text style={styles.modelInfoTitle}>Model Information</Text>
          <View style={styles.modelInfoRow}>
            <Text style={styles.modelInfoLabel}>Architecture:</Text>
            <Text style={styles.modelInfoValue}>{result.architecture || selectedArchitecture}</Text>
          </View>
          <View style={styles.modelInfoRow}>
            <Text style={styles.modelInfoLabel}>Task:</Text>
            <Text style={styles.modelInfoValue}>{taskInfo?.title || result.task}</Text>
          </View>
          <View style={styles.modelInfoRow}>
            <Text style={styles.modelInfoLabel}>Framework:</Text>
            <Text style={styles.modelInfoValue}>MONAI</Text>
          </View>
        </View>

        {/* AI Analysis */}
        {result.ai_analysis && (
          <View style={styles.aiAnalysisContainer}>
            <Text style={styles.aiAnalysisTitle}>AI Analysis</Text>
            <Text style={styles.aiAnalysisText}>{result.ai_analysis.interpretation}</Text>
            {result.ai_analysis.findings && result.ai_analysis.findings.length > 0 && (
              <View style={styles.findingsList}>
                {result.ai_analysis.findings.map((finding, index) => (
                  <View key={index} style={styles.findingItem}>
                    <Ionicons name="checkmark-circle" size={scale(16)} color={COLORS.success} />
                    <Text style={styles.findingText}>{finding}</Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}

        {/* Disclaimer */}
        <View style={styles.disclaimerContainer}>
          <Ionicons name="information-circle-outline" size={scale(16)} color={COLORS.warning} />
          <Text style={styles.disclaimerText}>
            This segmentation is for educational and research purposes only.
            Always consult a medical professional for clinical decisions.
          </Text>
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <Header />
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[styles.scrollContent, { paddingBottom: TAB_BAR_SPACE + scale(20) }]}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.titleContainer}>
          <Text style={styles.screenTitle}>Medical Image Segmentation</Text>
          <Text style={styles.screenSubtitle}>
            AI-powered organ and structure segmentation using MONAI
          </Text>
        </View>

        {renderTaskSelector()}
        {selectedTask && renderArchitectureSelector()}
        {selectedTask && renderImageUpload()}

        {selectedTask && imageBase64 && !isProcessing && !result && (
          <TouchableOpacity style={styles.analyzeButton} onPress={performSegmentation}>
            <Ionicons name="scan-outline" size={scale(20)} color={COLORS.surface} />
            <Text style={styles.analyzeButtonText}>Run Segmentation</Text>
          </TouchableOpacity>
        )}

        {isProcessing && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={COLORS.primary} />
            <Text style={styles.loadingText}>Processing with MONAI...</Text>
            <Text style={styles.loadingSubtext}>This may take a few moments</Text>
          </View>
        )}

        {renderResults()}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: SIZES.padding,
  },
  titleContainer: {
    marginVertical: scale(16),
  },
  screenTitle: {
    fontSize: SIZES.xl,
    fontWeight: '700',
    color: COLORS.text,
    marginBottom: scale(4),
  },
  screenSubtitle: {
    fontSize: SIZES.sm,
    color: COLORS.textSecondary,
  },
  sectionContainer: {
    marginBottom: scale(20),
  },
  sectionTitle: {
    fontSize: SIZES.md,
    fontWeight: '600',
    color: COLORS.text,
    marginBottom: scale(12),
  },
  taskGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  taskCard: {
    width: '48%',
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    padding: scale(12),
    marginBottom: scale(12),
    borderWidth: 2,
    borderColor: COLORS.border,
    ...SHADOWS.small,
  },
  taskCardSelected: {
    borderWidth: 2,
  },
  taskIconContainer: {
    width: scale(48),
    height: scale(48),
    borderRadius: scale(24),
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: scale(8),
  },
  taskTitle: {
    fontSize: SIZES.md,
    fontWeight: '600',
    color: COLORS.text,
    marginBottom: scale(4),
  },
  taskDescription: {
    fontSize: SIZES.xs,
    color: COLORS.textSecondary,
    marginBottom: scale(8),
  },
  dimensionBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: scale(8),
    paddingVertical: scale(2),
    borderRadius: scale(4),
  },
  dimensionText: {
    fontSize: SIZES.xs,
    fontWeight: '600',
  },
  architectureHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    padding: scale(12),
    borderRadius: SIZES.radius,
    ...SHADOWS.small,
  },
  selectedArchitecture: {
    fontSize: SIZES.sm,
    color: COLORS.primary,
    fontWeight: '500',
  },
  architectureList: {
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    marginTop: scale(8),
    ...SHADOWS.small,
  },
  architectureOption: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: scale(12),
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  architectureOptionSelected: {
    backgroundColor: COLORS.primaryLight + '20',
  },
  architectureInfo: {
    flex: 1,
  },
  architectureName: {
    fontSize: SIZES.md,
    fontWeight: '500',
    color: COLORS.text,
  },
  architectureDesc: {
    fontSize: SIZES.xs,
    color: COLORS.textSecondary,
  },
  uploadButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  uploadButton: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.surface,
    width: wp(35),
    height: hp(12),
    borderRadius: SIZES.radius,
    borderWidth: 2,
    borderColor: COLORS.primary + '40',
    borderStyle: 'dashed',
    ...SHADOWS.small,
  },
  uploadButtonText: {
    marginTop: scale(8),
    fontSize: SIZES.sm,
    color: COLORS.primary,
    fontWeight: '500',
  },
  imagePreviewContainer: {
    position: 'relative',
    alignItems: 'center',
  },
  imagePreview: {
    width: wp(80),
    height: wp(80),
    borderRadius: SIZES.radius,
    backgroundColor: COLORS.surface,
  },
  removeImageButton: {
    position: 'absolute',
    top: -scale(10),
    right: wp(5),
    backgroundColor: COLORS.surface,
    borderRadius: scale(14),
  },
  analyzeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.primary,
    padding: scale(16),
    borderRadius: SIZES.radius,
    marginVertical: scale(16),
    ...SHADOWS.medium,
  },
  analyzeButtonText: {
    color: COLORS.surface,
    fontSize: SIZES.md,
    fontWeight: '600',
    marginLeft: scale(8),
  },
  loadingContainer: {
    alignItems: 'center',
    padding: scale(32),
  },
  loadingText: {
    marginTop: scale(16),
    fontSize: SIZES.md,
    color: COLORS.text,
    fontWeight: '500',
  },
  loadingSubtext: {
    marginTop: scale(4),
    fontSize: SIZES.sm,
    color: COLORS.textSecondary,
  },
  errorContainer: {
    alignItems: 'center',
    padding: scale(24),
    backgroundColor: COLORS.error + '10',
    borderRadius: SIZES.radius,
    marginTop: scale(16),
  },
  errorText: {
    marginTop: scale(8),
    fontSize: SIZES.md,
    color: COLORS.error,
    textAlign: 'center',
  },
  resultsContainer: {
    marginTop: scale(16),
  },
  resultsTitle: {
    fontSize: SIZES.lg,
    fontWeight: '700',
    color: COLORS.text,
    marginBottom: scale(16),
  },
  overlayContainer: {
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    padding: scale(12),
    marginBottom: scale(16),
    ...SHADOWS.small,
  },
  overlayTitle: {
    fontSize: SIZES.md,
    fontWeight: '600',
    color: COLORS.text,
    marginBottom: scale(8),
  },
  overlayImage: {
    width: '100%',
    height: wp(70),
    borderRadius: SIZES.radius,
    backgroundColor: COLORS.background,
  },
  statisticsContainer: {
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    padding: scale(12),
    marginBottom: scale(16),
    ...SHADOWS.small,
  },
  statisticsTitle: {
    fontSize: SIZES.md,
    fontWeight: '600',
    color: COLORS.text,
    marginBottom: scale(12),
  },
  statRow: {
    marginBottom: scale(12),
  },
  statInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: scale(4),
  },
  statClassName: {
    fontSize: SIZES.sm,
    fontWeight: '500',
    color: COLORS.text,
  },
  statValue: {
    fontSize: SIZES.sm,
    color: COLORS.textSecondary,
  },
  statBar: {
    height: scale(8),
    backgroundColor: COLORS.border,
    borderRadius: scale(4),
    overflow: 'hidden',
  },
  statBarFill: {
    height: '100%',
    backgroundColor: COLORS.primary,
    borderRadius: scale(4),
  },
  modelInfoContainer: {
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    padding: scale(12),
    marginBottom: scale(16),
    ...SHADOWS.small,
  },
  modelInfoTitle: {
    fontSize: SIZES.md,
    fontWeight: '600',
    color: COLORS.text,
    marginBottom: scale(8),
  },
  modelInfoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: scale(4),
  },
  modelInfoLabel: {
    fontSize: SIZES.sm,
    color: COLORS.textSecondary,
  },
  modelInfoValue: {
    fontSize: SIZES.sm,
    fontWeight: '500',
    color: COLORS.text,
  },
  aiAnalysisContainer: {
    backgroundColor: COLORS.primaryLight + '20',
    borderRadius: SIZES.radius,
    padding: scale(12),
    marginBottom: scale(16),
  },
  aiAnalysisTitle: {
    fontSize: SIZES.md,
    fontWeight: '600',
    color: COLORS.primary,
    marginBottom: scale(8),
  },
  aiAnalysisText: {
    fontSize: SIZES.sm,
    color: COLORS.text,
    lineHeight: SIZES.sm * 1.5,
  },
  findingsList: {
    marginTop: scale(8),
  },
  findingItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: scale(4),
  },
  findingText: {
    fontSize: SIZES.sm,
    color: COLORS.text,
    marginLeft: scale(8),
    flex: 1,
  },
  disclaimerContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: COLORS.warning + '15',
    padding: scale(12),
    borderRadius: SIZES.radius,
    marginBottom: scale(16),
  },
  disclaimerText: {
    flex: 1,
    marginLeft: scale(8),
    fontSize: SIZES.xs,
    color: COLORS.text,
    lineHeight: SIZES.xs * 1.5,
  },
});
