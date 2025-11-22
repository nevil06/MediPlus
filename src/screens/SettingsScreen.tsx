import React from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Linking,
  Image,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES, COMPONENT_SIZES } from '../constants/theme';
import Header from '../components/Header';
import MONAIInfoCard from '../components/MONAIInfoCard';
import { scale } from '../utils/responsive';

interface SettingItemProps {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  subtitle?: string;
  onPress?: () => void;
}

const SettingItem = ({ icon, title, subtitle, onPress }: SettingItemProps) => (
  <TouchableOpacity style={styles.settingItem} onPress={onPress} disabled={!onPress}>
    <View style={styles.settingIcon}>
      <Ionicons name={icon} size={scale(20)} color={COLORS.primary} />
    </View>
    <View style={styles.settingContent}>
      <Text style={styles.settingTitle}>{title}</Text>
      {subtitle && <Text style={styles.settingSubtitle}>{subtitle}</Text>}
    </View>
    {onPress && <Ionicons name="chevron-forward" size={scale(18)} color={COLORS.textLight} />}
  </TouchableOpacity>
);

export default function SettingsScreen() {
  const TAB_BAR_SPACE = COMPONENT_SIZES.tabBarHeight + COMPONENT_SIZES.tabBarBottom;

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <Header />
      <ScrollView
        contentContainerStyle={[styles.scrollContent, { paddingBottom: TAB_BAR_SPACE + scale(20) }]}
        showsVerticalScrollIndicator={false}
      >
        {/* App Info */}
        <View style={styles.appInfo}>
          <View style={styles.appIcon}>
            <Image
              source={require('../../assets/logo.png')}
              style={styles.appLogoImage}
              resizeMode="contain"
            />
          </View>
          <Text style={styles.appName}>MediPlus</Text>
          <Text style={styles.appVersion}>Version 1.0.0</Text>
        </View>

        {/* MONAI Status */}
        <Text style={styles.sectionTitle}>AI Engine Status</Text>
        <MONAIInfoCard />

        {/* About Section */}
        <Text style={styles.sectionTitle}>About</Text>
        <View style={styles.section}>
          <SettingItem
            icon="information-circle"
            title="About MediPlus"
            subtitle="AI-powered health assistant"
          />
          <SettingItem
            icon="heart"
            title="Heart Disease Prediction"
            subtitle="Powered by MONAI/PyTorch"
          />
          <SettingItem
            icon="chatbubbles"
            title="MediBot AI"
            subtitle="Powered by Groq LLaMA 3.3"
          />
        </View>

        {/* Team Section */}
        <Text style={styles.sectionTitle}>Team</Text>
        <View style={styles.section}>
          <SettingItem icon="star" title="Nevil D'Souza" subtitle="Team Leader & Developer" />
          <SettingItem icon="people" title="Harsha N" subtitle="Developer" />
          <SettingItem icon="people" title="Naren V" subtitle="Developer" />
          <SettingItem icon="people" title="Manas Kiran Habbu" subtitle="Developer" />
          <SettingItem icon="people" title="Mithun Gowda B" subtitle="Developer" />
        </View>

        {/* Resources Section */}
        <Text style={styles.sectionTitle}>Resources</Text>
        <View style={styles.section}>
          <SettingItem
            icon="logo-github"
            title="GitHub Repository"
            subtitle="View source code"
            onPress={() => Linking.openURL('https://github.com/nevil06/MediPlus')}
          />
          <SettingItem
            icon="document-text"
            title="Privacy Policy"
            subtitle="How we handle your data"
            onPress={() => Alert.alert('Privacy', 'Your health data stays on your device and is not stored on our servers.')}
          />
        </View>

        {/* Disclaimer */}
        <View style={styles.disclaimer}>
          <Ionicons name="warning" size={scale(18)} color={COLORS.warning} />
          <Text style={styles.disclaimerText}>
            MediPlus is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
          </Text>
        </View>
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
  appInfo: {
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radiusLg,
    padding: scale(20),
    marginBottom: scale(20),
    marginTop: scale(14),
  },
  appIcon: {
    width: COMPONENT_SIZES.avatarXl,
    height: COMPONENT_SIZES.avatarXl,
    borderRadius: scale(20),
    backgroundColor: COLORS.primaryLight + '30',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: scale(10),
    overflow: 'hidden',
  },
  appLogoImage: {
    width: COMPONENT_SIZES.avatarXl - scale(16),
    height: COMPONENT_SIZES.avatarXl - scale(16),
    borderRadius: scale(12),
  },
  appName: {
    fontSize: SIZES.xxl,
    fontWeight: 'bold',
    color: COLORS.primary,
  },
  appVersion: {
    fontSize: SIZES.sm,
    color: COLORS.textSecondary,
    marginTop: scale(4),
  },
  sectionTitle: {
    fontSize: SIZES.md,
    fontWeight: '600',
    color: COLORS.textSecondary,
    marginBottom: scale(10),
    marginTop: scale(6),
  },
  section: {
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    overflow: 'hidden',
    marginBottom: scale(14),
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: scale(14),
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  settingIcon: {
    width: scale(34),
    height: scale(34),
    borderRadius: scale(8),
    backgroundColor: COLORS.primaryLight + '20',
    justifyContent: 'center',
    alignItems: 'center',
  },
  settingContent: {
    flex: 1,
    marginLeft: scale(10),
  },
  settingTitle: {
    fontSize: SIZES.sm,
    fontWeight: '500',
    color: COLORS.text,
  },
  settingSubtitle: {
    fontSize: SIZES.xs,
    color: COLORS.textSecondary,
    marginTop: scale(2),
  },
  disclaimer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: COLORS.warning + '15',
    borderRadius: SIZES.radius,
    padding: scale(14),
    marginTop: scale(6),
    gap: scale(10),
  },
  disclaimerText: {
    flex: 1,
    fontSize: SIZES.xs,
    color: COLORS.text,
    lineHeight: SIZES.xs * 1.6,
  },
});
