import React from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES, COMPONENT_SIZES } from '../constants/theme';
import Header from '../components/Header';
import { scale, wp, hp } from '../utils/responsive';

interface FeatureCardProps {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  description: string;
  color: string;
  onPress: () => void;
}

const FeatureCard = ({ icon, title, description, color, onPress }: FeatureCardProps) => (
  <TouchableOpacity style={styles.featureCard} onPress={onPress}>
    <View style={[styles.featureIcon, { backgroundColor: color + '20' }]}>
      <Ionicons name={icon} size={scale(24)} color={color} />
    </View>
    <View style={styles.featureContent}>
      <Text style={styles.featureTitle}>{title}</Text>
      <Text style={styles.featureDescription}>{description}</Text>
    </View>
    <Ionicons name="chevron-forward" size={scale(18)} color={COLORS.textLight} />
  </TouchableOpacity>
);

export default function HomeScreen({ navigation }: any) {
  const TAB_BAR_SPACE = COMPONENT_SIZES.tabBarHeight + COMPONENT_SIZES.tabBarBottom;

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <Header />
      <ScrollView
        contentContainerStyle={[styles.scrollContent, { paddingBottom: TAB_BAR_SPACE + scale(20) }]}
        showsVerticalScrollIndicator={false}
      >
        {/* Hero Card */}
        <View style={styles.heroCard}>
          <View style={styles.heroContent}>
            <Text style={styles.heroTitle}>Your Digital{'\n'}Health Assistant</Text>
            <Text style={styles.heroSubtitle}>
              AI-powered health insights and heart disease risk assessment
            </Text>
            <TouchableOpacity
              style={styles.heroButton}
              onPress={() => navigation.navigate('Chat')}
            >
              <Text style={styles.heroButtonText}>Chat with MediBot</Text>
              <Ionicons name="chatbubbles" size={scale(16)} color={COLORS.surface} />
            </TouchableOpacity>
          </View>
          <View style={styles.heroIcon}>
            <Ionicons name="fitness" size={scale(70)} color={COLORS.surface} />
          </View>
        </View>

        {/* Quick Stats */}
        <View style={styles.statsContainer}>
          <View style={styles.statCard}>
            <Ionicons name="heart" size={scale(22)} color={COLORS.accent} />
            <Text style={styles.statValue}>Heart</Text>
            <Text style={styles.statLabel}>ML Model</Text>
          </View>
          <View style={styles.statCard}>
            <Ionicons name="scan" size={scale(22)} color="#9B59B6" />
            <Text style={styles.statValue}>Image</Text>
            <Text style={styles.statLabel}>MONAI AI</Text>
          </View>
          <View style={styles.statCard}>
            <Ionicons name="chatbubble-ellipses" size={scale(22)} color={COLORS.primary} />
            <Text style={styles.statValue}>24/7</Text>
            <Text style={styles.statLabel}>MediBot</Text>
          </View>
        </View>

        {/* Features */}
        <Text style={styles.sectionTitle}>Features</Text>

        <FeatureCard
          icon="chatbubbles"
          title="MediBot Chat"
          description="Ask health questions and get instant AI-powered answers"
          color={COLORS.primary}
          onPress={() => navigation.navigate('Chat')}
        />

        <FeatureCard
          icon="heart"
          title="Heart Disease Risk"
          description="Assess your cardiovascular health with ML prediction"
          color={COLORS.accent}
          onPress={() => navigation.navigate('Heart')}
        />

        <FeatureCard
          icon="scan"
          title="Medical Image Analysis"
          description="AI analysis for X-rays, skin lesions & eye health"
          color="#9B59B6"
          onPress={() => navigation.navigate('Scan')}
        />

        <FeatureCard
          icon="information-circle"
          title="Health Tips"
          description="Learn about maintaining a healthy lifestyle"
          color={COLORS.success}
          onPress={() => navigation.navigate('Chat')}
        />

        {/* Info Banner */}
        <View style={styles.infoBanner}>
          <Ionicons name="information-circle" size={scale(22)} color={COLORS.info} />
          <Text style={styles.infoText}>
            MediPlus provides health information only. Always consult healthcare professionals for medical advice.
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
  heroCard: {
    backgroundColor: COLORS.primary,
    borderRadius: SIZES.radiusLg,
    padding: scale(18),
    flexDirection: 'row',
    overflow: 'hidden',
    marginBottom: scale(18),
    marginTop: scale(14),
  },
  heroContent: {
    flex: 1,
    zIndex: 1,
  },
  heroTitle: {
    fontSize: SIZES.xxl,
    fontWeight: 'bold',
    color: COLORS.surface,
    marginBottom: scale(6),
  },
  heroSubtitle: {
    fontSize: SIZES.sm,
    color: 'rgba(255,255,255,0.8)',
    marginBottom: scale(14),
    lineHeight: SIZES.sm * 1.4,
  },
  heroButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.primaryDark,
    paddingHorizontal: scale(14),
    paddingVertical: scale(10),
    borderRadius: SIZES.radius,
    alignSelf: 'flex-start',
    gap: scale(6),
  },
  heroButtonText: {
    color: COLORS.surface,
    fontWeight: '600',
    fontSize: SIZES.sm,
  },
  heroIcon: {
    position: 'absolute',
    right: scale(-10),
    bottom: scale(-10),
    opacity: 0.3,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: scale(20),
  },
  statCard: {
    flex: 1,
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    padding: scale(12),
    alignItems: 'center',
    marginHorizontal: scale(4),
  },
  statValue: {
    fontSize: SIZES.sm,
    fontWeight: 'bold',
    color: COLORS.text,
    marginTop: scale(6),
  },
  statLabel: {
    fontSize: SIZES.xs,
    color: COLORS.textSecondary,
  },
  sectionTitle: {
    fontSize: SIZES.xl,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: scale(14),
  },
  featureCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    padding: scale(14),
    marginBottom: scale(10),
  },
  featureIcon: {
    width: scale(48),
    height: scale(48),
    borderRadius: scale(24),
    justifyContent: 'center',
    alignItems: 'center',
  },
  featureContent: {
    flex: 1,
    marginLeft: scale(12),
  },
  featureTitle: {
    fontSize: SIZES.md,
    fontWeight: '600',
    color: COLORS.text,
  },
  featureDescription: {
    fontSize: SIZES.xs,
    color: COLORS.textSecondary,
    marginTop: scale(2),
  },
  infoBanner: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: COLORS.info + '15',
    borderRadius: SIZES.radius,
    padding: scale(14),
    marginTop: scale(10),
    gap: scale(10),
  },
  infoText: {
    flex: 1,
    fontSize: SIZES.xs,
    color: COLORS.info,
    lineHeight: SIZES.xs * 1.6,
  },
});
