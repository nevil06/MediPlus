import { moderateScale, scale, wp, hp, isSmallScreen, isTablet } from '../utils/responsive';

// MediPlus Theme Colors
export const COLORS = {
  primary: '#4A90D9',
  primaryDark: '#2E5C8A',
  primaryLight: '#7AB3E8',
  secondary: '#34C759',
  accent: '#FF6B6B',

  background: '#F5F7FA',
  surface: '#FFFFFF',
  card: '#FFFFFF',

  text: '#1A1A2E',
  textSecondary: '#6B7280',
  textLight: '#9CA3AF',

  border: '#E5E7EB',
  divider: '#F3F4F6',

  success: '#10B981',
  warning: '#F59E0B',
  error: '#EF4444',
  info: '#3B82F6',

  // Risk levels
  riskLow: '#10B981',
  riskModerate: '#F59E0B',
  riskHigh: '#EF4444',

  // Chat
  userBubble: '#4A90D9',
  botBubble: '#F3F4F6',
};

export const FONTS = {
  regular: 'System',
  medium: 'System',
  bold: 'System',
};

// Responsive sizes
export const SIZES = {
  // Font sizes - responsive
  xs: moderateScale(10),
  sm: moderateScale(12),
  md: moderateScale(14),
  lg: moderateScale(16),
  xl: moderateScale(18),
  xxl: moderateScale(22),
  xxxl: moderateScale(28),

  // Spacing - responsive
  padding: scale(16),
  paddingSm: scale(12),
  paddingLg: scale(20),
  margin: scale(16),

  // Border radius - responsive
  radius: scale(12),
  radiusSm: scale(8),
  radiusLg: scale(20),
  radiusXl: scale(28),

  // Icon sizes
  iconSm: scale(20),
  iconMd: scale(24),
  iconLg: scale(32),
  iconXl: scale(48),

  // Screen helpers
  screenWidth: wp(100),
  screenHeight: hp(100),
};

// Responsive component sizes
export const COMPONENT_SIZES = {
  // Tab bar
  tabBarHeight: scale(70),
  tabBarBottom: scale(16),
  tabBarRadius: scale(35),

  // Header
  headerHeight: scale(60),
  headerIconSize: scale(44),

  // Buttons
  buttonHeight: scale(48),
  buttonHeightSm: scale(40),
  buttonHeightLg: scale(56),

  // Input
  inputHeight: scale(48),
  inputHeightLg: scale(56),

  // Avatar
  avatarSm: scale(32),
  avatarMd: scale(44),
  avatarLg: scale(60),
  avatarXl: scale(80),

  // Cards
  cardPadding: scale(16),
  cardRadius: scale(16),
};

// Layout helpers
export const LAYOUT = {
  isSmallScreen,
  isTablet,
  maxWidth: isTablet ? 600 : wp(100),
  contentPadding: isSmallScreen ? scale(12) : scale(16),
};

// Shadow styles
export const SHADOWS = {
  small: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  medium: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.12,
    shadowRadius: 8,
    elevation: 4,
  },
  large: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.15,
    shadowRadius: 12,
    elevation: 8,
  },
};
