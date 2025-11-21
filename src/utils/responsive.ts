import { Dimensions, PixelRatio, Platform, StatusBar } from 'react-native';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Base dimensions (iPhone 14 as reference)
const BASE_WIDTH = 390;
const BASE_HEIGHT = 844;

// Scale based on screen width
export const scale = (size: number): number => {
  return (SCREEN_WIDTH / BASE_WIDTH) * size;
};

// Scale based on screen height
export const verticalScale = (size: number): number => {
  return (SCREEN_HEIGHT / BASE_HEIGHT) * size;
};

// Moderate scale - less aggressive scaling for fonts
export const moderateScale = (size: number, factor: number = 0.5): number => {
  return size + (scale(size) - size) * factor;
};

// Font scaling with pixel ratio consideration
export const fontScale = (size: number): number => {
  const newSize = scale(size);
  if (Platform.OS === 'ios') {
    return Math.round(PixelRatio.roundToNearestPixel(newSize));
  }
  return Math.round(PixelRatio.roundToNearestPixel(newSize)) - 2;
};

// Screen dimensions
export const screenWidth = SCREEN_WIDTH;
export const screenHeight = SCREEN_HEIGHT;

// Check if small screen (iPhone SE, older Android phones)
export const isSmallScreen = SCREEN_WIDTH < 375;

// Check if tablet
export const isTablet = SCREEN_WIDTH >= 768;

// Safe area values
export const statusBarHeight = Platform.OS === 'ios' ? 44 : StatusBar.currentHeight || 24;

// Responsive spacing
export const spacing = {
  xs: scale(4),
  sm: scale(8),
  md: scale(12),
  lg: scale(16),
  xl: scale(20),
  xxl: scale(24),
  xxxl: scale(32),
};

// Responsive font sizes
export const fontSize = {
  xs: moderateScale(10),
  sm: moderateScale(12),
  md: moderateScale(14),
  lg: moderateScale(16),
  xl: moderateScale(18),
  xxl: moderateScale(22),
  xxxl: moderateScale(28),
  display: moderateScale(36),
};

// Responsive border radius
export const radius = {
  sm: scale(8),
  md: scale(12),
  lg: scale(16),
  xl: scale(20),
  round: scale(100),
};

// Get percentage of screen width
export const wp = (percentage: number): number => {
  return (percentage * SCREEN_WIDTH) / 100;
};

// Get percentage of screen height
export const hp = (percentage: number): number => {
  return (percentage * SCREEN_HEIGHT) / 100;
};
