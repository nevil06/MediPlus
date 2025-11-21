import React from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';
import { COLORS, SIZES, COMPONENT_SIZES } from '../constants/theme';
import { scale, moderateScale } from '../utils/responsive';

export default function Header() {
  return (
    <View style={styles.header}>
      <View style={styles.logoContainer}>
        <Image
          source={require('../../assets/logo.png')}
          style={styles.logoImage}
          resizeMode="contain"
        />
      </View>
      <View style={styles.titleContainer}>
        <Text style={styles.title}>MediPlus</Text>
        <Text style={styles.subtitle}>Your Health Assistant</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.primary,
    paddingHorizontal: SIZES.padding,
    paddingVertical: scale(12),
    borderBottomLeftRadius: SIZES.radiusLg,
    borderBottomRightRadius: SIZES.radiusLg,
  },
  logoContainer: {
    width: COMPONENT_SIZES.headerIconSize,
    height: COMPONENT_SIZES.headerIconSize,
    borderRadius: COMPONENT_SIZES.headerIconSize / 2,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  logoImage: {
    width: COMPONENT_SIZES.headerIconSize - scale(8),
    height: COMPONENT_SIZES.headerIconSize - scale(8),
    borderRadius: (COMPONENT_SIZES.headerIconSize - scale(8)) / 2,
  },
  titleContainer: {
    marginLeft: scale(12),
  },
  title: {
    fontSize: SIZES.xl,
    fontWeight: 'bold',
    color: COLORS.surface,
  },
  subtitle: {
    fontSize: SIZES.sm,
    color: 'rgba(255,255,255,0.8)',
  },
});
