import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import { View, StyleSheet, Platform } from 'react-native';

import HomeScreen from '../screens/HomeScreen';
import ChatScreen from '../screens/ChatScreen';
import HeartScreen from '../screens/HeartScreen';
import ImageAnalysisScreen from '../screens/ImageAnalysisScreen';
import SettingsScreen from '../screens/SettingsScreen';
import SegmentationScreen from '../screens/SegmentationScreen';
import { COLORS, SIZES, COMPONENT_SIZES } from '../constants/theme';
import { scale, moderateScale } from '../utils/responsive';

type IconName = React.ComponentProps<typeof Ionicons>['name'];

// Navigation types
export type RootStackParamList = {
  MainTabs: undefined;
  Segmentation: undefined;
};

export type TabParamList = {
  Home: undefined;
  Chat: undefined;
  Heart: undefined;
  Scan: undefined;
  Settings: undefined;
};

const Tab = createBottomTabNavigator<TabParamList>();
const Stack = createNativeStackNavigator<RootStackParamList>();

function TabsNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarIcon: ({ focused, color }) => {
          let iconName: IconName;

          switch (route.name) {
            case 'Home':
              iconName = focused ? 'home' : 'home-outline';
              break;
            case 'Chat':
              iconName = focused ? 'chatbubbles' : 'chatbubbles-outline';
              break;
            case 'Heart':
              iconName = focused ? 'heart' : 'heart-outline';
              break;
            case 'Scan':
              iconName = focused ? 'scan' : 'scan-outline';
              break;
            case 'Settings':
              iconName = focused ? 'settings' : 'settings-outline';
              break;
            default:
              iconName = 'ellipse';
          }

          return (
            <View style={[styles.iconContainer, focused && styles.iconContainerFocused]}>
              <Ionicons name={iconName} size={scale(22)} color={color} />
            </View>
          );
        },
        tabBarActiveTintColor: COLORS.primary,
        tabBarInactiveTintColor: COLORS.textLight,
        tabBarStyle: styles.tabBar,
        tabBarLabelStyle: styles.tabBarLabel,
        tabBarHideOnKeyboard: true,
      })}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{ tabBarLabel: 'Home' }}
      />
      <Tab.Screen
        name="Chat"
        component={ChatScreen}
        options={{ tabBarLabel: 'MediBot' }}
      />
      <Tab.Screen
        name="Heart"
        component={HeartScreen}
        options={{ tabBarLabel: 'Heart' }}
      />
      <Tab.Screen
        name="Scan"
        component={ImageAnalysisScreen}
        options={{ tabBarLabel: 'Scan' }}
      />
      <Tab.Screen
        name="Settings"
        component={SettingsScreen}
        options={{ tabBarLabel: 'Settings' }}
      />
    </Tab.Navigator>
  );
}

// Root Navigator with Stack for modal screens
export default function RootNavigator() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="MainTabs" component={TabsNavigator} />
      <Stack.Screen
        name="Segmentation"
        component={SegmentationScreen}
        options={{
          presentation: 'modal',
          animation: 'slide_from_bottom',
        }}
      />
    </Stack.Navigator>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    position: 'absolute',
    bottom: COMPONENT_SIZES.tabBarBottom,
    left: COMPONENT_SIZES.tabBarBottom,
    right: COMPONENT_SIZES.tabBarBottom,
    height: COMPONENT_SIZES.tabBarHeight,
    backgroundColor: COLORS.surface,
    borderRadius: COMPONENT_SIZES.tabBarRadius,
    paddingBottom: Platform.OS === 'ios' ? scale(20) : scale(10),
    paddingTop: scale(10),
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 12,
    elevation: 8,
    borderTopWidth: 0,
  },
  tabBarLabel: {
    fontSize: moderateScale(10),
    fontWeight: '600',
    marginTop: scale(2),
  },
  iconContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconContainerFocused: {
    transform: [{ scale: 1.1 }],
  },
});
