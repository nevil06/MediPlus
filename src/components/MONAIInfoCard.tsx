import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES } from '../constants/theme';
import { scale, moderateScale } from '../utils/responsive';
import { monaiService, MONAIInfo } from '../services/monaiService';

interface MONAIInfoCardProps {
  compact?: boolean;
  showRefresh?: boolean;
}

export default function MONAIInfoCard({ compact = false, showRefresh = true }: MONAIInfoCardProps) {
  const [info, setInfo] = useState<MONAIInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  const fetchInfo = async () => {
    setLoading(true);
    try {
      const result = await monaiService.getMonaiInfo();
      setInfo(result);
    } catch (error) {
      console.error('Failed to fetch MONAI info:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchInfo();
  }, []);

  if (loading) {
    return (
      <View style={[styles.container, compact && styles.containerCompact]}>
        <ActivityIndicator size="small" color={COLORS.primary} />
        <Text style={styles.loadingText}>Loading MONAI info...</Text>
      </View>
    );
  }

  if (!info || (!info.monai_available && !info.torch_available)) {
    return (
      <View style={[styles.container, styles.errorContainer]}>
        <Ionicons name="cloud-offline" size={scale(20)} color={COLORS.warning} />
        <View style={styles.errorContent}>
          <Text style={styles.errorText}>Backend not connected</Text>
          <Text style={styles.errorHint}>Start the backend server to enable MONAI features</Text>
        </View>
        {showRefresh && (
          <TouchableOpacity onPress={fetchInfo} style={styles.retryBtn}>
            <Ionicons name="refresh" size={scale(16)} color={COLORS.primary} />
          </TouchableOpacity>
        )}
      </View>
    );
  }

  const StatusBadge = ({ available, label }: { available: boolean; label: string }) => (
    <View style={[styles.badge, available ? styles.badgeActive : styles.badgeInactive]}>
      <Ionicons
        name={available ? 'checkmark-circle' : 'close-circle'}
        size={scale(14)}
        color={available ? COLORS.success : COLORS.textSecondary}
      />
      <Text style={[styles.badgeText, available && styles.badgeTextActive]}>{label}</Text>
    </View>
  );

  if (compact) {
    return (
      <View style={[styles.container, styles.containerCompact]}>
        <View style={styles.compactHeader}>
          <View style={styles.titleRow}>
            <Ionicons name="hardware-chip" size={scale(18)} color={COLORS.primary} />
            <Text style={styles.compactTitle}>MONAI Engine</Text>
          </View>
          <View style={styles.statusDot}>
            <View style={[styles.dot, info.monai_available ? styles.dotActive : styles.dotInactive]} />
            <Text style={styles.statusText}>
              {info.monai_available ? 'Active' : 'Inactive'}
            </Text>
          </View>
        </View>
        {info.monai_version && (
          <Text style={styles.versionText}>v{info.monai_version}</Text>
        )}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.header}
        onPress={() => setExpanded(!expanded)}
        activeOpacity={0.7}
      >
        <View style={styles.titleRow}>
          <View style={styles.iconContainer}>
            <Ionicons name="hardware-chip" size={scale(22)} color={COLORS.primary} />
          </View>
          <View>
            <Text style={styles.title}>MONAI Medical AI</Text>
            <Text style={styles.subtitle}>
              {info.monai_available ? 'Framework Active' : 'Framework Unavailable'}
            </Text>
          </View>
        </View>
        <View style={styles.headerRight}>
          {showRefresh && (
            <TouchableOpacity onPress={fetchInfo} style={styles.refreshBtn}>
              <Ionicons name="refresh" size={scale(18)} color={COLORS.textSecondary} />
            </TouchableOpacity>
          )}
          <Ionicons
            name={expanded ? 'chevron-up' : 'chevron-down'}
            size={scale(20)}
            color={COLORS.textSecondary}
          />
        </View>
      </TouchableOpacity>

      <View style={styles.badgeRow}>
        <StatusBadge available={info.monai_available} label="MONAI" />
        <StatusBadge available={info.torch_available} label="PyTorch" />
        <StatusBadge available={info.cuda_available} label="CUDA" />
      </View>

      {expanded && (
        <View style={styles.expandedContent}>
          {info.monai_version && (
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>MONAI Version</Text>
              <Text style={styles.infoValue}>{info.monai_version}</Text>
            </View>
          )}

          {info.cuda_device && (
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>GPU Device</Text>
              <Text style={styles.infoValue}>{info.cuda_device}</Text>
            </View>
          )}

          {info.networks && info.networks.length > 0 && (
            <View style={styles.listSection}>
              <Text style={styles.listTitle}>Available Networks</Text>
              <View style={styles.chipContainer}>
                {info.networks.slice(0, 8).map((network, index) => (
                  <View key={index} style={styles.chip}>
                    <Text style={styles.chipText}>{network}</Text>
                  </View>
                ))}
                {info.networks.length > 8 && (
                  <View style={[styles.chip, styles.chipMore]}>
                    <Text style={styles.chipTextMore}>+{info.networks.length - 8} more</Text>
                  </View>
                )}
              </View>
            </View>
          )}

          {info.transforms && info.transforms.length > 0 && (
            <View style={styles.listSection}>
              <Text style={styles.listTitle}>Available Transforms</Text>
              <View style={styles.chipContainer}>
                {info.transforms.slice(0, 6).map((transform, index) => (
                  <View key={index} style={styles.chip}>
                    <Text style={styles.chipText}>{transform}</Text>
                  </View>
                ))}
                {info.transforms.length > 6 && (
                  <View style={[styles.chip, styles.chipMore]}>
                    <Text style={styles.chipTextMore}>+{info.transforms.length - 6} more</Text>
                  </View>
                )}
              </View>
            </View>
          )}

          {info.losses && info.losses.length > 0 && (
            <View style={styles.listSection}>
              <Text style={styles.listTitle}>Loss Functions</Text>
              <Text style={styles.listCount}>{info.losses.length} available</Text>
            </View>
          )}

          {info.metrics && info.metrics.length > 0 && (
            <View style={styles.listSection}>
              <Text style={styles.listTitle}>Evaluation Metrics</Text>
              <Text style={styles.listCount}>{info.metrics.length} available</Text>
            </View>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.surface,
    borderRadius: scale(12),
    padding: scale(14),
    marginBottom: scale(16),
  },
  containerCompact: {
    padding: scale(12),
    marginBottom: scale(12),
  },
  errorContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: scale(10),
  },
  errorContent: {
    flex: 1,
  },
  errorHint: {
    fontSize: moderateScale(11),
    color: COLORS.textSecondary,
    marginTop: scale(2),
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  compactHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: scale(10),
  },
  iconContainer: {
    width: scale(40),
    height: scale(40),
    borderRadius: scale(20),
    backgroundColor: COLORS.primary + '15',
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: moderateScale(16),
    fontWeight: '600',
    color: COLORS.text,
  },
  compactTitle: {
    fontSize: moderateScale(14),
    fontWeight: '600',
    color: COLORS.text,
  },
  subtitle: {
    fontSize: moderateScale(12),
    color: COLORS.textSecondary,
    marginTop: scale(2),
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: scale(8),
  },
  refreshBtn: {
    padding: scale(4),
  },
  statusDot: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: scale(6),
  },
  dot: {
    width: scale(8),
    height: scale(8),
    borderRadius: scale(4),
  },
  dotActive: {
    backgroundColor: COLORS.success,
  },
  dotInactive: {
    backgroundColor: COLORS.textSecondary,
  },
  statusText: {
    fontSize: moderateScale(12),
    color: COLORS.textSecondary,
  },
  versionText: {
    fontSize: moderateScale(11),
    color: COLORS.textSecondary,
    marginTop: scale(4),
  },
  badgeRow: {
    flexDirection: 'row',
    gap: scale(8),
    marginTop: scale(12),
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: scale(10),
    paddingVertical: scale(4),
    borderRadius: scale(12),
    gap: scale(4),
  },
  badgeActive: {
    backgroundColor: COLORS.success + '15',
  },
  badgeInactive: {
    backgroundColor: COLORS.border,
  },
  badgeText: {
    fontSize: moderateScale(11),
    color: COLORS.textSecondary,
    fontWeight: '500',
  },
  badgeTextActive: {
    color: COLORS.success,
  },
  expandedContent: {
    marginTop: scale(16),
    paddingTop: scale(16),
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: scale(10),
  },
  infoLabel: {
    fontSize: moderateScale(13),
    color: COLORS.textSecondary,
  },
  infoValue: {
    fontSize: moderateScale(13),
    color: COLORS.text,
    fontWeight: '500',
  },
  listSection: {
    marginTop: scale(12),
  },
  listTitle: {
    fontSize: moderateScale(13),
    fontWeight: '600',
    color: COLORS.text,
    marginBottom: scale(8),
  },
  listCount: {
    fontSize: moderateScale(12),
    color: COLORS.textSecondary,
  },
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: scale(6),
  },
  chip: {
    backgroundColor: COLORS.background,
    paddingHorizontal: scale(10),
    paddingVertical: scale(4),
    borderRadius: scale(8),
  },
  chipMore: {
    backgroundColor: COLORS.primary + '15',
  },
  chipText: {
    fontSize: moderateScale(11),
    color: COLORS.text,
  },
  chipTextMore: {
    fontSize: moderateScale(11),
    color: COLORS.primary,
    fontWeight: '500',
  },
  loadingText: {
    fontSize: moderateScale(12),
    color: COLORS.textSecondary,
    marginLeft: scale(8),
  },
  errorText: {
    fontSize: moderateScale(13),
    color: COLORS.error,
    flex: 1,
  },
  retryBtn: {
    paddingHorizontal: scale(12),
    paddingVertical: scale(6),
  },
  retryText: {
    fontSize: moderateScale(13),
    color: COLORS.primary,
    fontWeight: '500',
  },
});
