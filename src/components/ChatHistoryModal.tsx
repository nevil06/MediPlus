// Chat History Modal - Displays all saved chat sessions
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  Modal,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  TextInput,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES } from '../constants/theme';
import { scale, wp, hp } from '../utils/responsive';
import {
  chatStorageService,
  ChatSessionPreview,
} from '../services/chatStorageService';

interface ChatHistoryModalProps {
  visible: boolean;
  onClose: () => void;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
  currentSessionId: string | null;
}

export default function ChatHistoryModal({
  visible,
  onClose,
  onSelectSession,
  onNewChat,
  currentSessionId,
}: ChatHistoryModalProps) {
  const [sessions, setSessions] = useState<ChatSessionPreview[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  useEffect(() => {
    if (visible) {
      loadSessions();
    }
  }, [visible]);

  const loadSessions = async () => {
    setIsLoading(true);
    try {
      const allSessions = await chatStorageService.getAllSessionPreviews();
      setSessions(allSessions);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = async (query: string) => {
    setSearchQuery(query);
    if (query.trim()) {
      const results = await chatStorageService.searchSessions(query);
      setSessions(results);
    } else {
      loadSessions();
    }
  };

  const handleDeleteSession = (sessionId: string, title: string) => {
    Alert.alert(
      'Delete Chat',
      `Are you sure you want to delete "${title}"?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            await chatStorageService.deleteSession(sessionId);
            loadSessions();
            if (sessionId === currentSessionId) {
              onNewChat();
            }
          },
        },
      ]
    );
  };

  const handleRenameSession = async (sessionId: string) => {
    if (editTitle.trim()) {
      await chatStorageService.renameSession(sessionId, editTitle.trim());
      setEditingSessionId(null);
      setEditTitle('');
      loadSessions();
    }
  };

  const handleDeleteAllChats = () => {
    Alert.alert(
      'Delete All Chats',
      'Are you sure you want to delete all chat history? This cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete All',
          style: 'destructive',
          onPress: async () => {
            await chatStorageService.deleteAllSessions();
            loadSessions();
            onNewChat();
          },
        },
      ]
    );
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return date.toLocaleDateString([], { weekday: 'short' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };

  const renderSessionItem = ({ item }: { item: ChatSessionPreview }) => {
    const isCurrentSession = item.id === currentSessionId;
    const isEditing = editingSessionId === item.id;

    return (
      <TouchableOpacity
        style={[
          styles.sessionItem,
          isCurrentSession && styles.currentSession,
        ]}
        onPress={() => {
          if (!isEditing) {
            onSelectSession(item.id);
            onClose();
          }
        }}
        onLongPress={() => {
          setEditingSessionId(item.id);
          setEditTitle(item.title);
        }}
      >
        <View style={styles.sessionIcon}>
          <Ionicons
            name={isCurrentSession ? 'chatbubbles' : 'chatbubble-outline'}
            size={scale(20)}
            color={isCurrentSession ? COLORS.primary : COLORS.textSecondary}
          />
        </View>

        <View style={styles.sessionContent}>
          {isEditing ? (
            <View style={styles.editContainer}>
              <TextInput
                style={styles.editInput}
                value={editTitle}
                onChangeText={setEditTitle}
                autoFocus
                onBlur={() => handleRenameSession(item.id)}
                onSubmitEditing={() => handleRenameSession(item.id)}
              />
            </View>
          ) : (
            <>
              <Text
                style={[
                  styles.sessionTitle,
                  isCurrentSession && styles.currentSessionText,
                ]}
                numberOfLines={1}
              >
                {item.title}
              </Text>
              <Text style={styles.sessionPreview} numberOfLines={1}>
                {item.lastMessage}
              </Text>
            </>
          )}
        </View>

        <View style={styles.sessionMeta}>
          <Text style={styles.sessionDate}>{formatDate(item.updatedAt)}</Text>
          <Text style={styles.messageCount}>{item.messageCount} msgs</Text>
        </View>

        <TouchableOpacity
          style={styles.deleteButton}
          onPress={() => handleDeleteSession(item.id, item.title)}
        >
          <Ionicons name="trash-outline" size={scale(18)} color={COLORS.error} />
        </TouchableOpacity>
      </TouchableOpacity>
    );
  };

  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={onClose}
    >
      <SafeAreaView style={styles.container} edges={['top']}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={onClose} style={styles.closeButton}>
            <Ionicons name="close" size={scale(24)} color={COLORS.text} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Chat History</Text>
          <TouchableOpacity
            onPress={() => {
              onNewChat();
              onClose();
            }}
            style={styles.newChatButton}
          >
            <Ionicons name="add-circle" size={scale(24)} color={COLORS.primary} />
          </TouchableOpacity>
        </View>

        {/* Search Bar */}
        <View style={styles.searchContainer}>
          <Ionicons
            name="search"
            size={scale(18)}
            color={COLORS.textSecondary}
            style={styles.searchIcon}
          />
          <TextInput
            style={styles.searchInput}
            placeholder="Search conversations..."
            placeholderTextColor={COLORS.textLight}
            value={searchQuery}
            onChangeText={handleSearch}
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity onPress={() => handleSearch('')}>
              <Ionicons
                name="close-circle"
                size={scale(18)}
                color={COLORS.textSecondary}
              />
            </TouchableOpacity>
          )}
        </View>

        {/* Session Count */}
        <View style={styles.countContainer}>
          <Text style={styles.countText}>
            {sessions.length} {sessions.length === 1 ? 'conversation' : 'conversations'}
          </Text>
          {sessions.length > 0 && (
            <TouchableOpacity onPress={handleDeleteAllChats}>
              <Text style={styles.clearAllText}>Clear All</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Sessions List */}
        {isLoading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={COLORS.primary} />
            <Text style={styles.loadingText}>Loading chat history...</Text>
          </View>
        ) : sessions.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Ionicons
              name="chatbubbles-outline"
              size={scale(60)}
              color={COLORS.textLight}
            />
            <Text style={styles.emptyTitle}>
              {searchQuery ? 'No results found' : 'No chat history'}
            </Text>
            <Text style={styles.emptySubtitle}>
              {searchQuery
                ? 'Try a different search term'
                : 'Start a new conversation to see it here'}
            </Text>
            {!searchQuery && (
              <TouchableOpacity
                style={styles.startChatButton}
                onPress={() => {
                  onNewChat();
                  onClose();
                }}
              >
                <Text style={styles.startChatButtonText}>Start New Chat</Text>
              </TouchableOpacity>
            )}
          </View>
        ) : (
          <FlatList
            data={sessions}
            renderItem={renderSessionItem}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.listContent}
            showsVerticalScrollIndicator={false}
          />
        )}

        {/* Hint */}
        <View style={styles.hintContainer}>
          <Text style={styles.hintText}>
            Long press on a chat to rename it
          </Text>
        </View>
      </SafeAreaView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: scale(16),
    paddingVertical: scale(12),
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
    backgroundColor: COLORS.surface,
  },
  closeButton: {
    padding: scale(4),
  },
  headerTitle: {
    fontSize: SIZES.xl,
    fontWeight: '600',
    color: COLORS.text,
  },
  newChatButton: {
    padding: scale(4),
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    margin: scale(16),
    paddingHorizontal: scale(12),
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  searchIcon: {
    marginRight: scale(8),
  },
  searchInput: {
    flex: 1,
    paddingVertical: scale(12),
    fontSize: SIZES.md,
    color: COLORS.text,
  },
  countContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: scale(16),
    paddingBottom: scale(8),
  },
  countText: {
    fontSize: SIZES.sm,
    color: COLORS.textSecondary,
  },
  clearAllText: {
    fontSize: SIZES.sm,
    color: COLORS.error,
    fontWeight: '500',
  },
  listContent: {
    paddingHorizontal: scale(16),
    paddingBottom: scale(20),
  },
  sessionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: scale(12),
    marginBottom: scale(8),
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  currentSession: {
    borderColor: COLORS.primary,
    backgroundColor: COLORS.primaryLight + '15',
  },
  sessionIcon: {
    width: scale(40),
    height: scale(40),
    borderRadius: scale(20),
    backgroundColor: COLORS.background,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: scale(12),
  },
  sessionContent: {
    flex: 1,
    marginRight: scale(8),
  },
  sessionTitle: {
    fontSize: SIZES.md,
    fontWeight: '500',
    color: COLORS.text,
    marginBottom: scale(2),
  },
  currentSessionText: {
    color: COLORS.primary,
  },
  sessionPreview: {
    fontSize: SIZES.sm,
    color: COLORS.textSecondary,
  },
  sessionMeta: {
    alignItems: 'flex-end',
    marginRight: scale(8),
  },
  sessionDate: {
    fontSize: SIZES.xs,
    color: COLORS.textLight,
    marginBottom: scale(2),
  },
  messageCount: {
    fontSize: SIZES.xs,
    color: COLORS.textLight,
  },
  deleteButton: {
    padding: scale(8),
  },
  editContainer: {
    flex: 1,
  },
  editInput: {
    fontSize: SIZES.md,
    color: COLORS.text,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.primary,
    paddingVertical: scale(4),
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: scale(12),
    fontSize: SIZES.md,
    color: COLORS.textSecondary,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: scale(40),
  },
  emptyTitle: {
    fontSize: SIZES.lg,
    fontWeight: '600',
    color: COLORS.text,
    marginTop: scale(16),
  },
  emptySubtitle: {
    fontSize: SIZES.md,
    color: COLORS.textSecondary,
    textAlign: 'center',
    marginTop: scale(8),
  },
  startChatButton: {
    marginTop: scale(24),
    paddingHorizontal: scale(24),
    paddingVertical: scale(12),
    backgroundColor: COLORS.primary,
    borderRadius: SIZES.radius,
  },
  startChatButtonText: {
    color: COLORS.surface,
    fontSize: SIZES.md,
    fontWeight: '600',
  },
  hintContainer: {
    padding: scale(12),
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    backgroundColor: COLORS.surface,
  },
  hintText: {
    fontSize: SIZES.xs,
    color: COLORS.textLight,
  },
});
