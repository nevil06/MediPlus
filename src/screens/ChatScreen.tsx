import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Keyboard,
  TouchableWithoutFeedback,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES, COMPONENT_SIZES } from '../constants/theme';
import { groqService, Message } from '../services/groqService';
import {
  chatStorageService,
  ChatMessage,
  ChatSession,
} from '../services/chatStorageService';
import Header from '../components/Header';
import ChatHistoryModal from '../components/ChatHistoryModal';
import { scale, wp } from '../utils/responsive';

export default function ChatScreen() {
  // State
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [isHistoryVisible, setIsHistoryVisible] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);

  const flatListRef = useRef<FlatList>(null);

  const TAB_BAR_SPACE = COMPONENT_SIZES.tabBarHeight + COMPONENT_SIZES.tabBarBottom;

  // Initialize - load or create session
  useEffect(() => {
    initializeChat();
  }, []);

  const initializeChat = async () => {
    try {
      const session = await chatStorageService.getOrCreateActiveSession();
      setCurrentSession(session);
      setMessages(session.messages.map(msg => ({
        ...msg,
        timestamp: msg.timestamp,
      })));
      setIsInitialized(true);
    } catch (error) {
      console.error('Failed to initialize chat:', error);
      // Fallback to default welcome message
      setMessages([{
        id: 'welcome',
        role: 'assistant',
        content: "Hello! I'm MediBot, your personal health assistant.\n\nI can help you with:\n• General health questions\n• Understanding symptoms\n• Wellness tips\n• Heart health guidance\n\nHow can I assist you today?",
        timestamp: new Date().toISOString(),
      }]);
      setIsInitialized(true);
    }
  };

  // Keyboard listeners
  useEffect(() => {
    const keyboardDidShowListener = Keyboard.addListener(
      'keyboardDidShow',
      () => setKeyboardVisible(true)
    );
    const keyboardDidHideListener = Keyboard.addListener(
      'keyboardDidHide',
      () => setKeyboardVisible(false)
    );

    return () => {
      keyboardDidShowListener.remove();
      keyboardDidHideListener.remove();
    };
  }, []);

  // Scroll to end when messages change
  useEffect(() => {
    if (flatListRef.current && messages.length > 0) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages]);

  // Create new chat session
  const handleNewChat = useCallback(async () => {
    try {
      const newSession = await chatStorageService.createSession();
      await chatStorageService.setActiveSessionId(newSession.id);
      setCurrentSession(newSession);
      setMessages(newSession.messages);
    } catch (error) {
      console.error('Failed to create new chat:', error);
      Alert.alert('Error', 'Failed to create new chat. Please try again.');
    }
  }, []);

  // Load a specific session
  const handleSelectSession = useCallback(async (sessionId: string) => {
    try {
      const session = await chatStorageService.getSession(sessionId);
      if (session) {
        await chatStorageService.setActiveSessionId(sessionId);
        setCurrentSession(session);
        setMessages(session.messages);
      }
    } catch (error) {
      console.error('Failed to load session:', error);
      Alert.alert('Error', 'Failed to load chat. Please try again.');
    }
  }, []);

  // Send message
  const sendMessage = async () => {
    if (!inputText.trim() || isLoading || !currentSession) return;

    const userMessageContent = inputText.trim();
    setInputText('');
    setIsLoading(true);

    // Create user message
    const userMessage: ChatMessage = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: userMessageContent,
      timestamp: new Date().toISOString(),
    };

    // Optimistically add user message to UI
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    try {
      // Save user message to storage
      await chatStorageService.addMessage(currentSession.id, {
        role: 'user',
        content: userMessageContent,
      });

      // Prepare conversation history for API
      const history: Message[] = messages.slice(-10).map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      // Send to Groq API
      const response = await groqService.sendMessage(userMessageContent, history);

      // Create bot response message
      const botMessage: ChatMessage = {
        id: `msg_${Date.now() + 1}`,
        role: 'assistant',
        content: response.success
          ? response.message!
          : "I'm sorry, I'm having trouble connecting right now. Please try again later.",
        timestamp: new Date().toISOString(),
      };

      // Update UI with bot response
      setMessages((prev) => [...prev, botMessage]);

      // Save bot message to storage
      await chatStorageService.addMessage(currentSession.id, {
        role: 'assistant',
        content: botMessage.content,
      });

      // Refresh current session state
      const updatedSession = await chatStorageService.getSession(currentSession.id);
      if (updatedSession) {
        setCurrentSession(updatedSession);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message
      const errorMessage: ChatMessage = {
        id: `msg_${Date.now() + 1}`,
        role: 'assistant',
        content: "I'm sorry, something went wrong. Please try again.",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Render message bubble
  const renderMessage = ({ item }: { item: ChatMessage }) => {
    const isUser = item.role === 'user';
    return (
      <View
        style={[
          styles.messageBubble,
          isUser ? styles.userBubble : styles.botBubble,
        ]}
      >
        {!isUser && (
          <View style={styles.botAvatar}>
            <Ionicons name="medical" size={scale(14)} color={COLORS.primary} />
          </View>
        )}
        <View
          style={[
            styles.messageContent,
            isUser ? styles.userContent : styles.botContent,
          ]}
        >
          <Text
            style={[
              styles.messageText,
              isUser ? styles.userText : styles.botText,
            ]}
          >
            {item.content}
          </Text>
        </View>
      </View>
    );
  };

  // Loading state while initializing
  if (!isInitialized) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <Header />
        <View style={styles.initializingContainer}>
          <ActivityIndicator size="large" color={COLORS.primary} />
          <Text style={styles.initializingText}>Loading MediBot...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <KeyboardAvoidingView
        style={styles.keyboardAvoid}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 0}
      >
        <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
          <View style={styles.inner}>
            <Header />

            {/* Chat Action Bar */}
            <View style={styles.actionBar}>
              <TouchableOpacity
                style={styles.actionButton}
                onPress={() => setIsHistoryVisible(true)}
              >
                <Ionicons name="time-outline" size={scale(20)} color={COLORS.primary} />
                <Text style={styles.actionButtonText}>History</Text>
              </TouchableOpacity>

              <View style={styles.sessionInfo}>
                <Text style={styles.sessionTitle} numberOfLines={1}>
                  {currentSession?.title || 'New Chat'}
                </Text>
                <Text style={styles.sessionMeta}>
                  {currentSession?.messageCount || 0} messages
                </Text>
              </View>

              <TouchableOpacity
                style={styles.actionButton}
                onPress={handleNewChat}
              >
                <Ionicons name="add-circle-outline" size={scale(20)} color={COLORS.primary} />
                <Text style={styles.actionButtonText}>New</Text>
              </TouchableOpacity>
            </View>

            {/* Chat Messages */}
            <FlatList
              ref={flatListRef}
              data={messages}
              renderItem={renderMessage}
              keyExtractor={(item) => item.id}
              contentContainerStyle={[
                styles.messagesList,
                { paddingBottom: keyboardVisible ? scale(10) : TAB_BAR_SPACE + scale(10) }
              ]}
              showsVerticalScrollIndicator={false}
              onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
            />

            {/* Loading Indicator */}
            {isLoading && (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="small" color={COLORS.primary} />
                <Text style={styles.loadingText}>MediBot is thinking...</Text>
              </View>
            )}

            {/* Input Area */}
            <View style={[
              styles.inputContainer,
              { marginBottom: keyboardVisible ? 0 : TAB_BAR_SPACE }
            ]}>
              <TextInput
                style={styles.textInput}
                value={inputText}
                onChangeText={setInputText}
                placeholder="Ask me anything about health..."
                placeholderTextColor={COLORS.textLight}
                multiline
                maxLength={1000}
                onFocus={() => {
                  setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 300);
                }}
              />
              <TouchableOpacity
                style={[
                  styles.sendButton,
                  (!inputText.trim() || isLoading) && styles.sendButtonDisabled,
                ]}
                onPress={sendMessage}
                disabled={!inputText.trim() || isLoading}
              >
                <Ionicons
                  name="send"
                  size={scale(18)}
                  color={inputText.trim() && !isLoading ? COLORS.surface : COLORS.textLight}
                />
              </TouchableOpacity>
            </View>
          </View>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>

      {/* Chat History Modal */}
      <ChatHistoryModal
        visible={isHistoryVisible}
        onClose={() => setIsHistoryVisible(false)}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        currentSessionId={currentSession?.id || null}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  keyboardAvoid: {
    flex: 1,
  },
  inner: {
    flex: 1,
  },
  initializingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  initializingText: {
    marginTop: scale(12),
    fontSize: SIZES.md,
    color: COLORS.textSecondary,
  },
  actionBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: scale(12),
    paddingVertical: scale(8),
    backgroundColor: COLORS.surface,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: scale(10),
    paddingVertical: scale(6),
    borderRadius: SIZES.radius,
    backgroundColor: COLORS.primaryLight + '20',
  },
  actionButtonText: {
    marginLeft: scale(4),
    fontSize: SIZES.sm,
    color: COLORS.primary,
    fontWeight: '500',
  },
  sessionInfo: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: scale(8),
  },
  sessionTitle: {
    fontSize: SIZES.sm,
    fontWeight: '600',
    color: COLORS.text,
    maxWidth: wp(40),
  },
  sessionMeta: {
    fontSize: SIZES.xs,
    color: COLORS.textSecondary,
  },
  messagesList: {
    padding: SIZES.padding,
  },
  messageBubble: {
    flexDirection: 'row',
    marginVertical: scale(4),
    maxWidth: wp(85),
  },
  userBubble: {
    alignSelf: 'flex-end',
  },
  botBubble: {
    alignSelf: 'flex-start',
  },
  botAvatar: {
    width: COMPONENT_SIZES.avatarSm,
    height: COMPONENT_SIZES.avatarSm,
    borderRadius: COMPONENT_SIZES.avatarSm / 2,
    backgroundColor: COLORS.primaryLight + '30',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: scale(8),
  },
  messageContent: {
    padding: scale(12),
    borderRadius: SIZES.radius,
    maxWidth: '100%',
    flexShrink: 1,
  },
  userContent: {
    backgroundColor: COLORS.primary,
    borderBottomRightRadius: scale(4),
  },
  botContent: {
    backgroundColor: COLORS.surface,
    borderBottomLeftRadius: scale(4),
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  messageText: {
    fontSize: SIZES.md,
    lineHeight: SIZES.md * 1.5,
  },
  userText: {
    color: COLORS.surface,
  },
  botText: {
    color: COLORS.text,
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: SIZES.padding,
    paddingVertical: scale(8),
  },
  loadingText: {
    marginLeft: scale(8),
    color: COLORS.textSecondary,
    fontSize: SIZES.sm,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    paddingHorizontal: SIZES.padding,
    paddingVertical: scale(10),
    backgroundColor: COLORS.surface,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
  },
  textInput: {
    flex: 1,
    minHeight: COMPONENT_SIZES.inputHeight,
    maxHeight: scale(100),
    paddingHorizontal: scale(14),
    paddingVertical: scale(10),
    backgroundColor: COLORS.background,
    borderRadius: SIZES.radiusLg,
    fontSize: SIZES.md,
    color: COLORS.text,
  },
  sendButton: {
    width: COMPONENT_SIZES.inputHeight,
    height: COMPONENT_SIZES.inputHeight,
    borderRadius: COMPONENT_SIZES.inputHeight / 2,
    backgroundColor: COLORS.primary,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: scale(8),
  },
  sendButtonDisabled: {
    backgroundColor: COLORS.border,
  },
});
