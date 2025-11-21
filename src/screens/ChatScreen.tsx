import React, { useState, useRef, useEffect } from 'react';
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
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES, COMPONENT_SIZES } from '../constants/theme';
import { groqService, Message } from '../services/groqService';
import Header from '../components/Header';
import { scale, wp } from '../utils/responsive';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export default function ChatScreen() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hello! I'm MediBot, your personal health assistant.\n\nI can help you with:\n• General health questions\n• Understanding symptoms\n• Wellness tips\n• Heart health guidance\n\nHow can I assist you today?",
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  const TAB_BAR_SPACE = COMPONENT_SIZES.tabBarHeight + COMPONENT_SIZES.tabBarBottom;

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

  const sendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputText.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    const history: Message[] = messages.slice(-10).map((msg) => ({
      role: msg.role,
      content: msg.content,
    }));

    const response = await groqService.sendMessage(inputText.trim(), history);

    const botMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: response.success
        ? response.message!
        : "I'm sorry, I'm having trouble connecting right now. Please try again later.",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, botMessage]);
    setIsLoading(false);
  };

  useEffect(() => {
    if (flatListRef.current) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages]);

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
