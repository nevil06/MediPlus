// Chat Storage Service for MediBot - Handles chat persistence and session management
import AsyncStorage from '@react-native-async-storage/async-storage';

// Storage keys
const STORAGE_KEYS = {
  CHAT_SESSIONS: 'medibot_chat_sessions',
  ACTIVE_SESSION_ID: 'medibot_active_session_id',
  USER_PREFERENCES: 'medibot_user_preferences',
};

// Types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string; // ISO string for serialization
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
  messageCount: number;
}

export interface ChatSessionPreview {
  id: string;
  title: string;
  lastMessage: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
}

export interface UserPreferences {
  autoSave: boolean;
  maxSessionsToKeep: number;
}

// Default welcome message
const DEFAULT_WELCOME_MESSAGE: ChatMessage = {
  id: 'welcome',
  role: 'assistant',
  content: "Hello! I'm MediBot, your personal health assistant.\n\nI can help you with:\n• General health questions\n• Understanding symptoms\n• Wellness tips\n• Heart health guidance\n\nHow can I assist you today?",
  timestamp: new Date().toISOString(),
};

// Default preferences
const DEFAULT_PREFERENCES: UserPreferences = {
  autoSave: true,
  maxSessionsToKeep: 50,
};

class ChatStorageService {
  private cachedSessions: Map<string, ChatSession> = new Map();
  private isInitialized: boolean = false;

  // Initialize the service
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Load all sessions into cache for faster access
      const sessionsJson = await AsyncStorage.getItem(STORAGE_KEYS.CHAT_SESSIONS);
      if (sessionsJson) {
        const sessions: ChatSession[] = JSON.parse(sessionsJson);
        sessions.forEach(session => {
          this.cachedSessions.set(session.id, session);
        });
      }
      this.isInitialized = true;
    } catch (error) {
      console.error('Failed to initialize ChatStorageService:', error);
      this.isInitialized = true; // Continue anyway
    }
  }

  // Generate unique ID
  private generateId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Generate session title from first user message
  private generateTitle(messages: ChatMessage[]): string {
    const firstUserMessage = messages.find(m => m.role === 'user');
    if (firstUserMessage) {
      const title = firstUserMessage.content.substring(0, 50);
      return title.length < firstUserMessage.content.length ? `${title}...` : title;
    }
    return `Chat ${new Date().toLocaleDateString()}`;
  }

  // Create a new chat session
  async createSession(): Promise<ChatSession> {
    await this.initialize();

    const now = new Date().toISOString();
    const session: ChatSession = {
      id: this.generateId(),
      title: 'New Chat',
      messages: [{ ...DEFAULT_WELCOME_MESSAGE, timestamp: now }],
      createdAt: now,
      updatedAt: now,
      messageCount: 1,
    };

    this.cachedSessions.set(session.id, session);
    await this.persistSessions();

    return session;
  }

  // Get a session by ID
  async getSession(sessionId: string): Promise<ChatSession | null> {
    await this.initialize();
    return this.cachedSessions.get(sessionId) || null;
  }

  // Get the active session ID
  async getActiveSessionId(): Promise<string | null> {
    try {
      return await AsyncStorage.getItem(STORAGE_KEYS.ACTIVE_SESSION_ID);
    } catch (error) {
      console.error('Failed to get active session ID:', error);
      return null;
    }
  }

  // Set the active session ID
  async setActiveSessionId(sessionId: string): Promise<void> {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.ACTIVE_SESSION_ID, sessionId);
    } catch (error) {
      console.error('Failed to set active session ID:', error);
    }
  }

  // Get or create active session
  async getOrCreateActiveSession(): Promise<ChatSession> {
    await this.initialize();

    const activeId = await this.getActiveSessionId();
    if (activeId) {
      const session = await this.getSession(activeId);
      if (session) return session;
    }

    // No active session, create new one
    const newSession = await this.createSession();
    await this.setActiveSessionId(newSession.id);
    return newSession;
  }

  // Add a message to a session
  async addMessage(sessionId: string, message: Omit<ChatMessage, 'id' | 'timestamp'>): Promise<ChatMessage> {
    await this.initialize();

    const session = this.cachedSessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    const newMessage: ChatMessage = {
      ...message,
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
    };

    session.messages.push(newMessage);
    session.updatedAt = newMessage.timestamp;
    session.messageCount = session.messages.length;

    // Update title if this is the first user message
    if (message.role === 'user' && session.title === 'New Chat') {
      session.title = this.generateTitle(session.messages);
    }

    await this.persistSessions();
    return newMessage;
  }

  // Update session messages (batch update)
  async updateSessionMessages(sessionId: string, messages: ChatMessage[]): Promise<void> {
    await this.initialize();

    const session = this.cachedSessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    session.messages = messages;
    session.updatedAt = new Date().toISOString();
    session.messageCount = messages.length;

    // Update title based on messages
    if (session.title === 'New Chat') {
      session.title = this.generateTitle(messages);
    }

    await this.persistSessions();
  }

  // Get all session previews (for history list)
  async getAllSessionPreviews(): Promise<ChatSessionPreview[]> {
    await this.initialize();

    const previews: ChatSessionPreview[] = [];
    this.cachedSessions.forEach(session => {
      const lastMessage = session.messages[session.messages.length - 1];
      previews.push({
        id: session.id,
        title: session.title,
        lastMessage: lastMessage ?
          (lastMessage.content.length > 60 ?
            lastMessage.content.substring(0, 60) + '...' :
            lastMessage.content) :
          '',
        createdAt: session.createdAt,
        updatedAt: session.updatedAt,
        messageCount: session.messageCount,
      });
    });

    // Sort by updatedAt (most recent first)
    return previews.sort((a, b) =>
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
  }

  // Delete a session
  async deleteSession(sessionId: string): Promise<void> {
    await this.initialize();

    this.cachedSessions.delete(sessionId);
    await this.persistSessions();

    // If this was the active session, clear the active session ID
    const activeId = await this.getActiveSessionId();
    if (activeId === sessionId) {
      await AsyncStorage.removeItem(STORAGE_KEYS.ACTIVE_SESSION_ID);
    }
  }

  // Delete all sessions
  async deleteAllSessions(): Promise<void> {
    await this.initialize();

    this.cachedSessions.clear();
    await AsyncStorage.removeItem(STORAGE_KEYS.CHAT_SESSIONS);
    await AsyncStorage.removeItem(STORAGE_KEYS.ACTIVE_SESSION_ID);
  }

  // Rename a session
  async renameSession(sessionId: string, newTitle: string): Promise<void> {
    await this.initialize();

    const session = this.cachedSessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    session.title = newTitle;
    session.updatedAt = new Date().toISOString();
    await this.persistSessions();
  }

  // Persist sessions to AsyncStorage
  private async persistSessions(): Promise<void> {
    try {
      const sessions = Array.from(this.cachedSessions.values());

      // Enforce max sessions limit
      const preferences = await this.getPreferences();
      if (sessions.length > preferences.maxSessionsToKeep) {
        // Sort by updatedAt and keep only the most recent
        sessions.sort((a, b) =>
          new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
        );
        const sessionsToKeep = sessions.slice(0, preferences.maxSessionsToKeep);

        // Update cache
        this.cachedSessions.clear();
        sessionsToKeep.forEach(s => this.cachedSessions.set(s.id, s));

        await AsyncStorage.setItem(STORAGE_KEYS.CHAT_SESSIONS, JSON.stringify(sessionsToKeep));
      } else {
        await AsyncStorage.setItem(STORAGE_KEYS.CHAT_SESSIONS, JSON.stringify(sessions));
      }
    } catch (error) {
      console.error('Failed to persist sessions:', error);
    }
  }

  // Get user preferences
  async getPreferences(): Promise<UserPreferences> {
    try {
      const prefsJson = await AsyncStorage.getItem(STORAGE_KEYS.USER_PREFERENCES);
      if (prefsJson) {
        return { ...DEFAULT_PREFERENCES, ...JSON.parse(prefsJson) };
      }
      return DEFAULT_PREFERENCES;
    } catch (error) {
      console.error('Failed to get preferences:', error);
      return DEFAULT_PREFERENCES;
    }
  }

  // Update user preferences
  async updatePreferences(preferences: Partial<UserPreferences>): Promise<void> {
    try {
      const currentPrefs = await this.getPreferences();
      const newPrefs = { ...currentPrefs, ...preferences };
      await AsyncStorage.setItem(STORAGE_KEYS.USER_PREFERENCES, JSON.stringify(newPrefs));
    } catch (error) {
      console.error('Failed to update preferences:', error);
    }
  }

  // Export all chats (for backup)
  async exportAllChats(): Promise<string> {
    await this.initialize();
    const sessions = Array.from(this.cachedSessions.values());
    return JSON.stringify(sessions, null, 2);
  }

  // Import chats (from backup)
  async importChats(jsonData: string): Promise<number> {
    try {
      const sessions: ChatSession[] = JSON.parse(jsonData);
      let imported = 0;

      for (const session of sessions) {
        if (session.id && session.messages && Array.isArray(session.messages)) {
          // Generate new ID to avoid conflicts
          const newSession: ChatSession = {
            ...session,
            id: this.generateId(),
          };
          this.cachedSessions.set(newSession.id, newSession);
          imported++;
        }
      }

      await this.persistSessions();
      return imported;
    } catch (error) {
      console.error('Failed to import chats:', error);
      throw new Error('Invalid chat data format');
    }
  }

  // Search sessions by content
  async searchSessions(query: string): Promise<ChatSessionPreview[]> {
    await this.initialize();
    const lowerQuery = query.toLowerCase();

    const results: ChatSessionPreview[] = [];
    this.cachedSessions.forEach(session => {
      const titleMatch = session.title.toLowerCase().includes(lowerQuery);
      const messageMatch = session.messages.some(m =>
        m.content.toLowerCase().includes(lowerQuery)
      );

      if (titleMatch || messageMatch) {
        const lastMessage = session.messages[session.messages.length - 1];
        results.push({
          id: session.id,
          title: session.title,
          lastMessage: lastMessage ?
            (lastMessage.content.length > 60 ?
              lastMessage.content.substring(0, 60) + '...' :
              lastMessage.content) :
            '',
          createdAt: session.createdAt,
          updatedAt: session.updatedAt,
          messageCount: session.messageCount,
        });
      }
    });

    return results.sort((a, b) =>
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
  }

  // Get session count
  async getSessionCount(): Promise<number> {
    await this.initialize();
    return this.cachedSessions.size;
  }
}

export const chatStorageService = new ChatStorageService();
