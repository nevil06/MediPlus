// Groq AI Service for MediBot
import { API_CONFIG, MEDIBOT_SYSTEM_PROMPT } from '../constants/config';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface ChatResponse {
  success: boolean;
  message?: string;
  error?: string;
}

class GroqService {
  private apiKey: string;
  private baseUrl: string = 'https://api.groq.com/openai/v1/chat/completions';

  constructor() {
    this.apiKey = API_CONFIG.GROQ_API_KEY;
  }

  setApiKey(key: string) {
    this.apiKey = key;
  }

  async sendMessage(
    userMessage: string,
    conversationHistory: Message[] = []
  ): Promise<ChatResponse> {
    try {
      const messages: Message[] = [
        { role: 'system', content: MEDIBOT_SYSTEM_PROMPT },
        ...conversationHistory,
        { role: 'user', content: userMessage },
      ];

      const response = await fetch(this.baseUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: 'llama-3.3-70b-versatile',
          messages: messages,
          temperature: 0.7,
          max_tokens: 1024,
          top_p: 1,
          stream: false,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error?.message || 'Failed to get response');
      }

      const data = await response.json();
      const assistantMessage = data.choices[0]?.message?.content;

      if (!assistantMessage) {
        throw new Error('No response from AI');
      }

      return {
        success: true,
        message: assistantMessage,
      };
    } catch (error) {
      console.error('Groq API Error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  }

  async sendMessageWithReasoning(
    userMessage: string,
    conversationHistory: Message[] = []
  ): Promise<ChatResponse> {
    try {
      const messages: Message[] = [
        { role: 'system', content: MEDIBOT_SYSTEM_PROMPT },
        ...conversationHistory,
        { role: 'user', content: userMessage },
      ];

      const response = await fetch(this.baseUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: 'deepseek-r1-distill-llama-70b',
          messages: messages,
          temperature: 0.6,
          max_tokens: 4096,
          top_p: 1,
          stream: false,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error?.message || 'Failed to get response');
      }

      const data = await response.json();
      const assistantMessage = data.choices[0]?.message?.content;

      if (!assistantMessage) {
        throw new Error('No response from AI');
      }

      return {
        success: true,
        message: assistantMessage,
      };
    } catch (error) {
      console.error('Groq API Error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  }
}

export const groqService = new GroqService();
export type { Message, ChatResponse };
