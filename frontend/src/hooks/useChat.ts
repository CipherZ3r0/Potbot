import { useState, useCallback } from 'react'
import { api } from '@/lib/api'
import type { Message, Settings } from '@/types'

const generateId = () => Math.random().toString(36).slice(2, 11)

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const sendMessage = useCallback(
    async (query: string, settings: Settings) => {
      const userMsg: Message = {
        id: generateId(),
        role: 'user',
        content: query,
      }

      const assistantPlaceholder: Message = {
        id: generateId(),
        role: 'assistant',
        content: '',
        isLoading: true,
      }

      setMessages((prev) => [...prev, userMsg, assistantPlaceholder])
      setIsLoading(true)

      try {
        const response = await api.chat({
          query,
          retrieval_method: settings.retrieval_method,
          use_reranking: settings.use_reranking,
          use_query_rewriting: settings.use_query_rewriting,
          prompt_style: settings.prompt_style,
          llm_provider: settings.provider,
          llm_model: settings.model,
        })

        const assistantMsg: Message = {
          id: assistantPlaceholder.id,
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
          conversationId: response.conversation_id ?? undefined,
          responseTimeMs: response.response_time_ms,
          totalTokens: response.total_tokens,
          model: response.model,
        }

        setMessages((prev) =>
          prev.map((m) => (m.id === assistantPlaceholder.id ? assistantMsg : m)),
        )
      } catch (err) {
        const errorMsg: Message = {
          id: assistantPlaceholder.id,
          role: 'assistant',
          content: err instanceof Error ? err.message : 'An unexpected error occurred.',
          isError: true,
        }
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantPlaceholder.id ? errorMsg : m)),
        )
      } finally {
        setIsLoading(false)
      }
    },
    [],
  )

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return { messages, isLoading, sendMessage, clearMessages }
}

