import { useEffect, useRef } from 'react'
import type { Message, Settings } from '@/types'
import { ChatMessage } from './ChatMessage'
import { ChatInput } from './ChatInput'
import { EmptyState } from './EmptyState'

interface ChatAreaProps {
  messages: Message[]
  isLoading: boolean
  settings: Settings
  onSend: (query: string) => void
}

export function ChatArea({ messages, isLoading, onSend }: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      {/* Messages scroll area */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-4 md:px-8">
        <div className="mx-auto max-w-3xl">
          {messages.length === 0 ? (
            <EmptyState onSuggestionClick={onSend} />
          ) : (
            <>
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}
            </>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input bar — pinned to bottom */}
      <div className="border-t border-border bg-background px-4 py-4 md:px-8">
        <div className="mx-auto max-w-3xl">
          <ChatInput onSend={onSend} disabled={isLoading} />
          <p className="mt-2 text-center text-[11px] text-muted">
            potbot may make mistakes. Always verify critical information from source documents.
          </p>
        </div>
      </div>
    </div>
  )
}

