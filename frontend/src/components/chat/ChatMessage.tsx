import ReactMarkdown from 'react-markdown'
import type { Message } from '@/types'
import { SourceCard } from './SourceCard'
import { FeedbackButtons } from './FeedbackButtons'
import { cn } from '@/lib/utils'

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

  return (
    <div
      className={cn(
        'group flex gap-3 py-4 animate-fade-in',
        isUser ? 'flex-row-reverse' : 'flex-row',
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          'flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold',
          isUser
            ? 'bg-border text-foreground'
            : 'bg-accent text-white',
        )}
      >
        {isUser ? 'U' : 'PB'}
      </div>

      {/* Bubble */}
      <div className={cn('flex flex-col max-w-[85%]', isUser ? 'items-end' : 'items-start')}>
        <div
          className={cn(
            'rounded-xl px-4 py-3 text-sm leading-relaxed',
            isUser
              ? 'bg-card border border-border text-foreground'
              : message.isError
              ? 'bg-red-900/20 border border-red-800 text-red-300'
              : 'text-foreground',
          )}
        >
          {message.isLoading ? (
            <div className="flex items-center gap-2 text-muted">
              <span className="inline-flex gap-1">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="h-1.5 w-1.5 rounded-full bg-muted animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </span>
              <span className="text-xs">Generating…</span>
            </div>
          ) : (
            <ReactMarkdown
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                code: ({ children, className }) => {
                  const isBlock = className?.includes('language-')
                  return isBlock ? (
                    <code className="block bg-background rounded p-2 text-xs font-mono overflow-x-auto my-2">
                      {children}
                    </code>
                  ) : (
                    <code className="bg-background rounded px-1 py-0.5 text-xs font-mono">
                      {children}
                    </code>
                  )
                },
                ul: ({ children }) => (
                  <ul className="list-disc pl-4 mb-2 space-y-1">{children}</ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal pl-4 mb-2 space-y-1">{children}</ol>
                ),
                strong: ({ children }) => (
                  <strong className="font-semibold text-foreground">{children}</strong>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {/* Metadata + Sources + Feedback (assistant only) */}
        {!isUser && !message.isLoading && !message.isError && (
          <>
            {message.responseTimeMs != null && (
              <p className="mt-1.5 text-[11px] text-muted">
                {message.responseTimeMs}ms · {message.totalTokens ?? 0} tokens
              </p>
            )}

            {message.sources && message.sources.length > 0 && (
              <SourceCard sources={message.sources} />
            )}

            {message.conversationId != null && (
              <FeedbackButtons conversationId={message.conversationId} />
            )}
          </>
        )}
      </div>
    </div>
  )
}

