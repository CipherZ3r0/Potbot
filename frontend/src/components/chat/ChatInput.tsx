import { useState, useRef, KeyboardEvent } from 'react'
import { Send, Square } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ChatInputProps {
  onSend: (query: string) => void
  disabled?: boolean
  placeholder?: string
}

export function ChatInput({ onSend, disabled, placeholder = 'Message potbot…' }: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInput = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }

  return (
    <div className="relative flex items-end gap-2 rounded-xl border border-border bg-card p-3 transition-colors focus-within:border-accent focus-within:ring-1 focus-within:ring-accent/20">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => {
          setValue(e.target.value)
          handleInput()
        }}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
        className={cn(
          'flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted focus:outline-none leading-relaxed',
          'min-h-[24px] max-h-[200px] scrollbar-thin',
        )}
      />
      <button
        onClick={handleSend}
        disabled={!value.trim() || disabled}
        className={cn(
          'flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-all duration-200',
          value.trim() && !disabled
            ? 'bg-accent text-white hover:bg-accent-hover'
            : 'bg-border text-muted cursor-not-allowed',
        )}
        title="Send (Enter)"
      >
        {disabled ? (
          <Square className="h-3.5 w-3.5 fill-current" />
        ) : (
          <Send className="h-3.5 w-3.5" />
        )}
      </button>
    </div>
  )
}

