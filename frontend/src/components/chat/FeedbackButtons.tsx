import { useState } from 'react'
import { ThumbsUp, ThumbsDown } from 'lucide-react'
import { api } from '@/lib/api'
import { toast } from '@/components/ui/toaster'
import { cn } from '@/lib/utils'

interface FeedbackButtonsProps {
  conversationId: number
}

export function FeedbackButtons({ conversationId }: FeedbackButtonsProps) {
  const [voted, setVoted] = useState<'positive' | 'negative' | null>(null)

  const handleFeedback = async (sentiment: 'positive' | 'negative') => {
    if (voted) return
    try {
      await api.feedback({ conversation_id: conversationId, sentiment })
      setVoted(sentiment)
      toast({
        variant: 'success',
        title: sentiment === 'positive' ? '👍 Helpful' : '👎 Not helpful',
        description: 'Thank you for your feedback.',
      })
    } catch {
      toast({ variant: 'destructive', title: 'Error', description: 'Failed to save feedback.' })
    }
  }

  return (
    <div className="mt-2 flex items-center gap-1">
      <button
        onClick={() => handleFeedback('positive')}
        disabled={!!voted}
        className={cn(
          'flex h-7 w-7 items-center justify-center rounded-md border border-border text-muted transition-all duration-200',
          voted === 'positive'
            ? 'border-accent text-accent bg-accent/10'
            : 'hover:border-accent hover:text-accent hover:bg-accent/5',
          voted && voted !== 'positive' && 'opacity-30',
        )}
        title="Helpful"
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </button>
      <button
        onClick={() => handleFeedback('negative')}
        disabled={!!voted}
        className={cn(
          'flex h-7 w-7 items-center justify-center rounded-md border border-border text-muted transition-all duration-200',
          voted === 'negative'
            ? 'border-red-500 text-red-400 bg-red-500/10'
            : 'hover:border-red-500 hover:text-red-400 hover:bg-red-500/5',
          voted && voted !== 'negative' && 'opacity-30',
        )}
        title="Not helpful"
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

