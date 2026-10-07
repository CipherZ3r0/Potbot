import { Lock, ArrowRight } from 'lucide-react'

const SUGGESTIONS = [
  'Summarize the recent engineering guidelines.',
  'How do I setup a local database model?',
  'What are the main security protocols for internal documents?',
]

interface EmptyStateProps {
  onSuggestionClick: (text: string) => void
}

export function EmptyState({ onSuggestionClick }: EmptyStateProps) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-4 py-12">
      {/* Icon */}
      <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-accent/10">
        <Lock className="h-7 w-7 text-accent" />
      </div>

      {/* Title & subtitle */}
      <h2 className="mb-3 text-xl font-semibold text-foreground">
        Secure Enterprise RAG
      </h2>
      <p className="mb-8 max-w-md text-center text-sm text-muted leading-relaxed">
        Ask questions across your internal documents. Fully private embeddings and
        execution ensure no data leaks.
      </p>

      {/* Suggested prompts */}
      <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted">
        Suggested Prompts
      </p>
      <div className="grid w-full max-w-xl gap-2 sm:grid-cols-3">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onSuggestionClick(s)}
            className="flex min-h-[64px] items-center justify-between gap-2 rounded-xl border border-border bg-card p-3 text-left text-xs text-foreground transition-all duration-200 hover:border-accent hover:bg-card-hover group"
          >
            <span className="leading-relaxed">{s}</span>
            <ArrowRight className="h-3.5 w-3.5 shrink-0 text-muted group-hover:text-accent transition-colors" />
          </button>
        ))}
      </div>
    </div>
  )
}

