import type { SourceItem } from '@/types'
import { FileText } from 'lucide-react'
import { useState } from 'react'
import { cn } from '@/lib/utils'

interface SourceCardProps {
  sources: SourceItem[]
}

export function SourceCard({ sources }: SourceCardProps) {
  const [open, setOpen] = useState(false)

  if (!sources.length) return null

  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-xs text-muted hover:text-foreground transition-colors"
      >
        <FileText className="h-3.5 w-3.5" />
        <span>{open ? 'Hide' : 'Show'} sources</span>
        <span className="ml-1 rounded-full bg-border px-1.5 py-0.5 text-[10px] font-medium">
          {sources.length}
        </span>
      </button>

      {open && (
        <div className="mt-2 space-y-2">
          {sources.map((src, i) => (
            <div
              key={i}
              className="rounded-md border border-border bg-card p-3 text-xs"
            >
              <span className={cn('block font-medium text-accent mb-1 truncate')}>
                {src.file_name}
                {src.page_number != null && (
                  <span className="ml-2 text-muted font-normal">
                    Page {src.page_number}
                  </span>
                )}
              </span>
              <p className="text-muted leading-relaxed line-clamp-3">{src.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

