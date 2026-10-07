import { useState } from 'react'
import { FolderOpen, RotateCcw, Loader2, CheckCircle2 } from 'lucide-react'
import { api } from '@/lib/api'
import { toast } from '@/components/ui/toaster'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

export function FolderIngestTab() {
  const [path, setPath] = useState('')
  const [rebuild, setRebuild] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleIngest = async () => {
    if (!path.trim()) {
      toast({ variant: 'destructive', title: 'No path', description: 'Please enter a folder path.' })
      return
    }
    setLoading(true)
    try {
      const result = await api.ingestFolder(path.trim(), rebuild)
      if (result.indexed_count > 0) {
        toast({
          variant: 'success',
          title: 'Ingestion complete',
          description: `Indexed ${result.indexed_count} chunks from ${result.doc_count} document(s).`,
        })
      } else {
        toast({
          variant: 'default',
          title: result.status === 'all_skipped' ? 'All files up-to-date' : 'Nothing indexed',
          description: `Status: ${result.status}`,
        })
      }
    } catch (err) {
      toast({
        variant: 'destructive',
        title: 'Ingestion failed',
        description: err instanceof Error ? err.message : 'Unknown error',
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2">
        <FolderOpen className="h-4 w-4 shrink-0 text-muted" />
        <Input
          value={path}
          onChange={(e) => setPath(e.target.value)}
          placeholder="/path/to/documents"
          className="border-0 bg-transparent p-0 text-xs focus-visible:ring-0 h-auto"
          onKeyDown={(e) => e.key === 'Enter' && handleIngest()}
        />
      </div>

      <label className="flex items-center gap-2 text-xs text-muted cursor-pointer select-none">
        <input
          type="checkbox"
          checked={rebuild}
          onChange={(e) => setRebuild(e.target.checked)}
          className="accent-accent"
        />
        <RotateCcw className="h-3 w-3" />
        Rebuild index
      </label>

      <Button
        onClick={handleIngest}
        disabled={loading || !path.trim()}
        className="w-full"
        variant={path.trim() ? 'accent' : 'default'}
      >
        {loading ? (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Processing…
          </>
        ) : (
          <>
            <CheckCircle2 className="h-3.5 w-3.5" />
            Ingest Folder
          </>
        )}
      </Button>
    </div>
  )
}

